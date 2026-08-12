# Prefetching, reconsidered

This project removed a predictor and concluded that prediction does not work. That conclusion
was too broad. This branch re-examines it, and the short answer is: **prefetching works in the
regime the original was never tested in, and fails in the regime it was.**

## What the original got wrong

**It optimised the wrong metric.** The removed predictor was tuned and reported on *recall* -
"does my top-N contain the experts this token needs?" - where it scored 68-77% against LRU's
35%. But recall rewards fetching *more*. The quantity that decides whether prefetching pays is
**precision**: of the experts fetched, what fraction get used. Optimising recall is exactly how
you end up moving a superset, which is the mechanism that killed it.

**It was measured in the wrong regime.** At the ~60% hit rates it was tested at, the PCIe bus is
already saturated with demand traffic, so every speculative byte displaces a byte that was
genuinely needed. That is not a property of prediction; it is a property of a full bus.

## Method

`bells_precision.py` replays a trace through the real cache, builds a token-id frequency table
on the first 60% of tokens, and evaluates on the held-out 40%. A prefetch is issued only when
the predictor's confidence for that expert clears a threshold - the knob that trades recall for
precision. Prefetched experts are admitted like any other and can evict, so a bad guess costs
something real.

Qwen3-30B-A3B, 128 experts, 8 active, 2.92 MB each, PCIe measured at 7.09 GB/s.

## At a 2x cache ratio, it fails - exactly as before

| threshold | precision | demand traffic | prefetch traffic | bus |
|---|---|---|---|---|
| none | - | 557 MB | 0 | 87% |
| 0.90 | 65.1% | 421 MB | 214 MB | **100%** |
| 0.70 | 63.3% | 399 MB | 258 MB | 103% |
| 0.50 | 50.2% | 370 MB | 402 MB | 121% |

Demand traffic does drop - 24% at the strictest threshold - but the bus is already 87% busy, so
the speculative traffic pushes it to saturation and beyond. Past 100% the prefetches are
competing with the transfers they were meant to avoid. **This is the original failure, and it is
a bandwidth argument rather than an argument about predictors.**

## At an 8x cache ratio, it works

| threshold | precision | demand traffic | prefetch traffic | bus |
|---|---|---|---|---|
| none | - | 121 MB | 0 | 29% |
| **0.90** | **73.3%** | **87 MB** | 49 MB | **32%** |
| 0.70 | 70.5% | 82 MB | 59 MB | 33% |
| 0.50 | 57.7% | 77 MB | 84 MB | 38% |

A high hit rate leaves the bus mostly idle, and speculative traffic then costs spare capacity
rather than displacing anything. At a 0.90 threshold, **28.5% of demand traffic disappears while
the bus stays under a third busy.** Precision is 73%, so three of every four prefetched experts
are used - a very different picture from the superset-fetching the old design did.

## How much is it worth

Less than the traffic reduction suggests, because at a high hit rate transfers are no longer the
bottleneck. At 8x, demand traffic is ~29% of decode time, so removing 28.5% of it is roughly an
**8% speedup**. That number is an estimate twice over: bytes converted to time by measured
bandwidth, against a decode time that was assumed rather than benchmarked.

## Where this sits

It is the mirror image of per-layer gating, which is the other idea currently alive:

| | works when | effect |
|---|---|---|
| per-layer gating | low ratio, bus saturated | 1.14x, **measured** over five paired passes |
| confidence prefetch | high ratio, bus idle | ~8%, **estimated** from byte counts |

Gating removes the failure cases; prefetching raises the ceiling. They do not overlap, and
neither helps where the other does.

## Built and measured: correct, and still slower

Implemented rather than left as a projection. `bells_conf` loads a confidence table,
`begin_ubatch` prefetches every candidate above the threshold, and `bells_build_conf.py` builds
the table from a trace. Qwen3-30B-A3B, 64 slots (8x), two paired passes:

| | hit | moved | prefetched | perplexity | decode |
|---|---|---|---|---|---|
| no prefetch | 93.2% | 26.14 GiB | 0 | **13.8380** | 126.7 / 109.8 ms |
| prefetch 0.90 | 94.8% | 26.37 GiB | 2184 | **13.8380** | 148.9 / 119.0 ms |
| prefetch 0.70 | 95.5% | 26.83 GiB | 3313 | **13.8380** | 125.2 / 130.3 ms |

**Lossless, confirmed.** Perplexity is identical to four decimals across all six runs. Prefetch
changes which experts are resident, never what gets computed - `ensure()` still guarantees
residency before any matmul reads the cache.

**The prediction was right.** Offline this said ~27% of misses would be removed; at 93.2% hit
that is 6.8% miss dropping to ~5%, i.e. ~95% hit. Measured: 94.8% and 95.5%. The predictor does
what it claims, at the precision it claims.

**And it is slower**: mean 118 ms becomes 134 ms at 0.90 and 128 ms at 0.70.

The cause is plumbing, not prediction. Prefetch copies are issued **synchronously in
`begin_ubatch`, for every layer at once, before the token computes anything**. They are a serial
prologue rather than overlapped work, so the bus headroom the whole argument rests on is never
used - the transfers are simply extra latency in front of each token. The risk was written into
the section below before the code was built, and then not designed around.

**The two dead ends interlock.** Prefetching needs genuinely asynchronous transfers; async
transfers need a pinned source; pinned *staging* measured 37% slower because it duplicates the
copy the driver already makes internally. The only route left is `cudaHostRegister` on the
model's pages, which pins them against swap.

So: **correct and unbuildable on the current transfer path.** Not refuted, blocked - which is a
more useful state than the "prediction does not work" this project believed before, and it names
the one change that would unblock it.

## The root cause: transfers cannot overlap compute at all

The obvious response to "prefetch is a serial prologue" is to move the copies to a background
thread, so the main thread never waits. Built that too - a worker thread taking the copies while
cache bookkeeping stays on the main thread, with a per-layer gate so nothing reads a slot being
written. Correct (perplexity still 13.8380) and **marginally worse**:

| | synchronous | background thread |
|---|---|---|
| no prefetch | 118.3 ms | 121.7 ms |
| prefetch 0.90 | 133.9 ms | 143.9 ms |
| prefetch 0.70 | 127.7 ms | 138.0 ms |

The reason is in ggml, at `ggml-cuda.cu:2991`:

```c
static void ggml_backend_cuda_set_tensor_async(...) {
    CUDA_CHECK(cudaMemcpyAsync(..., cudaMemcpyHostToDevice, cuda_ctx->stream()));
}
```

**Host-to-device copies are issued on the same CUDA stream as the compute graph.** Streams are
strictly ordered, so a copy cannot execute alongside a kernel - it waits for the one in front of
it. Which CPU thread enqueued it is irrelevant; they all feed one ordered queue.

That single fact explains every failure in these notes and in OPTIMIZATION.md:

| attempt | outcome | explanation |
|---|---|---|
| pinned staging ring | 37% slower | extra work, still serialised |
| prefetch, synchronous | slower | serial prologue in front of the token |
| prefetch, background thread | slower | same stream regardless of thread |
| copies are 87% of layer cost | - | never hidden behind anything |

**The prerequisite for any of this is a second stream.** A second `ggml_backend_cuda_init` on
the same device gets its own stream; copies go there, and `ggml_backend_event_record` /
`ggml_backend_event_wait` order them against compute only where they must be.

### Built the second stream. It changes nothing, and that is the useful part.

Implemented: copies issued on a second CUDA backend, ordered against the graph with an event so
the dependency is stream-level rather than a host stall. Three paired passes at 17 slots:

| | copy/layer | perplexity | decode |
|---|---|---|---|
| one stream | 1083.9 / 1097.2 / 1091.7 us | 13.8380 | 76.38 / 77.49 / 76.63 ms |
| two streams | 1092.4 / 1103.4 / 1099.1 us | 13.8380 | 76.48 / 78.11 / 76.70 ms |

Identical within noise. Correct, and pointless.

**The copy timer is what gives it away.** It measures *host* time inside `copy_expert`, and it
stays at ~1090 us with or without a second stream. A genuinely asynchronous copy would leave
that near zero and do the work on the device. **The host is blocking for the full duration of
every transfer**, so stream ordering was never the binding constraint - it was the second half
of the problem, not the whole of it:

> `cudaMemcpyAsync` from **pageable** memory is not asynchronous. The driver stages it through
> an internal pinned buffer and the call blocks the caller.

A second stream cannot fix a host-side stall. Nor could the background copy thread (for demand
copies `on_routing` must finish before the layer proceeds). Nor could a staging buffer of our
own, which merely duplicates the copy the driver already makes.

**The two fixes are only useful together, and all the testing so far has been one at a time:**

| | alone | why it failed |
|---|---|---|
| pinned source (own staging) | 37% slower | duplicated the driver's internal copy |
| second stream | no change | the host still blocks |
| `cudaHostRegister` + second stream | **untested** | removes the stall *and* allows overlap |

Only registering the model's pages in place removes the host stall while leaving a copy that is
actually async, which is the thing a second stream can then overlap. That pins pages against
swap, so it needs a machine with RAM to spare - the 128 GB cloud instances, not a 32 GB desktop
holding a 17 GB model.

Worth one caution against over-claiming: on the configuration measured here, transfers (74.5 ms
per token) exceed GPU compute (~16 ms), so perfect overlap would still leave transfers dominant.
A second stream helps most where the hit rate is high and transfers are small - which is the same
regime this whole document is about.

## The precision above is in-distribution, and that matters

Everything above was measured by training the table on the first 60% of a trace and evaluating
on the remaining 40% **of the same trace**. Held-out tokens, but not held-out *text*. Since the
entire premise is that routing is conditioned on token id, a table counted on one kind of text
will flatter itself on that same kind of text.

A 205k-token trace collected over 54 markdown files followed by 171 C/C++ files puts a domain
boundary at record 152883. Training on records 0..91729 (prose only) and scoring the same table
on both sides of that boundary separates the two:

| | precision @0.90 | demand traffic removed | baseline demand |
|---|---|---|---|
| held-out **prose** (in-distribution) | **82.2%** | 30.0% | 119.1 MB/token |
| held-out **code** (cross-domain) | **67.9%** | 22.2% | 65.9 MB/token |

**Precision falls 14.3 points, from 82.2% to 67.9%.** The predictor degrades under domain shift
but does not collapse - two of every three prefetched experts are still used. Token-id routing
statistics are a real effect, not an artifact of a narrow trace.

Three cautions on reading this table:

- **The in-distribution figure is optimistic even for in-distribution use.** The prose evaluation
  window is directly adjacent to the training window - same documents, same section of `docs/`.
  It shares local vocabulary and topic, so 82.2% is an upper bound flattered by locality, not
  just by domain. There is no equally-distant prose window to control against, because the prose
  is contiguous.
- **"Traffic removed" is not comparable across the two rows.** The code region has a much lower
  intrinsic miss rate - 65.9 MB/token against 119.1 - so LRU alone already does well there and
  there is simply less to win. That column conflates domain shift with a different baseline.
  Precision is the clean comparison; traffic is not.
- **This is a mild shift.** Technical prose to C++, both from one repository, both English-
  adjacent. It is evidence, not proof. Fiction, dialogue or non-English text would be the real
  test, and has not been run.

Note also what this does to the 73.3% quoted earlier in this document: a larger table (6980
tokens rather than 605) raises in-distribution precision to 82.2%, but the number that
generalises is 67.9% - **below** the figure the earlier sections were built on. A bigger table
buys accuracy within a domain and does not buy it across domains.

## Before anyone builds this

- The 8% is an estimate, and this project's estimates have a poor record. Four of five
  optimizations that looked certain died on measurement, including one that was implemented and
  came back 37% slower.
- The predictor costs CPU time per token per layer, which is not in the model above.
- Prefetches must be issued early enough and asynchronously enough to actually overlap compute.
  The current copy path is synchronous from pageable memory, so this is not free plumbing.
- Only one model and one trace. The 0.90 threshold that works here is not necessarily the right
  one elsewhere.
- Restoring a predictor means restoring the table-building tooling that was deleted with it.

## Reproducing

```
python tools/bells-profile/bells_precision.py models/bells/qwen3-big.trace.bin \
    --mult 8 --expert-mb 2.92 --decode-ms 60
```

`--mult` is the cache ratio and is the variable that decides the answer.
