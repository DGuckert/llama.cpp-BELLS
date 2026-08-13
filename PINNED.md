# The host stall, and the 1.80x hiding behind it

Every optimisation in `PREFETCH.md` and `OPTIMIZATION.md` that failed, failed for one reason.
This is that reason, measured.

## The mechanism

`cudaMemcpyAsync` from **pageable** host memory is not asynchronous. The driver stages the
transfer through its own internal pinned buffer and blocks the calling thread for the duration.
Measured in isolation with `bells_h2d.cu` - same bytes, same block size, same stream, only the
source memory differs:

| block | source | bandwidth | host time inside memcpy | |
|---|---|---|---|---|
| 1.09 MB | pageable | 8.55 GB/s | 26.66 ms of 26.73 ms | **blocks** |
| 1.09 MB | pinned | 12.29 GB/s | 0.88 ms of 18.60 ms | async, **1.44x** |
| 2.92 MB | pageable | 11.17 GB/s | 54.75 ms of 54.84 ms | **blocks** |
| 2.92 MB | pinned | 12.57 GB/s | 0.82 ms of 48.75 ms | async, 1.12x |
| 11.3 MB | pageable | 11.05 GB/s | 214.39 ms of 214.46 ms | **blocks** |
| 11.3 MB | pinned | 12.58 GB/s | 0.85 ms of 188.45 ms | async, 1.14x |

**A pageable copy holds the caller for 99.7% of the transfer. A pinned one holds it for 4.7%.**

Block size decides how much bandwidth is on the table, and BELLS lands in the good part of the
curve by accident: it copies `gate`, `up` and `down` as three separate ~1 MB tensors rather than
one 2.92 MB expert. Pageable at 1.09 MB measures 8.55 GB/s against the 8.2 GB/s BELLS actually
achieves - close enough, derived independently, to confirm the model.

## End to end

OLMoE-1B-7B, 3.9 GB, 16 slots, three passes:

| | copy/layer | layer total | decode | perplexity |
|---|---|---|---|---|
| pageable | 1,925 us | 13,219 ms | 31.35 ms | 21.4760 |
| **pinned** | **31 us** | **1,553 ms** | **17.36 ms** | 21.4760 |

**Host time in copy falls 62x. Layer time falls 8.5x. Decode is 1.80x faster.** Bytes moved are
identical at 67.96 GiB and perplexity is identical to four decimals - the same work, the same
output, the host simply stops waiting for it. Variance across passes is under 1%.

And on the model this project actually benchmarks - Qwen3-30B-A3B, 17 slots, three paired
passes, with the chunked allocation below in place:

| | copy/layer | layer total | decode | tok/s |
|---|---|---|---|---|
| pageable | 1,399 us | 27,862 ms | 94.34 ms | 10.62 |
| **pinned** | **166 us** | **6,939 ms** | **59.42 ms** | **16.83** |

**1.59x**, perplexity identical at 13.8380 across all six runs, and that is with only 94% of the
weight pinned - a 1012 MiB tail still falls back. Pinned decode varies 3% across passes against
10% for pageable, which is itself a result: removing the stall removes most of the sensitivity to
whatever else the machine is doing.

## Why this took so long to find

Because it explains the failures rather than announcing itself. From these notes:

| attempt | outcome | actual reason |
|---|---|---|
| pinned staging ring | 37% slower | duplicated the driver's internal copy |
| prefetch, synchronous | slower | serial prologue, host blocked throughout |
| prefetch, background thread | slower | worker thread blocked instead of main |
| second CUDA stream | **no change** | stream ordering was never the constraint |

The second stream is the informative one. It was built correctly, ordered against the graph with
events, and changed nothing - because a second stream cannot help a host that is blocked. That
result was read at the time as "transfers cannot overlap compute"; it actually meant "the host
never gets far enough ahead to overlap anything".

## What blocked it on a 32 GB machine

```
ggml_cuda_host_malloc: failed to allocate 16740.00 MiB of pinned memory: out of memory
```

Windows will not hand out a 16.7 GB pinned block on a 32 GB desktop, and `ggml-cuda.cu` **silently
fell back to a plain CPU buffer**. The diagnostic was `GGML_LOG_DEBUG`, so at normal verbosity a
user asked for pinned memory, received pageable, and was told nothing. Every measurement taken
inside llama.cpp before this was therefore pageable against pageable, which is exactly why they
all tied.

**But the limit is mostly per-allocation, not total.** Requesting the same amount in pieces:

| chunk | total pinned | | chunk | total pinned |
|---|---|---|---|---|
| single 17.6 GB | **failed** | | 1024 MB | **16.1 GB** |
| 4096 MB | 12.9 GB | | 512 MB | 16.1 GB |
| 2048 MB | 15.0 GB | | 256 MB | 16.1 GB |

So the CUDA host buffer type now declares a 1 GiB `get_max_size`, which is where the returns
flatten. `ggml_backend_alloc_ctx_tensors_from_buft` splits on that, the per-allocation fallback
then degrades only the tail that no longer fits, and 94% of a 17.55 GB model pins on a machine
that could previously pin none of it. `GGML_CUDA_PINNED_CHUNK_MB` overrides the size.

A single tensor must still fit inside one chunk - `ggml_backend_alloc_ctx_tensors_from_buft`
treats a larger one as an error - but expert tensors are 110-160 MiB, far under.

The fallback also warns at normal log level now. A silent downgrade of the thing that turns out
to matter most is a bad default, and it cost an afternoon here.

## Getting at it

`common/arg.cpp` now exposes the pinned host buffer type to `-ot`, which previously could name
every buffer type except the one that matters for tensors deliberately kept on the CPU:

```
llama-bells-profile -m model.gguf -ngl 99 --no-mmap \
    -ot "\.ffn_(gate|up|down)_exps\.=CUDA_Host"
```

`--no-mmap` is required: `llama-model-loader.cpp:1191` replaces a host buffer type with plain CPU
whenever mmap is on, so an mmap'd model can never have pinned weights.

Note this replaces `--cpu-moe` rather than accompanying it - both push a buffer override for the
same tensors, and the first match wins.

## What is left afterwards, and where

With copies pinned, the readback becomes the larger per-layer item. Sweeping the cache size on
Qwen3-Next-80B - same model, same graph, miss rate varying 5.7x - separates fixed overhead from
time spent waiting on the GPU:

| slots | hit | readback | copy | ms/token |
|---|---|---|---|---|
| 16 | 53.8% | 30.6 us | 26.1 us | 27.13 |
| 64 | 78.0% | 30.5 us | 12.5 us | 21.59 |
| 192 | 91.0% | 30.3 us | 5.3 us | 19.27 |
| **256** | **92.2%** | 30.1 us | 4.6 us | **18.99 (52.66 tok/s)** |

**Readback is flat to within 2% while copy falls sixfold.** It is fixed synchronisation
overhead, not GPU wait - which is what makes it worth attacking in principle, and what makes it
the barrier stopping a second stream from finding anything to overlap.

**But the magnitude is platform-dependent, and that matters more than the mechanism.** Same
model, same slot counts, only the GPU and the operating system differing:

| slots | A10G / Linux | RTX 2060 / Windows | |
|---|---|---|---|
| 16 | 30.6 us | 279.3 us | **9.1x** |
| 32 | 30.6 us | 274.0 us | 9.0x |
| 48 | ~30.5 us | 268.4 us | 8.8x |

Flat on both, so it is fixed overhead in either case - but an order of magnitude larger under
Windows. Across 48 layers that is 1.4 ms of a 19 ms token on Linux, about 6%, against 13 ms of
a 48 ms token on Windows, about **27%**.

The cause is almost certainly WDDM, whose kernel-launch and synchronisation latency exceeds
Linux's by a wide margin. A GPU difference cannot plausibly produce 9x in a device sync, and the
ratio holds across three independent slot counts.

**So the cheapest remaining optimisation on a Windows machine is to stop running Windows.** No
code, no hardware, roughly a fifth to a quarter of a token.

One caution about the Windows run those figures come from: its *copy* column measured 448.5,
369.4 and 302.9 us, which works out at 10.9-12.0 GB/s - exactly the blocking PCIe 3.0 rate. The
pinned allocation had failed (a 27.2 GB model against 32 GB of RAM, with less free than during
the successful run earlier), so that column reflects pageable behaviour. Readback does not
depend on pinning and is unaffected.

So on Linux, BELLS' entire per-layer overhead is ~11% of the token at high slot counts and
there is little left to collect. On Windows it is worth about a fifth, and **the cheapest way to
collect it is to run Linux, not to write code.**

Note also 52.66 tok/s at 256 slots, with copy at 4.6 us per layer. On that configuration
transfers have been optimised out of relevance rather than merely reduced.

## Where this leaves things

The mechanism is confirmed, the integration is proven, and the remaining question is only how
much of a large model can be pinned at once. On 128 GB the answer should be "all of it".
