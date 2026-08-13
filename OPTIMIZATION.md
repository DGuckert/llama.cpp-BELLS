# Where the remaining performance is

Research notes on the `bells-opt` branch. Everything here follows from one measurement: on
Qwen3-30B-A3B at 17 slots, the per-layer cost splits as

| | time | share |
|---|---|---|
| readback (device -> host sync) | 167.1 us | 11% |
| **copy (host -> device experts)** | **1356.3 us** | **87%** |
| upload (slot table) | 28.7 us | 2% |

Across 48 layers that is 74.5 ms of a 90.84 ms decode. **Transfers are the system.** Anything
that does not reduce bytes moved, or make a miss stop requiring a transfer, is rounding error.

There are only three ways to make transfers cost less:

1. miss less often - **tested, dead end.** Seven policies, two architectures, nothing beats LRU.
2. transfer more cheaply - **staging tried and reverted, 37% slower.** One route remains
   (registering the model's pages), with a real hazard attached.
3. skip the low-value transfers - **tested, dead end.** The router's weights are nearly flat, so
   no threshold saves transfers without discarding real probability mass.
4. **cache only the layers where caching pays** - the most promising thing here, and the
   cheapest to build: a third of layers sit below break-even at a 2x ratio, and the graph
   already gates per layer.
5. stop transferring on a miss at all - the largest prize, mechanism verified, design untested.

Four of the six things tried are closed by measurement rather than argument, and **all four were
things I expected to work**: eviction looked free, pinned staging had a 1.73x bandwidth gap
apparently sitting there, low-weight experts looked obviously skippable, and uneven slot
allocation looked certain given a 58-point spread between layers. None survived a paired
measurement. That ratio is the honest prior for (4) and (5) too.

---

## 1. Eviction policy is not a lever

If a better eviction rule raised the hit rate it would be free: choosing a different victim adds
no PCIe traffic. That distinguishes it from the predictor this project already removed, which
raised hit rate by *prefetching* - moving a superset of what was needed, and losing on wall
clock as a result.

Seven policies, two architectures, replayed through the real admission semantics
(`tools/bells-profile/bells_policy.py`):

**Qwen3-Next, 128 experts, 8 active, 4 layers sampled**

| ratio | LRU | LRU-2 | rank-LRU | LFU | LFU aged | static | hybrid | **Belady** |
|---|---|---|---|---|---|---|---|---|
| 2x | 52.6 | 53.3 | 51.6 | 52.5 | 52.5 | 32.5 | 51.5 | **66.7** |
| 4x | **70.3** | 69.9 | 66.6 | 67.9 | 67.8 | 51.2 | 69.8 | **82.6** |
| 8x | 87.9 | 87.3 | 86.3 | 87.4 | 87.4 | 79.7 | **88.6** | **93.2** |

**OLMoE, 64 experts, 8 active, 4 layers sampled**

| ratio | LRU | LRU-2 | rank-LRU | LFU | LFU aged | static | hybrid | **Belady** |
|---|---|---|---|---|---|---|---|---|
| 2x | 61.9 | 63.2 | 62.3 | **64.5** | **64.5** | 53.6 | 63.7 | **75.3** |
| 4x | 82.2 | 82.3 | 82.0 | 83.1 | 83.1 | 77.4 | **83.3** | **91.1** |
| 8x | 99.3 | 99.3 | 99.3 | 99.3 | 99.3 | **100.0** | 99.6 | 99.3 |

**Nothing beats LRU consistently.** The policies that win on OLMoE (LFU, hybrid) lose on
Qwen3-Next. Every margin is 1-3 points, well inside the difference between two models. LRU
stays.

Two things worth recording:

**Router confidence does not predict reuse.** The topk list arrives in rank order, so rank 0 is
the expert the router weighted most heavily - information LRU discards for free. Weighting
recency by rank (`rank-LRU`) is **consistently worse**, by 1 to 3.7 points. An expert the router
merely scraped in at rank 7 is just as likely to come back as its first choice. This had not
been tested before and it is a clean negative.

**The Belady gap is real but appears unreachable.** Optimal sits 9-14 points above LRU at the
ratios that matter, worth ~39% fewer misses at 4x. No online policy tested captures any of it.
That gap is the price of not knowing the future, not a bug in LRU.

---

## 2. Cheaper transfers: a measured 1.43x is sitting there

This is the most valuable thing in these notes, and it comes straight out of the existing
counters. The same run reports 164.96 GiB moved and 1356.3 us of copy per layer-call over 18432
calls:

```
177.1 GB  /  25.0 s  =  7.09 GB/s effective
```

Against the selftest's own bandwidth benchmark on the same machine:

| | GB/s | vs effective |
|---|---|---|
| **BELLS copies, measured** | **7.09** | - |
| benchmark, pageable | 9.02 | 1.27x |
| benchmark, pinned | 12.25 | **1.73x** |

Two things follow. We are at **79% of pageable peak**, so per-transfer launch overhead is small
and coalescing copies is not where the money is. But we are at **58% of pinned peak**, and that
gap is worth having:

```
copy/layer        1356 -> 784 us
per-layer total   1552 -> 980 us
BELLS overhead    74.5 -> 47.1 ms/token
decode            90.84 -> 63.4 ms   =  1.43x
```

**Why the copies are pageable.** The source is the model's mmap, which is ordinary pageable
memory. `cudaMemcpyAsync` from pageable memory is not truly asynchronous - the driver stages it
internally and the call blocks - so BELLS pays both the slower path and a stall it did not ask
for. `ggml_backend_tensor_set_async` cannot fix that; the memory has to be pinned.

### Staging through a pinned ring: built, measured, 37% slower

Implemented and tested rather than argued: a pinned staging ring allocated through
`ggml_backend_dev_host_buffer_type`, `mmap -> pinned` memcpy then async DMA out of the ring,
sized so wrapping is rare. Correctness held (bit-identical). Performance did not. Alternating
staged and unstaged in one session, same model and slot count:

| pass | config | copy/layer | decode | tok/s |
|---|---|---|---|---|
| 1 | staged | 2099 us | 133.55 ms | 7.49 |
| 1 | **unstaged** | **1497 us** | **100.29 ms** | **9.97** |
| 2 | staged | 2171 us | 137.92 ms | 7.25 |
| 2 | **unstaged** | **1433 us** | **97.03 ms** | **10.31** |

Reproducible, and `readback` stayed at 192-196 us across all four runs, so the machine was not
drifting. A first attempt with an 8-slot ring was also worse; enlarging it to 256 made it worse
still, which falsified the obvious explanation that synchronisation frequency was to blame.

**The reasoning error is worth keeping.** "Copies run at 7.09 GB/s, pinned benchmarks at 12.25,
therefore pinning wins" assumes the pageable path is a slow DMA that pinning replaces. It is
not. `cudaMemcpy` from pageable memory **already stages through an internal pinned buffer**, so
adding a staging copy does not replace that work - it duplicates it. Pageable DMA at ~7 GB/s and
a host memcpy at ~8 GB/s are the same operation priced twice, which is exactly the ~2x seen.

The code was reverted. It is in the branch history if anyone wants it.

**What that leaves.** The 12.25 GB/s number is only reachable if the DMA reads the model's pages
*directly*, which means registering them (`cudaHostRegister`) rather than copying them. That
removes the duplicated work instead of adding to it. The hazard is real though: pinned pages
cannot be swapped, so registering the expert tensors of a 27 GB model on a 32 GB machine risks
destabilising the box. It is plausible on the 128 GB configurations where this technique
actually wins, and should be tried there first, on a machine nobody is relying on.

**PCIe width still dominates all of it.** On a laptop dGPU the same benchmark gives 3.11 GB/s
pageable and 3.14 pinned - pinning buys *nothing* there, because the card is wired x4 instead of
x16 (`LnkSta: Width x4 (downgraded)`). This optimization helps the machines where BELLS already
wins and does nothing for the ones where it loses.

---

## 2b. Skipping low-weight experts: no threshold works

If misses cannot be reduced and transfers cannot be sped up, the next idea is to not transfer on
*some* misses. A missing expert must be copied before its matmul can run, but an expert the
router weighted at 0.02 contributes almost nothing - drop it from the sum, renormalise the rest,
and the copy disappears. There was even a reason to expect the trade to be favourable: rarely
chosen experts should also be the ones the cache is least likely to hold, so the experts worth
skipping and the experts that cost a transfer would be the same experts.

Measured instead of assumed. `llama-bells-profile` now records the normalised router weights
beside the trace, and `bells_weights.py` replays LRU and asks what each threshold buys.

**The weight distribution is nearly flat.** Qwen3-30B-A3B, mean weight by rank:

```
r0 0.229   r1 0.170   r2 0.139   r3 0.118   r4 0.102   r5 0.089   r6 0.080   r7 0.073
```

Rank 7 carries **7.3%**, not the 1-2% the idea assumed - only 3.1x below the router's first
choice. There is no negligible tail. Some of that is structural: selecting top-k and *then*
renormalising forces the survivors to sum to 1, which compresses the range.

**And the correlation barely exists.** Mean weight of a hit 0.1300, of a miss 0.1186 - misses
carry 0.91x the weight of hits, not the 0.5x that would have made this work.

**So every threshold is a bad deal** (16 slots, 2x ratio, 44% miss rate):

| threshold | copies saved | probability mass dropped |
|---|---|---|
| 0.03 | 0.4% | 0.02% |
| 0.05 | 1.8% | 0.23% |
| 0.08 | **23.0%** | **5.33%** |
| 0.12 | 62.3% | 18.91% |

Saving a fifth of the transfers means discarding 5% of the router's probability mass *in every
layer*, and that compounds across 48 of them. For scale, `drop_missing` discarded all
non-resident experts and scored perplexity 52.97 against a baseline of 2.03. This is gentler and
still nowhere near free, and the 1.8% saving that is genuinely cheap is not worth any accuracy
at all.

Recorded because the flat weight distribution is the interesting part, and it kills a family of
ideas rather than one: any scheme that leans on "most experts barely matter" is working from a
false premise on this architecture.

## 2c. Layers are wildly unequal, and a third of them are cached at a loss

Two questions here. The first - should slots be shared out unevenly between layers? - is a dead
end. The second, which the first exposed, looks like the best remaining idea in these notes.

**Uneven allocation does not help.** `bells_alloc.py` measures each layer's hit-rate curve and
hands slots out greedily to whichever layer gains most, against the same total VRAM. On
Qwen3-Next it is worth **+0.26 points**, about 1% fewer misses. The reason is that the curves are
*parallel*: every layer gains roughly 5-6 points per 4 extra slots regardless of where it starts,
and equal marginal value is exactly the condition under which uniform allocation is already
optimal.

**But the levels are not remotely equal**, which is the interesting part:

```
16 slots (2x):  mean 63.7%   min 23.2% (L0)   max 81.8% (L30)   spread 58 points
32 slots (4x):  mean 80.7%   min 42.5% (L0)   max 93.8% (L30)
```

The worst layers are the **early** ones - L0 23%, L1 43%, L2 48%, L3 49% - plus the final layer.
Early-layer routing is diffuse and input-dependent; deeper layers settle into stable expert
specialisation. That is structural rather than noise, and it makes the problem addressable.

**There is a break-even, and a third of the layers are the wrong side of it.** Caching a layer
only pays if moving the missing experts costs less than the CPU work it replaces. Both scale
with the same bytes, so the bytes cancel:

```
miss_rate x (bytes / pcie_bw)  <  bytes / cpu_bw
miss_rate  <  pcie_bw / cpu_bw  ~  7.09 / 18  ~  0.39      i.e. hit rate above ~61%
```

At a 2x ratio **15 of 48 layers sit below that line**, and they are the same layers with the
highest miss rates, so they consume a disproportionate share of the PCIe traffic while returning
the least of it. At 4x only 2 of 48 do - which is consistent with the measured results, where
generous caches win and tight ones do not.

### Measured: 1.14x, and it turns a loss into a win

Tested with a `BELLS_SKIP_LAYERS=N` knob that leaves the first N MoE layers uncached, so they
run as plain `--cpu-moe`. Static rather than adaptive, but enough to test the claim.
Qwen3-30B-A3B at 17 slots, five paired passes alternating in one session:

| | mean | sd | vs baseline |
|---|---|---|---|
| `--cpu-moe` baseline | 89.95 ms | 3.95 (4.4%) | 1.000x |
| BELLS, all layers cached | 95.26 ms | 3.85 (4.0%) | **0.944x** |
| BELLS, first 8 layers skipped | 83.29 ms | 3.10 (3.7%) | **1.080x** |

Per-pass ratio of skip-8 to cache-all, which cancels session drift:

```
1.188  1.123  1.184  1.099  1.126      mean 1.144x, sd 0.035
```

**A consistent 1.14x**, above 1.09 in every pass. The headline is not the ratio though - it is
that cache-all measures **0.944x**, an outright loss, and gating turns the same configuration
into a **1.080x** win. This is a setting where the honest advice today is "do not use BELLS",
and gating reverses that.

Two cautions. A single early pass suggested 1.22x; five passes say 1.144x, and the difference is
exactly the single-pass optimism this project has been caught by repeatedly. And run-to-run noise
on the absolute numbers is ~4%, comparable to the effect, which is why only the paired ratio is
quoted as the result.

**The proposal: gate the cache per layer.** Measure each layer's hit rate over a warmup window
and, for layers below break-even, fall back to plain `--cpu-moe` for that layer alone. Their
slots are then free for layers that do pay, so the benefit compounds. The measurement above uses
a fixed "skip the first N", which works because the early layers happen to be the bad ones on
this model; the adaptive version would not depend on that holding elsewhere.

Why this is more attractive than anything else left here:

- **The graph already supports it.** `build_moe_ffn` gates on `bells->tensors().has(il)`, which
  is per-layer today. Turning a layer off is a flag, not a redesign.
- **It targets the configurations that currently lose.** The 17-slot Qwen3-30B setup measured
  60.7% hit overall - sitting exactly on the break-even, which is precisely why it benchmarked
  at ~1.0x. Removing the layers dragging that average down is the difference between "no better
  than `--cpu-moe`" and a win.
- **It degrades safely.** A layer that is gated off is just `--cpu-moe`, the baseline.

### Correction: "gating puts a floor under BELLS" is too strong

That last bullet was the most valuable-sounding claim here and it does not survive its own test.

GPT-OSS-120B is where it mattered - the model BELLS loses worst on, 0.16x - so the prediction
was that gating should walk it back toward 1.0x. Measured on the A10G with a warmup pass and the
baseline re-measured afterwards to catch drift:

| skip | tok/s | vs baseline |
|---|---|---|
| baseline `--cpu-moe` | 8.63 | 1.00x |
| 0 (all 36 layers cached) | 1.53 | 0.18x |
| 12 | 2.30 | 0.27x |
| 24 | 2.57 | 0.30x |
| 30 (only 6 cached) | 4.10 | **0.48x** |

Gating helps a great deal - 0.18x to 0.48x, and the 0.18x agrees with the 0.16x already on
record, so the setup is comparable. **But it does not reach parity.** With 30 of 36 layers
already uncached it is still half the baseline, and the 22% baseline drift the harness reported
comes nowhere near explaining a gap that size.

The reason is the caveat written above before the test ran: the break-even assumes the CPU expert
path is memory-bound. GPT-OSS is MXFP4 and dequantises per element, so its CPU path is cheap and
*no* hit rate justifies paying 12.6 MB per miss. The correct decision for that model is to gate
every layer - which is BELLS switched off. A fixed "skip the first N" cannot express that; a
çœŸ per-layer gater would, and would land at the baseline trivially.

So the honest statement is: **the floor is the baseline only in the limit where gating disables
the cache entirely.** Partial gating does not interpolate to it. That is much weaker than "safe
to leave on", and the earlier wording should not be relied on.

A first attempt at this measurement, without the warmup, reported the baseline at 0.80 tok/s
against the 13.21 recorded for the same model on the same instance type, and produced an
apparent **6.70x** that was entirely the page cache filling across run order. The sweep now runs
a discarded warmup first and re-measures the baseline at the end, printing the drift. That check
is in the harness rather than in my attention because this is the fourth time in this project
that a quantity drifting with run order has been mistaken for the variable under test.

**Where the reasoning is incomplete, stated plainly.** The break-even assumes the CPU expert path
is memory-bound, so its time scales with bytes read. That holds for K-quants. It does not
obviously hold for GPT-OSS-120B's MXFP4, which needs dequantisation work per element and may be
compute-bound - and GPT-OSS is the model where BELLS loses worst (0.16x), so the case most in
need of explanation is the one this model fits least. The threshold should be measured per
machine, not taken from the 0.39 above.

## 3. The idea worth trying: do not transfer on a miss

Everything above accepts the premise that a missing expert must be copied into VRAM. It does
not have to be. **The weights are already in host RAM, and the CPU can compute with them - that
is exactly what `--cpu-moe` does for every expert.**

So:

- **resident expert** -> matmul on the GPU against the cache, as now
- **missing expert** -> matmul on the **CPU**, against the mmap, and no transfer at all

PCIe traffic during decode drops to zero. The cost becomes CPU work proportional to the *miss*
rate rather than to everything. At the 70% hit rate measured on Qwen3-30B at 4x, the CPU does
30% of the expert work it does in the `--cpu-moe` baseline, and nothing crosses the bus.

**Why this fits every measurement so far:**

- It removes the 87% term outright rather than shrinking it.
- It removes the `bytes/token` quantity that made GPT-OSS-120B lose 6.1x - 12.6 MB experts stop
  mattering when they are never moved.
- It removes the PCIe-width dependence that makes laptop dGPUs hopeless.
- It degrades gracefully: at 0% hit it *is* `--cpu-moe`, so it should never be much worse than
  the baseline. The current design can be 6x worse.

**The mechanism it needs exists, and is verified.** The obvious objection is that masking wastes
work: point half the ids somewhere harmless and the matmul computes them anyway. That is not how
ggml's CPU `mul_mat_id` behaves. It buckets rows by expert first and then skips any expert no row
selected (`ggml-cpu.c:1628`):

```c
for (int cur_a = 0; cur_a < n_as; ++cur_a) {
    const int64_t cne1 = matrix_row_counts[cur_a];
    if (cne1 == 0) {
        continue;          // never touched, so never read from memory
    }
```

An expert nothing points at costs nothing - not even the weight read, which is the whole cost on
a memory-bound CPU path. So:

- **GPU path**: `mul_mat_id` over the cache. Non-resident ids -> the zero slot, contributing 0.
  This is exactly the machinery built for the removed `drop_missing` mode.
- **CPU path**: `mul_mat_id` over the host tensors. Resident ids are pointed at *one of the
  experts this token is already missing*, and their router weight is set to 0 so the garbage
  they produce is multiplied away.
- sum the two.

Pointing the masked ids at an expert already being computed rather than at a spare is what makes
the CPU cost exactly "the distinct missing experts, and nothing else". A dedicated dummy expert
would have added one wasted weight read per layer. (When a token misses nothing there is no such
expert to borrow; that token pays one wasted expert, or the layer skips the CPU path entirely.)

**Costs and unknowns, honestly:**

- Needs `build_moe_ffn` to emit two expert paths and a join, which is a real change to graph
  construction rather than a tuning knob.
- Relies on `ggml_backend_sched` placing the CPU matmul on the CPU and not migrating the expert
  tensors. `--cpu-moe` already depends on that working, so the risk is moderate.
- The join is another cross-device dependency per layer. The current design already pays a
  host round trip per layer, so this may be a wash - but "may be" is doing work in that sentence.
- The weights for masked slots must be zeroed on the CPU side only, so the two paths need
  different weight vectors. Cheap, but fiddly.
- **Still untested end to end.** The mechanism is verified; the design is not. Three of the four
  avenues in these notes were expected to work and did not. The falsification test:
  build it and measure against `--cpu-moe` on GPT-OSS-120B, the configuration the current design
  loses worst (0.16x), where a design whose floor is `--cpu-moe` should show the largest gain.

---

## Reproducing

```
python tools/bells-profile/bells_policy.py models/bells/qwen3-big.trace.bin \
    --limit 1200 --layers 4 --mults 2,4,8
```

Belady is O(cache x accesses) and slow; pass `--no-belady` for a quick pass over more records.
