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
3. stop transferring on a miss at all - the largest prize and the largest unknown; a design
   sketch, not a result.

Two of the three are now closed by measurement rather than argument. Both were things I expected
to work: the eviction policies looked free, and the pinned staging had a 1.73x bandwidth gap
apparently sitting there for the taking. Neither survived contact with a paired A/B.

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

**It is implementable with machinery that already exists.** The slot table already maps a
non-resident expert to a zero slot - that mechanism was built for the removed `drop_missing`
mode, where a missing expert contributed nothing. Reuse it on both sides:

- GPU path: `mul_mat_id` over the cache, non-resident ids -> zero slot, contributing 0
- CPU path: `mul_mat_id` over the host tensors, resident ids -> a zero row, contributing 0
- sum the two

Both paths run over the same routing ids, each masking out what the other handled, and the sum
is exact. No mid-graph decision is needed beyond the slot table the runtime already uploads.
`ggml_backend_sched` already places ops across backends, which is how `--cpu-moe` works at all.

**Costs and unknowns, stated before rather than after:**

- Doubles the number of MoE matmuls, most of them mostly-zero work. Whether ggml's `mul_mat_id`
  skips zero rows cheaply is unverified and decides whether this is viable.
- Adds a CPU/GPU sync per layer for the join, which the current design also has.
- The CPU path cannot start until routing is known, same as today.
- **Entirely untested.** This is a design note, not a result. Given that five predictions in
  this project were confidently wrong before measurement, treat it as a hypothesis with a clear
  falsification test: build it, and measure against `--cpu-moe` on GPT-OSS-120B, the case the
  current design loses worst.

---

## Reproducing

```
python tools/bells-profile/bells_policy.py models/bells/qwen3-big.trace.bin \
    --limit 1200 --layers 4 --mults 2,4,8
```

Belady is O(cache x accesses) and slow; pass `--no-belady` for a quick pass over more records.
