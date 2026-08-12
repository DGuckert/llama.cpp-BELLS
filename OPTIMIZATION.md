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

There are only three ways to move fewer bytes:

1. miss less often - **tested below, dead end**
2. transfer more cheaply - bounded by PCIe, and on some hardware badly so
3. **stop transferring on a miss at all** - the promising one, sketched at the end

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

## 2. Cheaper transfers: mostly not available

**Pinned staging is a wash or a loss.** Measured host-to-device: 9.0 GB/s pageable, 12.25 GB/s
pinned on the desktop. Tempting, but the source is the model's mmap, so using a pinned buffer
means `mmap -> pinned` (host memcpy, ~20 GB/s) then `pinned -> device` (12.25). Serially that is
`1/(1/20 + 1/12.25)` = **7.6 GB/s, worse than the 9.0 direct path**. It only pays if the memcpy
of expert *n+1* overlaps the DMA of expert *n*, which needs a double-buffered pipeline and wins
at most 36%.

**PCIe width matters more than any of this.** The same benchmark on a laptop dGPU: 3.11 GB/s
pageable, 3.14 pinned, because the card is wired x4 instead of x16 (`LnkSta: Width x4
(downgraded)`). Pinned memory stops helping entirely there. No software change recovers a
missing 12 lanes.

**Batching the per-expert copies** is untested and plausible. Each miss issues 3-4 separate
`ggml_backend_tensor_set_async` calls (gate, up, down), each a strided read of one expert. At
~2 misses per layer that is ~6 small transfers per layer, 288 per token on a 48-layer model.
Coalescing adjacent slots into one transfer would cut launch overhead, though not bytes.

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
