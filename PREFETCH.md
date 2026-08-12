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
