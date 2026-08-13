# BELLS measurements

Everything here is measured on real hardware, not projected. Where a number is an estimate
it says so. Configurations that lost are included; the point of the table is to tell you
which situation you are in, not to sell the technique.

Test machine unless stated otherwise: RTX 2060 6 GB (PCIe 3.0), 32 GB DDR4, Ryzen-class CPU,
NVMe. Decode timings are steady-state generation, batch 1.

## The headline

Qwen3-Next-80B-A3B (Q2_K, 27 GB), 128 generated tokens:

Measured **back to back in a single session**, alternating baseline and BELLS. This matters:
see the methodology warning below.

| Config | ms/token | tok/s | vs baseline | safe? |
|---|---|---|---|---|
| plain `-ngl 10` | 125.5 | 7.97 | ~0.5x | yes |
| `--cpu-moe` (stock, strong baseline) | 64.4 | 15.5 | 1.00x | yes |
| **BELLS exact, 48 slots** | **54.4** | **18.4** | **1.18x** | **yes** |

Markdown corpus, same protocol: baseline 65.70/73.04, BELLS 64.21/61.95 -> **1.10x**.

So on a 6 GB RTX 2060 the honest figure is **1.10-1.18x**.

### Then the expert source was pinned, and the number moved

Everything above copies experts out of pageable host memory, where `cudaMemcpyAsync` blocks the
caller for 99.7% of the transfer. Pinning the source removes that stall - see [PINNED.md](PINNED.md).
Re-measured on the same machine, three paired passes alternating in one session:

| Config | ms/token | tok/s | vs pageable |
|---|---|---|---|
| BELLS 48 slots, pageable | 50.00 | 20.04 | 1.00x |
| **BELLS 48 slots, pinned** | **43.38** | **23.06** | **1.15x** |

Perplexity identical at 10.5201 across all six runs. 26688 MiB of experts pinned, a 944 MiB tail
still pageable.

**23.06 tok/s for an 80B model on a card that holds about a fortieth of it.** Note the pageable
baseline reads 20.04 here against 18.4 above - the same configuration, months apart, differing by
9%. That is the session drift the warning below is about, and it is why only the paired ratio is
quoted.

Slot sweep with pinning on, same session:

| slots | tok/s | cache VRAM |
|---|---|---|
| 48 | 23.06 | 2.51 GB |
| 32 | 21.46 | 1.67 GB |
| 24 | 18.96 | 1.26 GB |
| 16 | 17.09 | 0.84 GB |

More slots still wins, so pinning did not move the optimum. But **32 slots pinned beats 48 slots
pageable while using 840 MB less VRAM**, and on a 6 GB card spare VRAM is what decides whether a
larger model fits at all. Given ~22 tok/s already exceeds reading speed, that trade is usually the
better one.

The 1.15x here is much smaller than the 1.59x the same change gives on Qwen3-30B at 17 slots, and
the reason is the hit rate: 72% here against 61% there. Less copying leaves less stall to remove.
**Pinning pays in proportion to how much the configuration was transferring.**

## Methodology warning: pair your runs in time

An earlier version of this file claimed 1.25x mean and 1.40x best. Those numbers were wrong,
and the way they were wrong is worth recording.

Absolute throughput on this machine drifts by 30%+ over tens of minutes - thermal state,
background load, page cache. Runs taken minutes apart are not comparable. The same BELLS
configuration, with byte-identical cache behaviour (78.3% hit, 13356 experts copied, 14.16
GiB moved on every run), measured 42.05 ms early in a session and 58.55 ms later.

The earlier ratios compared BELLS runs from a fast period against baselines from a slow one.
Re-measured alternating baseline/BELLS back to back, within-config variance drops to 0.4% and
the ratio falls from 1.40x to 1.18x.

Rule: alternate the configurations inside one session and quote the paired ratio. Never
compare a number from one session against a number from another.

### The same trap, second encounter: a cold page cache

The rule above is necessary and not sufficient. Measuring `llama-server` concurrency, a first
attempt produced 0.68x for BELLS and a confident explanation - batching turns the CPU's
per-token GEMV into a GEMM, so the baseline gains what BELLS loses. The explanation was
plausible and the measurement was worthless.

The model is 27 GB against 32 GB of RAM. A freshly restarted server reads experts off disk,
and each configuration needed its own restart because the flag is set at startup. The BELLS
run went first and paid the cold reads; the baseline ran second on a cache BELLS had warmed.
An 8-token warmup did not begin to fix it. The giveaway was that throughput rose monotonically
with *run order* rather than with concurrency: 7.92, 11.48, 20.58, 22.36 tok/s.

With three full-concurrency warmup rounds and passes repeated until consecutive ones agree:

| concurrency | baseline | BELLS | ratio |
|---|---|---|---|
| 1 | 15.0 | 15.2 | 1.01x |
| 2 | 19.3 | 22.0 | 1.14x |
| 3 | 22.2 | 25.0 | 1.12x |
| 4 | 23.1 | 23.0 | 1.00x |

Cold-versus-warm on this machine is worth more than 2x, which is larger than any effect being
measured. Whenever the model approaches the size of RAM, warm until consecutive passes agree
before believing anything.

Note also that concurrency 1 here is 1.01x, against 1.18x from `llama-bells-profile` on the
same model and machine. Both are correct: a server request also pays prefill, which BELLS
bypasses, plus template and sampling overhead that a 100-token generation does not amortise.
Profiler numbers are an upper bound on what a served workload sees.

Each row is the mean of two runs. Run-to-run variance: baseline 60.04/60.99, exact
52.32/48.21, drop 42.41/42.10.

Repeated on a different corpus (C++ source instead of markdown docs, with the predictor
table still trained on the markdown):

| Config | ms/token | tok/s | vs baseline | diversity |
|---|---|---|---|---|
| `--cpu-moe` | 64.5 | 15.5 | 1.00x | 53 unique, 239 experts |
| **BELLS exact, 40 slots** | **49.8** | **20.1** | **1.29x** | 62 unique, 242 experts |
| BELLS drop, 64 slots, pf 12 | 28.0 | 35.7 | *1 token repeated 120x* | **degenerate** |

Exact mode holds out of sample and its output diversity matches the baseline. Use it.

Every exact-mode measurement taken, 128 tokens each. Baselines: markdown 60.04/60.99
(mean 60.5), C++ 58.27/58.90/60.03/64.47 (mean 60.4).

| corpus | slots | BELLS ms (each run) | mean | ratio |
|---|---|---|---|---|
| markdown | 40 | 52.32, 48.21 | 50.3 | 1.20x |
| markdown | 48 | 47.35, 57.50 | 52.4 | 1.15x |
| C++ source | 40 | 49.79, 47.27 | 48.5 | 1.25x |
| C++ source | 48 | 42.48, 42.05, 45.06 | 43.2 | **1.40x** |

Overall mean **~1.25x**, range 1.15-1.40x.

Run-to-run variance is the honest caveat: up to 21% on a single configuration (markdown, 48
slots gave 47.35 and 57.50 on consecutive runs). 48 slots looks best on the C++ corpus and is
within noise of 40 slots on markdown, so treat "48 is optimal" as unproven. Always compare
paired runs on the same corpus; a single number from a single run means nothing here.

## Quality, measured properly

Teacher-forced perplexity, 256 tokens scored one at a time so ubatch stays 1 and BELLS is
actually exercised (`llama-perplexity` runs large batches, which BELLS bypasses, so it cannot
measure the cache at all):

| Config | perplexity | vs baseline |
|---|---|---|
| `--cpu-moe` baseline | 2.0296 | - |
| **BELLS exact, 48 slots** | **2.0276** | **no cost** |
| BELLS drop-missing, 64 slots, pf 12 | **52.97** | **26x worse** |

Exact mode is free. The earlier token-alignment scores that looked alarming (48% similarity)
were measuring greedy trajectory divergence, not quality - two runs that differ in the last
bits of a float will separate permanently and score badly while being equally good. Perplexity
is the right instrument and it says exact mode changes nothing.

Drop-missing is not "slightly approximate". A perplexity of 53 is a broken model.

## `--bells-drop-missing` was not safe, and has been removed

**Removed along with the predictor.** Prefetching was the only way an expert became resident
in this mode - `on_routing` was skipped entirely - so with the predictor gone, every expert
would be non-resident and contribute nothing. The flag could not survive its dependency.

A second defect found while removing it, which does not change the numbers below but is worth
recording: the slot-table publish sat behind a `predictor enabled` check. Running
`--bells-drop-missing` *without* `--bells-table` therefore never wrote the slot tensors at
all, and the graph indexed the cache with whatever uninitialised VRAM `ggml_new_tensor_1d`
returned. The measurements here all used a table (`pf 12`), so they were taken on the path
that worked; anyone who tried the flag on its own was measuring garbage.

Letting a non-resident expert contribute nothing removes the per-layer host sync and is
genuinely faster, but it destabilises generation in an input-dependent way:

- markdown corpus, 128 tokens: fine (72% unique tokens, comparable to baseline)
- markdown corpus, 256 tokens: collapsed to a 3-token cycle
- C++ corpus, 128 tokens: collapsed to a single token repeated 120 times

Both collapses produced the *fastest* timings in the whole project (25.9 and 28.0 ms/token),
because a looping model touches almost no distinct experts.

An 80B model at 23.6 tok/s on a 6 GB card - and not worth having at that perplexity.

Note the run length. Over 24 generated tokens the same drop-missing config measures 54.5
ms/token; over 128 it measures 42.4. The cache is still warming during the first few dozen
tokens, so short benchmarks understate it.

## Always check for degeneration before believing a speedup

The profiler generates greedily (argmax, no sampling, no repetition penalty), which is
fragile. At 256 tokens the drop-missing run fell into a 3-token loop - "most of the"
repeating - and the timing dropped to 25.9 ms/token, an apparent 2.36x.

That number is worthless. A looping model touches almost no distinct experts, so the cache
hits ~100% and looks brilliant because the model broke:

| 256-token run | unique tokens | distinct experts (layer 0) | cycle |
|---|---|---|---|
| baseline | 169 (66%) | 437 | none |
| BELLS exact | 110 (43%) | 346 | none, but repetitive tail |
| BELLS drop | **3 (1.2%)** | **38** | **period 3** |

The 128-token runs in the table above were checked and are clean: 89-92 unique tokens
(69-72%) and 339-362 distinct experts across all three configs, so the comparison is fair.

Any expert-caching benchmark needs this check. Degeneration makes routing trivially
cacheable, so the technique flatters itself exactly when the output is worst. Use
`scripts/degen.py`-style diversity counts, not just tokens/sec.

The approximate path trades output for speed and the trade is not subtle: aligned similarity
against the baseline falls to ~48% over 128 tokens (9 edits). The text stays fluent, but it
is a different continuation.

Token-match is a weak quality proxy and gets weaker with length. Greedy decoding compounds:
once one token differs the trajectories separate permanently, so even *exact* mode diverges
from `--cpu-moe` simply because it computes experts on the GPU rather than the CPU and the
last bits disagree. Do not read a low alignment score as a quality loss on its own - it needs
perplexity. Exact mode remains the honest default.

"Equivalent" rather than "identical" for exact mode: BELLS computes experts on the GPU while
`--cpu-moe` computes them on the CPU, and the two disagree in the last bits. The cache path
itself is bit-identical (see the selftest); the drift is ordinary GPU-vs-CPU float behaviour.

## On a 24 GB card

Same model, A10G 24 GB / 8 vCPU / 64 GB RAM (Modal), 128 tokens, exact mode:

| Config | ms/token | tok/s | cache hit | vs baseline |
|---|---|---|---|---|
| `--cpu-moe` | 70.14 | 14.26 | - | 1.00x |
| plain `-ngl` | fails | - | - | 27 GB model does not fit in 24 GB |
| BELLS, 64 slots | 30.88 | 32.39 | 75.8% | 2.27x |
| BELLS, 128 slots | 25.53 | 39.17 | 84.2% | 2.75x |
| BELLS, 256 slots | 25.09 | 39.85 | 86.2% | **2.80x** |

Two caveats:

- **The instance CPU is weaker than the 6 GB test machine's** (8 vCPU; baseline 70.1 ms there
  versus 60.4 ms locally). That inflates the ratio. Scaling to the faster CPU suggests roughly
  **2.4x** on a desktop 3090, not 2.8x. Cloud instances with few vCPUs are the configuration
  BELLS flatters most.
- **The degeneration check was not run on these** - traces stayed in the container. Hit rates
  climbing 75.8 -> 84.2 -> 86.2% with cache size look like healthy behaviour rather than the
  ~100% a collapsed run produces, but this is unverified.

Note also that `-ngl` is not merely slower here, it cannot run the model at all. Expert-level
caching lets a 27 GB model run on a 24 GB card; layer-level offload does not.

## Portability: three GPU architectures and the first Linux build

Everything in this project was developed on Windows/MSVC against sm_75 and sm_86. Both of those
turned out to be hiding things.

**The code had never compiled on Linux.** MSVC pulls in headers transitively that gcc does not,
so `exp()` and `log()` in the perplexity path had no `<cmath>`. A cloud builder found it months
in. With that one line fixed, a full build on Debian 13 / gcc 14.2 is clean - 220/220 targets,
no errors and no warnings in the BELLS sources.

**sm_61 (Pascal) works.** Built and run on a GTX 1050 Mobile: cache path bit-identical, mid-graph
slot update visible to the gather, 16000 residency states, runtime loop exact. The graph trick
this project depends on - `mul_mat_id` over a slot-remapped tensor with an in-graph
`ggml_get_rows` id remap - is not architecture-specific. Three architectures validated: **sm_61,
sm_75, sm_86.**

**PCIe width matters more than the GPU.** The same bandwidth benchmark on both machines:

| | pageable | pinned | link |
|---|---|---|---|
| RTX 2060 desktop | 9.0 GB/s | 12.25 GB/s | 3.0 x16 |
| GTX 1050 laptop | 3.11 GB/s | 3.14 GB/s | 3.0 x4 (downgraded from x16) |

A laptop dGPU wired at x4 gets a quarter of the bandwidth, and pinned memory stops helping
entirely. Since copies are ~87% of BELLS's per-layer cost, that scales the dominant term
directly. Worth checking `lspci -vv | grep LnkSta` before assuming a card's bandwidth from its
model name.

## On a 6 GB card, under a server workload, BELLS does not help

The 1.52x recorded for Qwen3-30B-A3B on an RTX 2060 was measured with `llama-bells-profile` at
`-c 512`: pure decode, no prefill, no chat template, no sampling, and a large cache precisely
because a 512-token context leaves VRAM free. Re-measured through `llama-server` with
100-token chat completions, three passes each after a heavy warmup, paired within one session:

| concurrency | `--cpu-moe` | BELLS, 17 slots | ratio |
|---|---|---|---|
| 1 | 10.70 | 10.02 | 0.94x |
| 2 | 13.18 | 12.79 | 0.97x |
| 3 | 15.16 | 14.40 | 0.95x |
| 4 | 17.69 | 17.00 | 0.96x |

Uniformly about 5% slower. The shape is the informative part: with 17 slots BELLS serves
`ubatch <= 2`, so at concurrency 3 and 4 it bypasses itself entirely and should cost nothing -
yet those are slower too. **The cache occupies its VRAM whether or not it is used**, roughly
1.5 GB here, and squeezing the compute buffers on a 6 GB card costs about 5% unconditionally.
The benefit at low concurrency does not cover that fixed tax.

### The cache and the context come out of the same VRAM

llama.cpp allocates the KV cache first and BELLS sizes itself from the remainder, so context
length silently determines whether a cache is possible at all. Same card, same model:

| context | free VRAM at BELLS init | slots | ratio | verdict printed |
|---|---|---|---|---|
| 16384 | 2.4 GiB | 10 | 1.2x | "poor fit, expect a slowdown" |
| 4096 | 3.6 GiB | 17 | 2.1x | "workable" |

At `-c 16384` the tool warns and then proceeds anyway, and the measured result is 0.83x at
concurrency 1 - exactly what it warned about. That is a usability bug: it should refuse below a
ratio it knows is bad, not print a line the user will not read.

**The honest summary for a 6 GB card: use `--cpu-moe` and leave BELLS off.** It needs room for a
large cache *and* a real context at the same time, and 6 GB cannot provide both. The 24 GB
results below are unaffected - that card holds a 16K context and a 128-slot cache simultaneously,
which is why they work.

## The comparable set: one GPU, one CPU count, four models

Everything below was measured on the same A10G 24 GB with **16 vCPU and 128 GB RAM**, same
build, 128 generated tokens, at the best slot count found for each model. This is the table to
trust; the older 8 vCPU numbers elsewhere in this file are superseded and marked as such.

| Model | expert | active | baseline | BELLS | result | best slots |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B (Q2_K) | 1.09 MB | 3 B | 19.84 | **41.30** | **2.08x** | 128+ |
| Qwen3-235B-A22B (Q2_K) | 6.61 MB | 14 B | 3.10 | **5.61** | **1.81x** | 28 |
| Mixtral-8x7B (Q4_K_M) | 18 MB | 13 B | 3.70 | 3.93 | 1.06x | 4 |
| GPT-OSS-120B (MXFP4) | 12.6 MB | 5 B | 13.21 | 2.17 | **0.16x** | <=16 |

Plain `-ngl` failed to load every one of these on a 24 GB card, so `--cpu-moe` is the only
honest baseline. Highlights: **an 80B at 41 tok/s and a 235B at 5.6 tok/s, on one 24 GB card.**

**Pair within a run, never across runs.** Modal reads the model from a network volume, so the
first configuration in a fresh container pays cold reads. The 235B baseline measured 2.37,
3.10 and 2.11 tok/s in three different containers - a 47% spread on identical settings. Every
ratio above comes from a baseline and a BELLS run inside the *same* container, back to back.
Absolute tok/s from different runs are not comparable, which is the cloud version of the
session-drift warning at the top of this file.

### Cache size has an optimum, and it is per-model

Slot counts were first carried over from a 6 GB card and reported as if tuned, which understated
two models badly. Swept properly:

| Model | behaviour as the cache grows |
|---|---|
| Qwen3-Next-80B | climbs to a knee at 128 slots (6.7 GB), then flat: 38.2, 40.0, 39.8, 41.3 tok/s at 128/192/256/320. The last 10 GB of VRAM buys nothing. |
| Qwen3-235B-A22B | climbs to 28 slots (17.4 GB, 5.61 tok/s), then collapses: 32 slots 1.02x, 36 slots 0.73x |
| Mixtral-8x7B | peaks at 4 slots; 8 slots (every expert resident) is 0.80x |
| GPT-OSS-120B | already past its peak at 16; 28 slots is slower despite a better hit rate |

Two different failure modes for oversizing. On a small card, or when the cache approaches VRAM
capacity, it starves the context and compute buffers - the 235B at 32 slots is 19.9 GB on a
24 GB card, and an 81-slot cache on a 6 GB card measured 25% slower than 48. On GPT-OSS the
cache never got large in absolute terms and it still degraded, which is not explained.

**There is no general answer to "how big should the cache be".** It depends on the model and
the card, the penalty for guessing high can be severe, and the auto-sizer's "one third of free
VRAM" happens to land near the knee on a 6 GB card by coincidence rather than design.

### Confirmed: BELLS throughput is CPU-independent, and that is the predictor

The one hypothesis that survived a controlled test. Same A10G, same Qwen3-Next-80B, same 192
slots, same image - only `cpu=` changes:

| vCPU | `--cpu-moe` baseline | BELLS | ratio |
|---|---|---|---|
| 4 | 7.57 | **44.02** | 5.82x |
| 8 | 15.35 | **42.25** | 2.75x |
| 16 | 19.62 | **40.28** | 2.05x |
| 32 | 16.11 | **38.81** | 2.41x |

**BELLS moves 44.0 -> 38.8 tok/s across an 8x change in core count. The baseline moves 2.6x.**
Once the experts are resident the work is on the GPU, so the CPU stops mattering - and every
speedup ratio in this file is therefore a statement about the baseline's CPU, not about BELLS.

This is the useful form of the predictor that four earlier attempts failed to find. It does not
predict the *ratio*, which is not a property of BELLS at all. It predicts the *absolute number*:

> BELLS converges on a CPU-independent throughput. Measure your `--cpu-moe` baseline; if it
> sits below that number, BELLS wins by the difference, and if it sits above, BELLS loses.

It retro-explains the rest of the table. Mixtral's 5.88x evaporated because an 8-core baseline
was pathological for 13 B active parameters. GPT-OSS-120B loses because its baseline is already
13.21 tok/s, above what BELLS reaches on it. The 6 GB card gains nothing because a desktop CPU
gives a decent baseline while 6 GB caps what BELLS can reach.

One caveat on that run: the 32-core baseline (16.11) came in *below* the 16-core one (19.62),
which is probably thread contention and is not explained.

**Replicated on Mixtral-8x7B**, deliberately chosen because it was the model that appeared to
contradict this hypothesis - 4.73 tok/s at 8 vCPU dropping to 3.93 at 16, at the same slot count:

| vCPU | `--cpu-moe` baseline | BELLS, 4 slots | ratio |
|---|---|---|---|
| 4 | 2.51 | 3.60 | 1.43x |
| 8 | 2.30 | 3.74 | 1.63x |
| 16 | 3.77 | 4.47 | 1.19x |
| 32 | **5.47** | 3.94 | **0.72x** |

BELLS sits at 3.6-4.5 tok/s with no trend; the baseline climbs 2.30 to 5.47. The apparent
contradiction was inside Mixtral's own 24% noise band, not a real effect.

**And the rule predicted something before it was measured.** BELLS converges near 4 tok/s on this
model, so it must lose once the baseline passes 4 - and at 32 cores it does, 0.72x. The same
model, same code and same cache spans 1.63x to 0.72x on CPU alone. No property of BELLS changed.

### The four predictors that did not work

Four hypotheses were advanced in this file, each from the data available at the time, and each
falsified by the next measurement:

1. **Sparsity** decides it. Falsified by Mixtral, which is dense-ish and once measured 5.88x.
2. **Cache ratio** decides it, threshold ~2x. Falsified by GPT-OSS-120B: rated 7.8x "good fit",
   measured the worst result in the project.
3. **Active parameters** decide it. Rested entirely on Mixtral's 5.88x, which was an 8-core
   artifact - at 16 cores Mixtral is 1.06x and the hypothesis dies with it.
4. **Expert size** decides it. Falsified within one table: Mixtral's 18 MB experts score 1.06x
   while GPT-OSS's 12.6 MB score 0.16x.
5. **It works on the card it was built on.** The flagship 1.52x came from `llama-bells-profile`
   at `-c 512`. Falsified by running the same model through `llama-server` at a usable context:
   0.94-0.97x, consistently slower. See the 6 GB section above.

Five hypotheses, each reasonable on the data available when it was written, all five dead. The
sixth - CPU-independence - is the one that survived, and it works because it predicts an
absolute throughput rather than a ratio. Ratios were never a property of BELLS.

With four models and infrastructure noisy enough to move a baseline 47%, almost any rule will
fit the points in hand and break on the next one. The discipline that actually worked was not
better theorising: it was pairing every comparison inside one run, controlling one variable at a
time, and re-measuring anything that looked interesting.

Hit rate does not work either, in either direction: GPT-OSS got *slower* as its hit rate rose
(61.1% -> 74.2%, 460 -> 577 ms), while Qwen3-Next got faster (53.1% -> 73.6%). Any two of these
models can be used to argue any of the above rules, which is exactly the danger of a
three-model sample.

GPT-OSS also loses by much more than transfer volume explains. At a 61.1% hit rate it moves
roughly 706 MB per token, about 59 ms of PCIe at measured bandwidth, against a 385 ms
regression. Something specific to it - MXFP4 copy cost, or VRAM pressure from a 7-12 GB cache
squeezing the compute buffers - accounts for the rest, and it is not characterised.

**The practical consequence:** `bells_calc.py` answers "can this possibly fit", not "will this
help". Ten minutes with `llama-bells-profile` on your own model and hardware is the only
reliable answer, and the tool exists for that.

### Measured: copies dominate, not the round trip

The section below claimed the per-layer host round trip was the ceiling on this design. That
was inferred from the Mixtral result, not measured. Instrumented properly - separate timers for
reading the router's ids back to the host, copying missing experts in, and uploading the slot
table - on Qwen3-30B-A3B at 17 slots, 60.7% hit rate:

| per layer-call | time | share |
|---|---|---|
| readback (device -> host sync) | 167.1 us | 11% |
| **copy (host -> device experts)** | **1356.3 us** | **87%** |
| upload (slot table) | 28.7 us | 2% |

48 layers per token makes that **74.5 ms of a 90.84 ms decode - 82% of decode time spent inside
the BELLS callback**, overwhelmingly in transfers. The run moved 164.96 GiB.

So the round trip is real but small: readback plus upload is ~196 us per layer, about 13% of
the callback. **The cost is the copies**, which means the hit rate has to be high enough that
misses are rare, which needs a large cache, which needs VRAM. That single mechanism explains
most of the table:

| model / slots | hit rate | expert | bytes/token | result |
|---|---|---|---|---|
| Qwen3-Next-80B, 192 | 91.0% | 1.09 MB | ~47 MB | 2.08x |
| Qwen3-30B-A3B, 17 | 60.7% | 2.92 MB | ~550 MB | ~1.0x |
| GPT-OSS-120B, 16 | 61.1% | 12.6 MB | ~706 MB | 0.16x |

**The anomaly that survives:** Mixtral at 8 slots holds every expert, hits 100%, copies nothing
- and still measured 0.80x. These timers cannot see that, because they only measure work inside
the callback. Two candidates remain, both unmeasured: the graph split forced at every MoE layer
costs kernel pipelining, and a 4.6 GB cache squeezes the compute buffers. The 6 GB server
result points at the second - it was ~5% slower even at concurrencies where BELLS bypassed
itself entirely and did no work at all.

### Superseded reasoning: the per-layer round trip as a floor

Mixtral with 8 slots holds **every** expert in VRAM: 100% hit rate, zero copies, no PCIe
traffic during decode. It still measured 0.80x, slower than 4 slots and slower than the
baseline.

If the design loses while moving no data at all, the residual cost is structural: for every
layer, the graph stops, the router's selection is read back to the host, residency is corrected
and a slot table is uploaded. Thirty-two of those per token for Mixtral, ninety-four for the
235B. No cache policy, hit rate or slot count touches it. That fixed cost is the ceiling on
this whole approach, and removing it would mean keeping the correction on-device rather than
round-tripping through the host - which is a different design, not a tuning parameter.

## Correction: the A10G numbers were measured against a crippled baseline

Every earlier A10G result in this file used `cpu=8.0`, a Modal default nobody chose. BELLS
wins by moving work off the CPU, so an 8-core baseline flatters it in precisely the dimension
being measured, and a 24 GB card in the real world sits next to a desktop CPU. Re-run at
**16 vCPU**, 128 GB RAM, same GPU, same model file, 128 tokens:

Qwen3-Next-80B-A3B Q2_K:

| Config | ms/token | tok/s | vs baseline | cache hit |
|---|---|---|---|---|
| `--cpu-moe` baseline | 53.88 | 18.56 | 1.00x | - |
| plain `-ngl` | OOM on a 24 GB card | | | |
| BELLS, 16 slots | 49.63 | 20.15 | 1.09x | 53.1% |
| BELLS, 32 slots | 37.92 | 26.37 | 1.42x | 67.0% |
| **BELLS, 48 slots** | **33.12** | **30.19** | **1.63x** | 73.6% |

**2.80x becomes 1.63x.** Doubling the cores lifted the baseline from 14.3 to 18.56 tok/s and
dropped BELLS from 39.9 to 30.19.

**Superseded again, upward:** 48 slots was a cache size carried over from a 6 GB card and is far
too small for 24 GB. Swept properly the same configuration reaches 41.30 tok/s at 128+ slots, so
the figure a 3090 owner should expect is **2.08x**, not 1.63x. Kept here because the 8-vCPU
comparison above is what this section exists to correct.

Two notes. 48 slots was the largest tried and the curve was still improving, so this is a lower
bound rather than a tuned optimum. And the slot response here is the exact opposite of
GPT-OSS-120B below, where more cache made things worse - small experts reward a large cache,
large experts punish it.

## Correction: GPT-OSS-120B is much worse than recorded, and the ratio heuristic failed

This file previously recorded GPT-OSS-120B at 0.63x on a 6 GB RTX 2060 and attributed the loss
to the model exceeding RAM: 59 GB of weights against 32 GB, so every miss paid a disk read plus
a PCIe copy. The prediction that followed was that a machine with enough RAM would flip it to a
win, since the calculator rated 24 GB VRAM / 128 GB RAM at a 7.8x cache ratio - comfortably
inside "good fit".

Measured on an A10G 24 GB with **16 vCPU and 128 GB RAM**, 128 tokens:

| Config | ms/token | tok/s | vs baseline | cache hit |
|---|---|---|---|---|
| `--cpu-moe` baseline | 75.72 | **13.21** | 1.00x | - |
| plain `-ngl` | OOM, 59 GB model on a 24 GB card | | | |
| BELLS, 16 slots (4x ratio) | 460.37 | 2.17 | **6.1x slower** | 61.1% |
| BELLS, 28 slots (7x ratio) | 576.63 | 1.73 | **7.6x slower** | 74.2% |

The prediction was wrong, and wrong in the opposite direction. Removing the disk bottleneck
did not help BELLS; it helped the *baseline*, which went from 1.82 tok/s on the 2060 to 13.21
here. BELLS could not follow.

Three things to take from this.

**The cache ratio is not a reliable screen.** It rated this configuration 7.8x and it produced
the worst result in the project. Every earlier statement in this file about a "~2x threshold"
should be read as necessary but nowhere near sufficient.

**More cache was slower.** 28 slots achieved a better hit rate than 16 (74.2% vs 61.1%) and ran
25% slower. Hit rate and wall clock are anti-correlated again, exactly as they were for the
predictor. Anyone tuning this by hit rate will tune it in the wrong direction.

**Bytes per token is the thing that matters.** GPT-OSS has 12.6 MB experts against
Qwen3-Next's 1.09 MB:

| Model | expert | active x layers | bytes/token at measured hit rate | result |
|---|---|---|---|---|
| Qwen3-Next-80B | 1.09 MB | 10 x 48 | ~115 MB | 1.18x faster |
| Qwen3-30B-A3B | 2.93 MB | 8 x 48 | ~250 MB | 1.52x faster |
| GPT-OSS-120B | 12.6 MB | 4 x 36 | ~707 MB | 6.1x slower |

Mixtral is the counter-case that stops this being a clean rule: 18 MB experts and still the
largest speedup recorded, because ~11 B active parameters make its CPU baseline enormous. Both
terms matter and four models cannot fit two parameters. The practical advice is to measure.

**Useful side result for anyone with a 24 GB card and 128 GB of RAM:** GPT-OSS-120B runs at
13.21 tok/s on stock `--cpu-moe`, with BELLS switched off. That is a perfectly usable speed for
a 120B model on one consumer GPU, and it needs nothing from this repository.

## SUPERSEDED: active parameters matter more than sparsity

> **This section is wrong and is kept as a record of how.** Every number in it was measured at
> 8 vCPU. Re-run at 16 vCPU, Mixtral's baseline alone improves from 0.80 to 3.70 tok/s and the
> 5.88x becomes **1.06x**. The reasoning below was sound given the data available; the data was
> taken against a baseline crippled in exactly the dimension being argued about, which is the
> one thing that could have manufactured this conclusion out of nothing. See
> [the comparable set](#the-comparable-set-one-gpu-one-cpu-count-four-models).

An earlier version of this file, and of the README, said Mixtral-style models (2 of 8 experts)
were a poor fit and that high sparsity was what made BELLS work. Measured on an A10G 24 GB
with 8 vCPU and 128 GB RAM, Mixtral-8x7B-Instruct Q4_K_M, 64 tokens:

| Config | ms/token | tok/s | hit | vs baseline |
|---|---|---|---|---|
| `--cpu-moe` | 1244.4 | 0.80 | - | 1.00x |
| BELLS, 2 slots (1.0x ratio) | 323.3 | 3.09 | 43.2% | **3.85x** |
| BELLS, 4 slots (2.0x ratio) | 211.4 | 4.73 | 64.3% | **5.88x** |

That is the largest speedup measured anywhere in this project, on the model previously
described as hopeless, and it wins even at a 1.0x cache ratio - below the threshold where the
calculator prints "cache too small to help".

The mistake was treating sparsity as the driver. The real quantity is **how much CPU work is
being offloaded relative to how many bytes must move**. Mixtral activates ~13B parameters per
token against Qwen3-Next's ~3B, so its CPU baseline is enormous (1244 ms versus 64 ms) and
even 2.5 GB/token of PCIe traffic is cheap by comparison. Qwen3-Next gains only 1.18x because
there is very little CPU work to take away in the first place.

Restated: BELLS trades PCIe bandwidth for CPU compute. It wins when the compute you avoid
costs more than the bytes you move. High sparsity keeps the byte cost down, which helps, but
plenty of active parameters keeps the compute saving up, which helps more.

**Caveat that cuts the other way.** This instance has 8 vCPU, far weaker than the 6 GB test
desktop. A slow CPU inflates every BELLS result, because the baseline it is beating is worse.
The same model on a fast desktop CPU would show a smaller gain. Ratios measured on cloud
instances with few vCPUs should not be read as desktop numbers.

## A note on old GGUFs

MoE quantisations made before roughly early 2024 will not load in current llama.cpp at all,
BELLS or not. Expert weights moved from per-expert tensors (`blk.0.ffn_down.0.weight`) to a
stacked layout (`blk.0.ffn_down_exps.weight`), and older files fail with
`missing tensor 'blk.0.ffn_down_exps.weight'`.

This bites when testing older models, because the best-known GGUF repositories for them often
predate the change. Use a recent requantisation.

## What is measured and what is calculated

Four configurations have real benchmarks: Qwen3-Next-80B and Qwen3-30B-A3B and GPT-OSS-120B
on a 6 GB RTX 2060, plus Qwen3-Next-80B on a 24 GB A10G. Everything in the compatibility
matrix in the README beyond those is arithmetic from each model's `config.json`.

The arithmetic is validated where it can be: predicted expert sizes land within 1% of what the
runtime reports (1.10 vs 1.09 MB, 2.93 vs 2.92 MB, 12.69 vs 12.61 MB), and predicted cache
ratios match the auto-sizer's choices (4.4x vs 4.3x, 2.0x vs 2.2x, 1.2x vs 1.0x). But a
correct ratio is not a measured speedup, and the ratio is explicitly not a promise - see
Qwen3-30B beating Qwen3-Next despite a worse ratio.

## Which models it helps, remeasured

Every model below was rerun with the current code (pure LRU, no prefetch) using paired
back-to-back runs. **These supersede earlier numbers in this file that were taken with
prefetch enabled, which we later found actively harmful.**

| Model | quant | cache ratio | baseline | BELLS | result |
|---|---|---|---|---|---|
| GPT-OSS-120B | MXFP4 | 1.0x | 550 ms | 872 ms | **1.59x slower** |
| **Qwen3-30B-A3B** | Q4_K_M | 2.2x | 86.2 ms | 56.8 ms | **1.52x faster** |
| Qwen3-Next-80B | Q2_K | 4.7x | 64.4 ms | 54.4 ms | **1.18x faster** |

Two things to take from this.

**The useful threshold is about 2x, not 8x.** Qwen3-30B-A3B was written off earlier in this
file as a model BELLS could not help. That was wrong: it was measured with prefetch on, and
with prefetch removed it is the *largest* win we have on a 6 GB card.

**Ratio is a screen, not a predictor.** Qwen3-30B wins bigger at 2.2x than Qwen3-Next does at
4.7x. Likely because Qwen3-30B is Q4 rather than Q2, so its CPU baseline is slower (86 ms vs
64 ms) and there is more CPU work for the GPU to take over. BELLS saves most where the CPU
path is most expensive, not simply where the cache ratio is highest.

## Why this model and not others

The single number that predicts whether BELLS helps is the per-token working set:

```
working_set = n_layer x n_expert_used x expert_bytes
```

You need VRAM to hold roughly 4-8x that for the cache to get enough reuse. Sparsity
(`n_expert / n_expert_used`) is what lets a small card cache a big model.

| Model | expert size | working set | 6 GB card holds | measured |
|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 1.1 MB | 0.49 GB | **9.4x** | **1.42x speedup** |
| Qwen3-30B-A3B | 2.7 MB | 1.05 GB | 4.3x | 0.35x (slower) |
| GPT-OSS-120B | 12.6 MB | 1.81 GB | 2.4x | 0.52x (slower) |
| OLMoE-1B-7B | 3.5 MB | 0.45 GB | 10x | n/a, fits in VRAM anyway |

Qwen3-Next wins because 512 experts with only 10 active gives 51:1 sparsity and 1 MB experts.
GPT-OSS has 12.6 MB experts, so 6 GB holds only 2.4 working sets and the cache thrashes.

## Where it loses

Measured, on the same machine:

- **Qwen3-30B-A3B**: baseline 86.8 ms/token, BELLS 248-340 ms. Working set too large.
- **GPT-OSS-120B** (59 GB, exceeds RAM): baseline 748 ms, BELLS 1281-2042 ms. Cache holds
  6.25% of experts, hit rate 35.9%, so 288 experts move per token.
- **Any model that exceeds RAM**: BELLS reads source weights from the same mmap the baseline
  does, so a miss costs the same disk read *plus* a PCIe copy. Prefetch multiplies disk
  traffic. Strictly worse.
- **Dense-ish MoE** (Mixtral, 2 of 8 active): would need more VRAM than the model.

## The predictor does not earn its place

Every winning configuration above runs **without** a predictor table: plain LRU, demand-loaded.
Turning prediction on makes things worse, in both modes.

Exact mode, 48 slots, C++ corpus, 128 tokens:

| | ms/token | hit rate |
|---|---|---|
| pure LRU, no table | **42.1-45.1** | 78.3% |
| + predictor, prefetch 4 | 52.3 | 77.5% |
| + predictor, prefetch 10 | 75.4 | 75.1% |

Prefetching is slower *and* lowers the hit rate: speculative admissions evict entries that
were about to be needed, and the copies cost bandwidth whether or not the guess was right.
The same held in drop-missing mode (Qwen3-30B: 248 ms demand-only vs 340 ms with prediction).

This is worth stating plainly because it contradicts the premise of the project. The offline
analysis was sound - token id really does predict 68-77% of expert selection, and a counting
table really does beat LRU by 20-35 points *as a hit-rate statistic*. But hit rate was the
wrong objective. Demand loading moves exactly the experts a token needs; prediction moves a
superset. On a bandwidth-bound path, a worse policy that moves less data wins.

**The predictor has since been deleted from the runtime.** Keeping it switched off was the
wrong call: a flag that never helps is a trap for whoever reads the code next, and it kept
`--bells-drop-missing` alive as a dependent. The offline analysis below is unaffected, and
`bells_predict.py` still scores prediction against LRU and Belady on any trace, which is where
the question belongs. Git history has the runtime for anyone who wants to revisit it with a
cheaper transfer path - and note that the finding is specific to *this* transfer path, not a
proof that prediction cannot work.

## Predictor hit-rate analysis (offline, not load-bearing)

Token id alone determines most of the expert selection. Held-out, 20583 test records:

| cache | LRU | token-id table | Belady |
|---|---|---|---|
| 1x working set | 35.5% | **68.8%** | 61.3% |
| 2x | 65.9% | **86.7%** | 81.1% |
| 4x | 86.4% | **95.9%** | 94.1% |

Beating Belady is not an error: Belady is optimal only for demand paging, and prefetching
can cache an expert before anything has asked for it.

A plain counting table matches what a trained model would need to beat. Replicated on OLMoE
and Qwen3; controlled for training-set size (OLMoE cut to Qwen3's record count still scored
68.5%, so the difference is the model, not the data).

Two negative results worth keeping:

- **Cross-layer expert identity carries no signal**: 0.99 shared experts between layer L and
  L+1 against a chance value of 1.00. Predicting deeper layers from shallower ones is dead.
- **There is no small "burning" hot set.** Covering 80% of routing decisions takes roughly
  half of all experts. Expert usage is mildly skewed, not power-law.

## Hardware limits found

- **Pinned-memory H2D**: 12.25 GB/s on PCIe 3.0 (78% of theoretical). Pageable: 5.5-7.7 GB/s.
  The runtime must stage through pinned buffers.
- **NVMe queue depth matters.** Faulting expert pages one at a time left the drive at 0.89
  GB/s. Faulting across 8 threads first recovered 30% (2042 -> 1430 ms/token on GPT-OSS).
- **VRAM pressure has an optimum.** More cache is not monotonically better: 40 slots beat 64
  beat 80 on Qwen3-Next, because a large cache starves the context and compute buffers.
  Measured, exact mode, 128 tokens: 32 slots 73.1 ms, 40 slots 52.3, 64 slots 54.2, 80 slots
  73.3. Hit rate rises monotonically with slots (64.3% -> 77.0%) but throughput does not.
- **The per-layer sync cost is model-specific, not universal.** On Qwen3-30B exact mode cost
  ~2.4 ms/layer. On Qwen3-Next it costs ~0.15 ms/layer over 48 layers. An early conclusion
  that "the stall dominates" was drawn from one model and did not generalise.
- **`--cpu-moe` is a much stronger baseline than `-ngl` for MoE models.** On Qwen3-Next,
  `-ngl 10` gives 125.5 ms/token against `--cpu-moe`'s 60.0, because `--cpu-moe` keeps all
  attention on the GPU and only offloads experts. Benchmarks that compare against `-ngl`
  alone will overstate any expert-offloading technique by ~2x.

## Correctness

`llama-bells-selftest` covers, on the actual GPU:

- `mul_mat_id` over a slot-remapped cache is bit-identical to the full expert tensor
- mid-graph `slot_of` updates are visible to the downstream gather
- 16000 randomised residency states keep the mapping injective and fully resident
- the whole runtime loop, with real weight copies, matches the full expert stack exactly

Three constraints the graph depends on, each found by crashing into it:

1. `b` must be `[n_embd, 1, n_tokens]`, the broadcast form
2. experts within a token must be distinct, so the slot mapping must be injective
3. every selected expert must be resident, or `slot_of = -1` indexes out of bounds

## Reproducing

```
llama-bells-profile -m model.gguf -f corpus.txt -o profile.json --chunks 8 -c 512 -n 128 \
    -ngl 99 --cpu-moe
llama-bells-profile -m model.gguf -f corpus.txt -o out.json -c 512 -n 128 -ngl 99 \
    --cpu-moe --bells-slots 40
```

Alternate the two inside one session and quote the paired ratio - see the methodology warning
at the top of this file, twice.

The prediction analysis is offline and needs no runtime support:

```
python bells_build_table.py profile.trace.bin model.bells 32   # table, for analysis only
python bells_predict.py profile.trace.bin                      # LRU vs table vs Belady
```
