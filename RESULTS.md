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

## Correction: active parameters matter more than sparsity

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
