# BELLS

**B**ounded **E**xpert **L**RU **L**oading **S**ystem.

A per-expert VRAM cache for MoE inference in llama.cpp. Keeps the experts a model is likely
to need in VRAM and streams the rest from RAM, so a Mixture-of-Experts model larger than your
graphics card can still run its expert layers on the GPU.

Every word is load-bearing. **Bounded**: the cache is a fixed number of slots, and bigger is
not better - 40 slots beat 64 beat 80 in testing, because oversizing starves the context and
compute buffers. **Expert**: per-expert granularity, as opposed to the layer granularity of
`-ngl`. **LRU**: plain least-recently-used, which beat the predictive scheme this project was
originally built around.

**A 235B model on one 24 GB card at 5.6 tok/s, up from 3.1.** An 80B on the same card at
41.3 tok/s, up from 19.8. Both are models plain `-ngl` cannot load at all.

**It does not help every model, and there is no known way to predict which.** Of four models
measured on identical hardware, two gain about 1.5x, one gains nothing, and one is six times
*slower*. Four separate rules for telling them apart were proposed and each was falsified by
the next measurement - see [there is no working
predictor](RESULTS.md#there-is-no-working-predictor-and-this-is-the-main-negative-result).
Measure your own model; the tool for it is in this repo and takes ten minutes.

---

## Measured results

Four models, one machine: A10G 24 GB (the same silicon as an RTX 3090), 16 vCPU, 128 GB RAM,
128 generated tokens, best slot count each. Plain `-ngl` cannot load any of them on 24 GB, so
`--cpu-moe` is the only honest baseline.

| Model | expert size | active params | `--cpu-moe` | BELLS | result | slots |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B (Q2_K) | 1.09 MB | 3 B | 19.84 tok/s | **41.30** | **2.08x** | 128+ |
| Qwen3-235B-A22B (Q2_K) | 6.61 MB | 14 B | 3.10 tok/s | **5.61** | **1.81x** | 28 |
| Mixtral-8x7B (Q4_K_M) | 18 MB | 13 B | 3.70 tok/s | 3.93 | 1.06x | 4 |
| GPT-OSS-120B (MXFP4) | 12.6 MB | 5 B | 13.21 tok/s | 2.17 | **0.16x** | 16 |

No property in that table orders the results. Expert size fails (18 MB beats 12.6 MB), cache
ratio fails (GPT-OSS was rated a 7.8x "good fit"), active parameters fail (14 B wins, 13 B does
not), and hit rate fails in both directions.

**Cache size has a per-model optimum and guessing high can be expensive.** The 235B peaks at 28
slots and falls to 0.73x at 36; Mixtral peaks at 4 and drops to 0.80x at 8; Qwen3-Next reaches
a knee at 128 and then simply stops improving. Start with `--bells-slots -1` and sweep.

Across three models on the 6 GB card:

| Model | cache ratio | baseline | BELLS | result |
|---|---|---|---|---|
| Qwen3-30B-A3B (Q4_K_M) | 2.2x | 11.4 tok/s | 17.6 tok/s | **1.52x** |
| Qwen3-Next-80B (Q2_K) | 4.7x | 15.5 tok/s | 18.4 tok/s | **1.18x** |
| GPT-OSS-120B (MXFP4) | 1.0x | 1.82 tok/s | 1.15 tok/s | **0.63x - slower** |

### Superseded: the same models at 8 vCPU

An earlier version of this file led with these, from the same GPU on Modal's default 8 vCPU:

| Model | baseline | BELLS | claimed | actual, at 16 vCPU |
|---|---|---|---|---|
| Mixtral-8x7B (Q4_K_M) | 0.80 tok/s | 4.73 tok/s | ~~5.88x~~ | **1.06x** |
| Qwen3-Next-80B (Q2_K) | 14.3 tok/s | 39.9 tok/s | ~~2.80x~~ | **2.08x** |

8 vCPU cripples the baseline in exactly the dimension BELLS exploits, and a 24 GB card normally
sits beside a real desktop CPU. Mixtral is the cautionary one: its baseline alone went from
0.80 to 3.70 tok/s on twice the cores, because ~13 B active parameters on 8 cores was
pathological. Almost the whole "5.88x" was that. Note that BELLS's own throughput barely moved
in either case - it is the baseline that changed.

GPT-OSS-120B moved the other way, and its 6 GB result had the wrong explanation attached. It was
blamed on the model exceeding RAM. Given 128 GB it fits entirely, the baseline reaches 13.21
tok/s, and BELLS falls *further* behind: the problem was never disk, it is that 12.6 MB experts
move too many bytes per token. The larger cache also scored a **better** hit rate (74.2% vs
61.1%) and was **slower**.

**Quality is unaffected.** Teacher-forced perplexity, scored one token at a time so the cache
is actually exercised: **2.0276 with BELLS, 2.0296 without.**

---

## Will it help my model?

> **The cache ratio below is not reliable.** It rated GPT-OSS-120B on a 24 GB card with
> 128 GB of RAM at **7.8x, "good fit"**. Measured, that configuration is **6.1x slower** than
> `--cpu-moe` - the worst result in this project. A screening tool that green-lights the worst
> case is worse than none, so treat what follows as a rough filter and **measure before you
> trust it.** What actually predicts the outcome, as far as three models can tell, is bytes
> moved per token against how much CPU work is displaced - see
> [expert size is the discriminator](#expert-size-not-cache-ratio) below.

The ratio is still the first thing to compute, because a cache smaller than one token's
working set cannot help at all:

```
working_set = n_layer x n_experts_active x expert_bytes
cache_ratio = usable_VRAM / working_set
```

Below ~1.5x it reliably loses. Above that it *may* win, and the ratio does not tell you which.
There is a calculator:

```
$ python tools/bells-profile/bells_calc.py --vram 6 --ram 32 --preset qwen3-30b-a3b

  expert size              2.93 MB
  sparsity                 16.0:1   (128 experts, 8 active)
  per-token working set    1.13 GB   (48 layers x 8 x expert)
  model                    18.0 GB   fits RAM
  cache ratio               2.0x

  --> YES: workable
```

It knows nine common models by `--preset`, or take the numbers from a model's `config.json`
with `--layers --experts --active --hidden --ffn --quant`. Or just run with
`--bells-slots -1` and it tells you at load:

```
init: working set 0.51 GiB (48 layers x 10 experts x 1.09 MiB), cache holds 4.3x it -> good fit
```

### Compatibility at a glance

`python tools/bells-profile/bells_calc.py --matrix`, columns are VRAM/RAM in GB:

```
model                    4/16     6/32     8/32    12/32    12/64    16/64    24/64   24/128   48/256   80/512
-------------------------------------------------------------------------------------------------------------
qwen3-next-80b            RAM     4.4x     6.9x    11.9x    11.9x    17.0x    27.1x    27.1x     vram     vram
qwen3-30b-a3b             RAM     2.0x     3.2x     5.6x     5.6x     8.0x     vram     vram     vram     vram
qwen3-235b-a22b           RAM      RAM      RAM      RAM      RAM      RAM      RAM     2.9x     6.0x    10.4x
gpt-oss-120b              RAM      RAM      RAM      RAM      RAM      RAM      RAM     7.8x    16.5x     vram
gpt-oss-20b                 .        .     2.0x     4.0x     4.0x     vram     vram     vram     vram     vram
olmoe-1b-7b              2.0x     vram     vram     vram     vram     vram     vram     vram     vram     vram
deepseek-v4-flash         RAM      RAM      RAM      RAM      RAM      RAM      RAM     4.7x    11.7x    21.0x
deepseek-v3               RAM      RAM      RAM      RAM      RAM      RAM      RAM      RAM     4.0x     6.8x
kimi-k2                   RAM      RAM      RAM      RAM      RAM      RAM      RAM      RAM      RAM     5.1x
mixtral-8x7b              RAM        .        .        .        .    ~1.0x     2.0x     2.0x     vram     vram
mixtral-8x22b             RAM      RAM      RAM      RAM      RAM      RAM      RAM        .    ~1.0x     2.0x

N.Nx = the cache can hold this many working sets    . = cache too small to help anything
vram = model fits on the card, load it normally     RAM = model exceeds RAM
```

**Read the numbers as "can this possibly fit", not "will this help".** The ratio is calculated,
not measured, and it does not predict the outcome: `gpt-oss-120b` at 24/128 is rated 7.8x and
measures **6.1x slower** than the baseline, the worst result in this project. A high number
here means a cache is physically possible, nothing more.

**RAM is usually the binding constraint, not VRAM** - most cells that say no say `RAM`, and
that part of the table is reliable.

Ten cells have benchmarks behind them: qwen3-next-80b, qwen3-30b-a3b and gpt-oss-120b at 6/32;
qwen3-next-80b, qwen3-235b-a22b, mixtral-8x7b and gpt-oss-120b on a 24 GB A10G. Predicted
expert sizes land within 1% of what the runtime reports, so the *arithmetic* is sound - it is
the inference from arithmetic to speedup that fails.

### The one hard floor

Mixtral with 8 slots holds **every** expert in VRAM - 100% hit rate, zero copies, no PCIe
traffic during decode - and still measured 0.80x. If the design loses while moving no data at
all, the residual cost is structural: every layer stops the graph, reads the router's selection
back to the host, corrects residency and uploads a slot table. 32 of those per token for
Mixtral, 94 for the 235B. No slot count or cache policy touches it.

That fixed per-layer round trip is the ceiling on this approach, and removing it means keeping
the correction on-device instead of going through the host - a different design, not a tuning
knob.

Three further conditions:

- **The model must fit in RAM.** Once it does not, BELLS reads cold experts from the same
  mmap the baseline does, so a miss costs the same disk read *plus* a PCIe copy. Strictly worse.
- **Helps decode, not prefill.** A 512-token prompt touches 57-64 of 64 experts, so there is
  no hot set to exploit. Long prompts pay full price.
- **The weaker your CPU, the more this helps** - possibly the whole of the effect. BELLS moves
  expert compute onto the GPU, so its own throughput barely responds to core count while the
  `--cpu-moe` baseline does. Doubling cores left Qwen3-Next's BELLS number at 39.9 -> 41.3
  tok/s while its baseline went 14.3 -> 19.8. Stated as a hypothesis, not a finding: Mixtral
  contradicts it and the infrastructure noise is large enough to matter.

---

## Usage

```
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots -1
```

- `--cpu-moe` keeps attention on the GPU and experts on the host, where BELLS reads them from.
  On its own it is worth ~2x over plain `-ngl` on MoE models - use it whether or not you use
  BELLS.
- `--bells-slots -1` sizes the cache from free VRAM. Pass a number to override.

### Serving concurrent requests

BELLS serves batched decode up to `n_slot / n_expert_used` tokens per ubatch, which it prints
at load (`serves ubatch <= 3`). Beyond that the cache could not be guaranteed to hold every
expert a batch asks for, so those batches run the normal path. Prefill always runs the normal
path: a 512-token batch touches nearly every expert, so there is no hot set to exploit.

On `llama-server` with Qwen3-Next-80B on a 6 GB 2060, aggregate throughput over 100-token chat
completions, three passes each after a heavy warmup:

| concurrency | baseline | BELLS | ratio |
|---|---|---|---|
| 1 | 15.0 | 15.2 tok/s | 1.01x |
| 2 | 19.3 | 22.0 tok/s | 1.14x |
| 3 | 22.2 | 25.0 tok/s | 1.12x |
| 4 | 23.1 | 23.0 tok/s | 1.00x |

Note this is well below the 1.18x the same model shows in single-stream profiling, and the
gap is the point: a server request also pays prefill, which BELLS bypasses, plus template and
sampling overhead that a short generation does not amortise. Benchmark numbers taken with
`llama-bells-profile` are an upper bound on what a server workload will see.

Two flags that used to be here, `--bells-table` and `--bells-drop-missing`, have been removed.
Both are described under [Findings](#findings): one never won a benchmark, the other produced
a broken model. Git history has them if you want to revisit either.

---

## Tools

`llama-bells-profile` records per-token expert routing for any MoE architecture and writes a
numpy-loadable trace, plus coverage curves and LRU-versus-Belady bounds. Useful independently
of the cache.

```
llama-bells-profile -m model.gguf -f corpus.txt -o out.json -c 512 -n 128 -ngl 99 --cpu-moe
python tools/bells-profile/bells_calc.py --vram 6 --ram 32 --preset qwen3-next-80b
python tools/bells-profile/bells_degen.py out.trace.bin       # degeneration check
python tools/bells-profile/bells_quality.py ref.bin got.bin   # aligned output comparison
python tools/bells-profile/bells_predict.py out.trace.bin     # LRU vs table vs Belady
```

`llama-bells-selftest` verifies on your GPU that the cache path is bit-identical to the full
expert tensor, that the residency mapping stays injective across 16000 randomised states, and
that the whole runtime loop matches exactly.

---

## Findings

Beyond the speedups, the measurements produced results worth recording. Full detail with
methodology in [RESULTS.md](RESULTS.md); a plain-language overview in
[WHAT_BELLS_IS.md](WHAT_BELLS_IS.md).

**Token id predicts 68-77% of expert selection**, replicated on two architectures. A plain
counting table beats LRU by 20-35 points on hit rate. Held-out, on Qwen3: 1x working set gives
LRU 35.5% against the table's 68.8%.

**Prediction still loses.** Every winning configuration here is plain LRU with prediction
switched off. Hit rate was the wrong objective: demand loading moves exactly the experts a
token needs, prediction moves a superset, and on a bandwidth-bound path the extra traffic
costs more than the extra hits save. The project is named after a predictive loading system
that did not survive its own benchmarks.

The predictor has now been **deleted from the runtime** rather than left switched off, because
a flag that never helps is a trap for the next reader. `bells_predict.py` still scores
prediction against LRU and Belady offline, which is where the question belongs.

`--bells-drop-missing` went with it. Letting a non-resident expert contribute nothing removed
the per-layer host sync and was the fastest thing this project ever measured, but it scored
perplexity **52.97 against a baseline of 2.03** - a broken model, not an approximate one. It
also only ever half-worked: prefetching was the sole way an expert became resident in that
mode, so without a predictor table loaded the slot tensors were never written at all and the
graph indexed the cache with uninitialised VRAM. It could not survive the predictor's removal.

**Cross-layer expert identity carries no signal.** 0.99 shared experts between layer L and
L+1 against a chance value of 1.00. Predicting deeper layers from shallower ones is dead.

**There is no small "burning" hot set.** Covering 80% of routing decisions takes roughly half
of all experts. Expert usage is mildly skewed, not power-law.

**Prefetching can beat Belady's optimum**, because Belady is only optimal for demand paging
and a predictor can cache an expert before anything has asked for it.

### Two ways to fool yourself, both of which fooled me

**Degeneration inflates speedups.** A generation that collapses into a loop touches almost no
distinct experts, so the cache hits ~100% and the timing looks superb. The two fastest results
ever recorded here - 25.9 and 28.0 ms/token - were a model repeating one token 120 times.
`bells_degen.py` exists because of this.

**Session drift fakes ratios.** Absolute throughput on a warm desktop drifts 30%+ over tens of
minutes. The same configuration with byte-identical cache behaviour measured 42.05 ms early in
a session and 58.55 ms later. An earlier draft of these results claimed 1.40x by comparing a
fast BELLS run against a slow baseline. Alternate configurations back-to-back and quote the
paired ratio.

**A third trap, for anyone extending this:** the cache ratio screens candidates but does not
rank them. Qwen3-30B-A3B wins bigger at 2.2x than Qwen3-Next-80B does at 4.7x, because its
CPU baseline is slower and there was more work to move to the GPU. Two models with the same
ratio can behave differently, so treat the calculator as a filter, not a prediction.

---

## Building

```
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

Set `CMAKE_CUDA_ARCHITECTURES` for your card: `86` for RTX 30-series/A10, `89` for 40-series,
`75` for RTX 20-series, `61` for GTX 10-series. See [LAPTOP.md](LAPTOP.md) for notes on
running this on small cards.

---

## Credits and licence

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and
contributors, forked from
[antirez/llama.cpp-deepseek-v4-flash](https://github.com/antirez/llama.cpp-deepseek-v4-flash)
for its DeepSeek V4 architecture support. MIT, as upstream.

The BELLS code and the benchmarking in this repository were written with heavy AI assistance
(Claude). The measurements are real and reproducible; the code has not had the scrutiny of a
human author who wrote every line. Treat it accordingly, and note that llama.cpp upstream does
not accept predominantly AI-generated contributions - this fork is not intended for upstream
submission in its current form.
