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

**An 80B model at 17.5 tok/s on a 6 GB RTX 2060.** On a 24 GB card, the same model runs at
39.9 tok/s - a model that plain `-ngl` cannot load at all.

It does not help every model. It made one of the three models tested measurably slower. The
tables below include the failures, because knowing which case you are in is most of the value.

---

## Measured results

Qwen3-Next-80B-A3B (Q2_K, 27 GB), 128 generated tokens, paired back-to-back runs:

| Config | RTX 2060 6 GB | A10G 24 GB |
|---|---|---|
| plain `-ngl` | 7.97 tok/s | cannot load the model |
| `--cpu-moe` (best stock llama.cpp) | 15.5 tok/s | 14.3 tok/s |
| **BELLS** | **18.4 tok/s** | **39.9 tok/s** |

Across three models on the 6 GB card:

| Model | cache ratio | baseline | BELLS | result |
|---|---|---|---|---|
| Qwen3-30B-A3B (Q4_K_M) | 2.2x | 11.4 tok/s | 17.6 tok/s | **1.52x** |
| Qwen3-Next-80B (Q2_K) | 4.7x | 15.5 tok/s | 18.4 tok/s | **1.18x** |
| GPT-OSS-120B (MXFP4) | 1.0x | 1.82 tok/s | 1.15 tok/s | **0.63x - slower** |

**Quality is unaffected.** Teacher-forced perplexity, scored one token at a time so the cache
is actually exercised: **2.0276 with BELLS, 2.0296 without.**

---

## Will it help my model?

One number decides it:

```
working_set = n_layer x n_experts_active x expert_bytes
cache_ratio = usable_VRAM / working_set
```

**Above ~2x it wins. Below ~1.5x it loses.** There is a calculator:

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
mixtral-8x7b              RAM        .        .        .        .        .     2.0x     2.0x     vram     vram
mixtral-8x22b             RAM      RAM      RAM      RAM      RAM      RAM      RAM        .        .     2.0x

N.Nx = worth using    ~ = marginal    . = cache too small to help
vram = model fits on the card, load it normally    RAM = model exceeds RAM
```

**This table is calculated, not measured.** Only four cells have benchmarks behind them:
qwen3-next-80b and qwen3-30b-a3b at 6/32, gpt-oss-120b at 6/32, and qwen3-next-80b on a
24 GB A10G. The rest is arithmetic from each model's `config.json`, validated against those
four (predicted expert sizes land within 1% of what the runtime reports). Treat it as a
screening tool.

Two patterns worth reading off it. **RAM is usually the binding constraint, not VRAM** - most
cells that say no say `RAM`. And **Mixtral is the shape that does not work**: 2-of-8 routing
means the cache would have to be half the model before it helps.

Three further conditions:

- **The model must fit in RAM.** Once it does not, BELLS reads cold experts from the same
  mmap the baseline does, so a miss costs the same disk read *plus* a PCIe copy. Strictly worse.
- **Sparse models only.** Look for many small experts and few active. Qwen3-Next has 512
  experts of ~1 MB with 10 active. GPT-OSS has 12.6 MB experts, and that alone sinks it.
- **Helps decode, not prefill.** A 512-token prompt touches 57-64 of 64 experts, so there is
  no hot set to exploit. Long prompts pay full price.

---

## Usage

```
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots -1 -fit off
```

- `--cpu-moe` keeps attention on the GPU and experts on the host, where BELLS reads them from.
  On its own it is worth ~2x over plain `-ngl` on MoE models - use it whether or not you use
  BELLS.
- `--bells-slots -1` sizes the cache from free VRAM. Pass a number to override.
- `-fit off` avoids llama.cpp's memory-probe pass, which otherwise constructs the cache twice
  and briefly doubles VRAM use.

`--bells-drop-missing` exists and is **not safe**: perplexity 52.97 against a baseline of
2.03, with generations collapsing into repetition loops. Research only.

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
