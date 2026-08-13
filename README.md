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

**A 235B model on one 24 GB card at 11.3 tok/s, up from 3.1.** An 80B on the same card at
53.1 tok/s, up from 19.8. Both are models plain `-ngl` cannot load at all.

Those figures need the expert source in **pinned** host memory - `--cpu-moe-pinned`. Out of
pageable memory `cudaMemcpyAsync` blocks the caller for 99.7% of every transfer, which is worth
1.3x to 2.3x depending on how much a configuration moves, and which silently explains most of
the failed optimisations in this repository. See [PINNED.md](PINNED.md).

**Multi-GPU works, and buys capacity rather than speed.** Each device gets its own cache and
every layer's slice lives on the device that layer's graph runs on. Verified by perplexity
matching the single-GPU result exactly, since reading experts off the wrong card would produce
entirely plausible throughput. But a layer split is sequential - only one card computes at a
time - so two GPUs measure about **5% slower** than one. They are worth having when a model
needs the VRAM, not when it fits.

**It gains 2-3x on most MoE models, and the one exception is not a caching problem.** Against a
`--cpu-moe` baseline measured in the same session, at the best slot count for each:

| Model | `--cpu-moe` | BELLS pinned | |
|---|---|---|---|
| Qwen3-Next-80B (Q2_K) | 15.56 - 19.79 tok/s | **47.49 - 53.23** | **2.7 - 3.1x** |
| Qwen3-235B-A22B (Q2_K) | 3.09 | **8.03** | **2.60x** |
| Mixtral-8x7B (Q4_K_M) | 3.79 | **8.46** | **2.23x** |
| DeepSeek V4-Flash (IQ2_M) | 3.26 | **6.68** | **2.05x** |
| GPT-OSS-120B (MXFP4) | 13.21 | 2.83 | **0.21x** |

The 80B row is a range because it is the only one measured twice, and the two runs disagreed by
13% on the ratio - baseline 15.56 against 19.79, BELLS 47.49 against 53.23. Nothing changed
between them but the container. **Read every other row as carrying the same uncertainty**; they
are single runs and only look more precise.

GPT-OSS is the exception, and per-layer counters say the cache is not responsible: BELLS
contributes about **50 us against a 263 ms token, under 1%**. The cache behaves correctly
throughout - hit rate climbing 46% to 78% as slots rise, copy and layer totals falling - and the
model gets slower anyway. Something worth 1% cannot cause a 6x loss. The time is in graph
execution, which makes the real comparison GPU expert compute against CPU, and the GPU loses 3.4x
*on this model alone*. It is MXFP4, a Blackwell-native format, on Ampere.

**Two caveats that matter more than the numbers.**

The ratio is **not a property of BELLS**. Its own throughput is roughly CPU-independent, so the
multiplier is really "your CPU against this GPU" - the same Mixtral configuration measured 5.88x
on 8 vCPU and 1.06x on 16. A fast desktop CPU will show less than the table above. What travels
is the absolute figure: experts run at GPU speed instead of CPU speed.

And the slot count needs sweeping. Wrong values measured 0.94x on a 6 GB card, and oversizing is
punished harder than undersizing. `--bells` auto-sizes conservatively; beat it with
`--bells-slots`.

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

### Re-measured with a pinned expert source, and the table changes

Everything above copies experts out of **pageable** host memory, where `cudaMemcpyAsync` blocks
the caller for 99.7% of each transfer. With `--cpu-moe-pinned` ([PINNED.md](PINNED.md)):

| Model | pageable | **pinned** | gain | vs `--cpu-moe` |
|---|---|---|---|---|
| Qwen3-Next-80B, 256 slots | 40.65 tok/s | **52.66** | 1.31x | ~2.7x |
| Qwen3-235B-A22B, 28 slots | 5.45 | **11.27** | 2.08x | ~3.6x |
| Qwen3-235B-A22B, 16 slots | 3.84 | **8.64** | 2.25x | |
| **Mixtral-8x7B, 4 slots** | 3.24 | **8.46** | **2.61x** | **2.23x** |
| GPT-OSS-120B, 16 slots | 2.14 | 2.83 | 1.32x | still 0.21x |

**Mixtral moves from 1.06x to 2.23x**, against a baseline measured in the same session. It was
the headline example of a model BELLS does not help.

**And GPT-OSS is not a cache failure.** Per-layer counters put BELLS' entire contribution at
~50 us against a 263 ms token - **under 1%**. The cache behaves correctly throughout (hit rate
46% to 78% as slots rise, copy and layer totals falling) and the model still gets slower. A
component worth 1% cannot cause a 6x loss. The time is in graph execution, so the real
comparison is GPU expert compute against CPU, and the GPU loses 3.4x *on this model only* -
consistent with MXFP4 being a Blackwell-native format emulated on Ampere.

So the negative result above needs reading with care. **Three of its four rules were falsified
by GPT-OSS**, and GPT-OSS appears not to have been measuring the cache at all. Two of the four
`--cpu-moe` baselines in the first table are also from earlier sessions, which this repository
is emphatic elsewhere is not a valid comparison. The claim has not been rewritten because
settling it properly needs those two baselines re-measured, not because it still stands.

One correction to the first table regardless: Mixtral's experts are **~99 MB**, not 18 MB. The
cache measures 15.89 GiB for five allocated slots across 32 layers, and 3 x 4096 x 14336 at
Q4_K_M independently gives 105 MB. Expert size was one of the properties used to argue that
nothing orders the results.

### The ratio is not a property of BELLS

Varying **only** the core count, same GPU, same model, same 192 slots:

| vCPU | `--cpu-moe` | BELLS | ratio |
|---|---|---|---|
| 4 | 7.57 | **44.02** | 5.82x |
| 8 | 15.35 | **42.25** | 2.75x |
| 16 | 19.62 | **40.28** | 2.05x |
| 32 | 16.11 | **38.81** | 2.41x |

BELLS moves 13% across an 8x change in cores. The baseline moves 2.6x. Once experts are
resident the work is on the GPU, so **every speedup ratio here is really a measurement of the
baseline's CPU.** Which gives the one rule in this project that survived a controlled test:

> **BELLS converges on a CPU-independent throughput. Measure your `--cpu-moe` baseline - if it
> is below that number, BELLS wins by the difference; if above, it loses.**

Replicated on Mixtral-8x7B, where BELLS holds 3.6-4.5 tok/s while the baseline climbs 2.30 to
5.47 - so the same model, same code and same cache spans **1.63x at 8 cores to 0.72x at 32**.
The rule called that in advance: BELLS converges near 4 tok/s on this model, so it has to lose
once the baseline passes 4.

That is also why Mixtral's headline 5.88x collapsed to 1.06x on a better CPU, and why
GPT-OSS-120B loses: its baseline of 13.21 tok/s is already faster than BELLS manages on it.

**Cache size has a per-model optimum and guessing high can be expensive.** The 235B peaks at 28
slots and falls to 0.73x at 36; Mixtral peaks at 4 and drops to 0.80x at 8; Qwen3-Next reaches
a knee at 128 and then simply stops improving. Start with `--bells-slots -1` and sweep.

### On a 6 GB card, don't bother

Three models with `llama-bells-profile` at `-c 512` - pure decode, no prefill or sampling, and
a large cache because a 512-token context leaves VRAM free:

| Model | cache ratio | baseline | BELLS | result |
|---|---|---|---|---|
| Qwen3-30B-A3B (Q4_K_M) | 2.2x | 11.4 tok/s | 17.6 tok/s | 1.52x |
| Qwen3-Next-80B (Q2_K) | 4.7x | 15.5 tok/s | 18.4 tok/s | 1.18x |
| GPT-OSS-120B (MXFP4) | 1.0x | 1.82 tok/s | 1.15 tok/s | 0.63x - slower |

**Those do not survive a server workload.** The same 2060 and the same Qwen3-30B, through
`llama-server` at `-c 4096` with 100-token chat completions, paired and warmed:

| concurrency | `--cpu-moe` | BELLS | ratio |
|---|---|---|---|
| 1 | 10.70 | 10.02 | 0.94x |
| 2 | 13.18 | 12.79 | 0.97x |
| 3 | 15.16 | 14.40 | 0.95x |
| 4 | 17.69 | 17.00 | 0.96x |

Uniformly ~5% slower, including at concurrencies where BELLS bypasses itself and should cost
nothing - **the cache holds its VRAM whether used or not**, and on a 6 GB card that squeeze is
worth about 5%. Worse, the cache and the KV cache come out of the same pool: at `-c 16384`
there is only room for a 1.2x ratio, which BELLS itself labels *"poor fit, expect a slowdown"*
before proceeding anyway, and it duly measures 0.83x.

**If you have 6 GB, use `--cpu-moe` and leave BELLS off.** It needs room for a large cache *and*
a real context at once. The 24 GB results above are unaffected, because that card has room for
both.

> **This section predates pinned expert memory and needs re-testing.** Every number in it copies
> experts out of pageable host memory, where `cudaMemcpyAsync` blocks the caller for 99.7% of the
> transfer. Pinning the source ([PINNED.md](PINNED.md)) is worth **1.59x on Qwen3-30B at 17 slots**
> and **1.15x on Qwen3-Next-80B at 48 slots**, both paired in-session on this same 6 GB 2060:
>
> | model | pageable | pinned | |
> |---|---|---|---|
> | Qwen3-30B-A3B (Q4_K_M), 17 slots | 10.62 tok/s | **16.83** | 1.59x |
> | Qwen3-Next-80B (Q2_K), 48 slots | 20.04 tok/s | **23.06** | 1.15x |
>
> The *server* verdict above may or may not survive that - the ~5% loss there was attributed to
> the cache holding VRAM whether used or not, which pinning does not change, so it plausibly
> stands. It has not been re-measured. The VRAM squeeze argument is unaffected either way.
>
> One thing that does change: at 32 slots pinned the 80B still reaches 21.46 tok/s while giving
> back 840 MB, which is a better trade than it looks when 22 tok/s is already faster than most
> people read.

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

### Where the time actually goes

Instrumented per layer-call on Qwen3-30B-A3B at 17 slots (60.7% hit):

| | time | share |
|---|---|---|
| readback (device -> host sync) | 167.1 us | 11% |
| **copy (host -> device experts)** | **1356.3 us** | **87%** |
| upload (slot table) | 28.7 us | 2% |

Across 48 layers that is **74.5 ms of a 90.84 ms decode**. The transfers are the cost, so the
hit rate has to be high enough that misses are rare - which needs a large cache, which needs
VRAM. Qwen3-Next wins at 91% hit; everything at ~60% hit does not.

An earlier version of this file claimed the per-layer host round trip was the ceiling. It is
about 13% of the callback, and that claim was inferred rather than measured.

**One thing this does not explain.** Mixtral at 8 slots holds every expert, hits 100%, copies
nothing, and still measured 0.80x. The timers above only see work inside the callback, so two
candidates remain unmeasured: the graph split forced at every MoE layer costs kernel
pipelining, and the cache squeezes the compute buffers whether or not it is used. The 6 GB
result supports the second - it was ~5% slower even where BELLS bypassed itself entirely.

Three further conditions:

- **The model must fit in RAM.** Once it does not, BELLS reads cold experts from the same
  mmap the baseline does, so a miss costs the same disk read *plus* a PCIe copy. Strictly worse.
- **Helps decode, not prefill.** A 512-token prompt touches 57-64 of 64 experts, so there is
  no hot set to exploit. Long prompts pay full price.
- **You need enough VRAM for a large cache *and* your context, at once.** They come out of the
  same pool and llama.cpp allocates the KV cache first. On a 6 GB card at `-c 16384` only a
  1.2x ratio fits, which is worse than useless. 24 GB holds a 16K context and a 128-slot cache
  together, which is why every positive result here is on the larger card.
- **The weaker your CPU, the more this helps** - possibly the whole of the effect. BELLS moves
  expert compute onto the GPU, so its own throughput barely responds to core count while the
  `--cpu-moe` baseline does. Doubling cores left Qwen3-Next's BELLS number at 39.9 -> 41.3
  tok/s while its baseline went 14.3 -> 19.8. Stated as a hypothesis, not a finding: Mixtral
  contradicts it and the infrastructure noise is large enough to matter.

---

## Start here

Nothing about your model or your card predicts whether this helps. The only thing that does is
your own `--cpu-moe` number, so measure that first. Twenty minutes, four steps.

**1. Get your baseline.** This is stock llama.cpp and needs nothing from this fork. It is worth
~2x over plain `-ngl` on MoE models, and for many people it is the whole answer.

```
llama-bells-profile -m model.gguf -f corpus.txt -o base.json -c 512 -n 128 -ngl 99 --cpu-moe
```

**2. Check a cache can physically fit.** This rules things out; it cannot rule them in.

```
python tools/bells-profile/bells_calc.py --vram 24 --ram 128 --preset qwen3-next-80b
```

**3. Measure BELLS against it, in the same session.**

```
llama-bells-profile -m model.gguf -f corpus.txt -o bells.json -c 512 -n 128 -ngl 99 \
    --cpu-moe --bells-slots 128
```

**4. Sweep the slot count.** There is a per-model optimum and no rule for finding it. The 235B
peaks at 28 slots and falls to 0.73x at 36; Mixtral peaks at 4 and drops to 0.80x at 8;
Qwen3-Next climbs to 128 and then goes flat. Try a few, keep the best.

> **Alternate the two configurations inside one session and quote the paired ratio.** Absolute
> throughput drifts 30%+ over tens of minutes on a warm machine, which is more than the effect
> you are measuring. Nearly every wrong number in this project came from ignoring that.

Then run it for real:

```
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots 128
```

`--bells-slots -1` sizes the cache automatically. It is deliberately conservative and a sweep
usually beats it - see [Automatic sizing](#automatic-sizing) below.

### Will it help me?

| your situation | expect |
|---|---|
| 24 GB card, weak or busy CPU, model too big for `-ngl` | **the good case** - up to ~2x |
| 24 GB card, strong CPU | smaller gain, possibly none - measure |
| your `--cpu-moe` already beats what BELLS reaches | **it will lose.** Do not use it |
| 6-8 GB card | **no.** Not enough VRAM for a cache and a real context at once |
| model does not fit in RAM | **no.** Strictly worse than `--cpu-moe` |

### Automatic sizing

`--bells-slots -1` takes free VRAM minus a third for headroom. That heuristic was calibrated on
a 6 GB card, which is now known to be a configuration where BELLS does not help at all, so treat
it as a safe starting point rather than a tuned one. It errs small on purpose: oversizing is
punished hard - a 19.9 GB cache on a 24 GB card took the 235B to 1.02x and 22.4 GB took it to
0.73x, because the cache and the compute buffers come out of the same VRAM.

It has not been re-tuned because doing so honestly needs a slot sweep per model per card, and a
heuristic fitted to four models would be the sixth predictor this project has had to retract.
Sweep instead; the runtime prints the cache size in MiB at load so you can see what you got.

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
