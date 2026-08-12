# I built an MoE expert cache and spent most of my time discovering my benchmarks were lying

I wanted to run models that do not fit in my GPU. The result is a working expert cache that
gets a 235B model to 5.6 tok/s on a single 24 GB card. That part worked.

The more useful output is the other part: over the course of building it I proposed **six
different rules** for predicting when the technique helps. Five are dead. Each was reasonable
given the data I had, and each was killed by the next measurement. What killed them was almost
never the code - it was the way the numbers were taken.

The sixth survived, and it explains why the other five could not have worked: **they all tried
to predict the speedup ratio, and the ratio is not a property of this technique at all.** It is
a property of the CPU you are comparing against.

The last one to die was the one I cared about most. I built this on a 6 GB RTX 2060, tuned it
there, and published a 1.52x speedup from it. Measured properly, **on the card I built it on it
is 5% slower than not using it at all.** It works on 24 GB cards. The card it was developed on
is the one where it does not help.

This is a writeup of both halves.

---

## What the thing does

Mixture-of-Experts models activate a small fraction of their weights per token. Qwen3-Next-80B
has 512 experts per layer and uses 10. So most of the model sits idle for any given token,
which is why `llama.cpp`'s `--cpu-moe` works so well: park the experts in host RAM, keep
attention on the GPU, and stream what you need.

The obvious next move is to cache the hot experts in VRAM. Keep `n` experts per layer resident,
rewrite the routing ids to point at cache slots, and run the matmul against the small cache
tensor instead of the full expert stack. On a miss, copy the expert in.

That is BELLS. The mechanism is unremarkable and works: the cache path is bit-identical to the
full expert tensor, and teacher-forced perplexity is 2.0276 against a 2.0296 baseline. Quality
is free.

**This is not a new idea.** `mixtral-offloading` (Eliseev & Mazur, 2023) did LRU expert caching
with speculative prefetch. ktransformers does GPU/CPU MoE splitting and has real users.
MoE-Infinity and Fiddler cover the research angle. I rediscovered a known design. That is fine
for learning and worth stating plainly before quoting any numbers.

---

## The results that survived

A10G 24 GB (the same silicon as an RTX 3090), 16 vCPU, 128 GB RAM. `--cpu-moe` is the baseline
because plain `-ngl` cannot load any of these on 24 GB.

| Model | expert size | active | baseline | BELLS | result |
|---|---|---|---|---|---|
| Qwen3-Next-80B (Q2_K) | 1.09 MB | 3 B | 19.84 | **41.30 tok/s** | **2.08x** |
| Qwen3-235B-A22B (Q2_K) | 6.61 MB | 14 B | 3.10 | **5.61 tok/s** | **1.81x** |
| Mixtral-8x7B (Q4_K_M) | 18 MB | 13 B | 3.70 | 3.93 tok/s | 1.06x |
| GPT-OSS-120B (MXFP4) | 12.6 MB | 5 B | 13.21 | 2.17 tok/s | **0.16x** |

An 80B at 41 tok/s and a 235B at 5.6 tok/s on one consumer-class card, on models that
layer-level offloading cannot load at all. That is the honest headline, and it is a real one.

Note the spread. Same code, same GPU, same day: one model gains 2x, one gains nothing, one is
six times slower.

---

## Five hypotheses, four dead

Each of these was written into the project's README at some point, with data behind it.

**1. Sparsity decides it.** Qwen3-Next uses 10 of 512 experts and wins; Mixtral uses 2 of 8 and
should lose. *Killed by* Mixtral measuring 5.88x, the largest speedup in the project at the
time.

**2. Cache ratio decides it, threshold ~2x.** `usable_VRAM / working_set`. I built a calculator
around it with eleven model presets and a compatibility matrix. *Killed by* GPT-OSS-120B, which
the calculator rated **7.8x, "good fit"**, and which measured **6.1x slower** - the worst result
in the project. A screening tool that confidently green-lights the worst case is worse than no
tool.

**3. Active parameters decide it.** Mixtral activates ~13 B parameters per token against
Qwen3-Next's ~3 B, so there is far more CPU work to displace. This explained hypothesis 1's
failure elegantly. *Killed by* re-running Mixtral on a machine with twice the cores: its
**baseline** went from 0.80 to 3.70 tok/s and the 5.88x became 1.06x. The entire hypothesis
rested on one measurement taken against a CPU too weak to run the model.

**4. Expert size decides it.** After 3 died, the remaining pattern looked clean: 1.09 MB experts
win, 12.6 MB experts lose catastrophically. Bytes moved per token is what matters. *Killed
within one table* - Mixtral's 18 MB experts score 1.06x while GPT-OSS's 12.6 MB score 0.16x.

**5. It works on the card I built it on.** For months the flagship result was 1.52x on a 6 GB
RTX 2060, measured with my own profiler at `-c 512`. *Killed by* running it through
`llama-server` at a context length someone would actually use: **0.94-0.97x, consistently
slower**. More on this below, because it is the most instructive failure of the six.

**6. BELLS is CPU-independent, so the ratio only measures how weak your CPU is.** *This one
survived*, and it survived because I finally ran a controlled experiment instead of comparing
whatever numbers I happened to have. Same GPU, same model, same cache size, varying **only** the
core count:

| vCPU | `--cpu-moe` | BELLS | ratio |
|---|---|---|---|
| 4 | 7.57 | **44.02** | 5.82x |
| 8 | 15.35 | **42.25** | 2.75x |
| 16 | 19.62 | **40.28** | 2.05x |
| 32 | 16.11 | **38.81** | 2.41x |

BELLS moves 13% across an 8x change in cores. The baseline moves 2.6x.

**Five hypotheses died because they all tried to predict the ratio, and the ratio is not a
property of BELLS.** It is a property of the thing BELLS is being compared against. Once experts
are resident the work is on the GPU and the CPU stops mattering, so "2.08x faster" is really
"your CPU was doing 19.62 tok/s and the GPU does 40."

The rule that works, and the only one I would put my name on:

> **BELLS converges on a CPU-independent throughput. Measure your `--cpu-moe` baseline. If it
> sits below that number, BELLS wins by the difference. If above, it loses.**

Which retro-explains everything. Mixtral's 5.88x was an 8-core baseline on a 13 B-active model.
GPT-OSS-120B loses because its baseline is already 13.21 tok/s, faster than BELLS manages on it.
The 6 GB card gains nothing because a decent desktop CPU meets a cache too small to beat it.

The lesson is not that I should have guessed better. It is that I spent months comparing
uncontrolled pairs, and one experiment varying a single variable settled in twenty minutes what
five hypotheses could not.

### The benchmark that measured the wrong thing for months

Hypothesis 5 deserves its own section, because it did not fail on a subtlety. It failed because
my benchmark harness measured pure decode and nothing else.

`llama-bells-profile` runs at `-c 512`, generates tokens one at a time, and does no prefill
worth the name, no chat template, and no sampling. It measured 1.52x on Qwen3-30B-A3B. That
number is *correct for what it measures*.

Through `llama-server` at `-c 4096`, 100-token chat completions, paired and warmed, same card
and same model:

| concurrency | `--cpu-moe` | BELLS | ratio |
|---|---|---|---|
| 1 | 10.70 | 10.02 | 0.94x |
| 2 | 13.18 | 12.79 | 0.97x |
| 3 | 15.16 | 14.40 | 0.95x |
| 4 | 17.69 | 17.00 | 0.96x |

Two things were hiding behind the profiler. First, **a 512-token context leaves VRAM free, and a
16K context does not.** The expert cache and the KV cache come out of the same pool and
llama.cpp allocates the KV cache first, so at `-c 16384` only a 1.2x cache ratio fits - a
configuration my own code labels *"poor fit, expect a slowdown"* before enabling itself anyway.
It duly measured 0.83x.

Second, and worse: the ratios above are ~0.95x at *every* concurrency, including the ones where
the cache is bypassed entirely and should cost exactly nothing. **The cache holds its VRAM
whether it is used or not.** On a 6 GB card, taking 1.5 GB away from the compute buffers costs
about 5% unconditionally, and the benefit never covers it.

So the correct advice for the hardware this was built on is: don't use it. Use `--cpu-moe`,
which is stock llama.cpp. The 24 GB results are untouched, and the reason is now obvious - that
card has room for a 16K context *and* a 128-slot cache at the same time. Six gigabytes does not.

I have since made it refuse to enable itself below a 1.5x ratio rather than warn and continue,
which is what it should have done from the start.

---

## The ways the measurements lied

None of the corrections above came from finding a bug. The code was largely right early. Every
one came from discovering the measurement was wrong.

**Degeneration inflates speedups.** The two fastest results ever recorded in this project -
25.9 and 28.0 ms/token, an apparent 2.36x - were the model collapsing into a repetition loop.
A looping model touches almost no distinct experts, so the cache hits ~100% and the timing looks
superb. Any expert-caching benchmark needs a degeneration check. I wrote one after being fooled
twice.

**Session drift fakes ratios.** Absolute throughput on a warm desktop drifts 30%+ over tens of
minutes. The same configuration with byte-identical cache behaviour measured 42.05 ms early in
a session and 58.55 ms later. An early draft claimed 1.40x by comparing a fast BELLS run against
a slow baseline. Paired back-to-back, it was 1.18x.

**Cold page cache fakes the opposite.** Measuring server concurrency, I got 0.68x for BELLS and
had a tidy explanation ready about batching turning the CPU's GEMV into a GEMM. The model was
27 GB on a 32 GB machine, each configuration needed its own restart, and whichever ran second
inherited a warmed cache. Properly warmed: 1.01/1.14/1.12x. **A 27 GB model on 32 GB of RAM is
not a benchmark platform** - page-cache state dominates everything.

**The same trap exists in the cloud.** Modal serves models from a network volume, so the first
configuration in a fresh container pays cold reads. One model's baseline measured 2.37, 3.10 and
2.11 tok/s across three containers on identical settings - a 47% spread.

**A default nobody chose.** Every cloud number for months used Modal's `cpu=8.0` default. BELLS
wins by moving work *off* the CPU, so an 8-core baseline flatters it in precisely the dimension
being measured. Raising it to 16 cut the headline from 2.80x to 2.08x and destroyed hypothesis 3
outright.

**Parameters ported across hardware.** I tuned cache sizes on a 6 GB card, then reused those
slot counts on a 24 GB card and published the result as though it were tuned. 48 slots is 2.5 GB
on a card with 24 GB. Sweeping properly took Qwen3-Next from 1.63x to 2.08x and the 235B from
1.50x to 1.81x.

**It had never compiled on Linux.** Every build was MSVC, which pulls in headers transitively
that GCC does not. The perplexity path used `exp()` and `log()` with no `<cmath>`. For a
llama.cpp fork - where essentially the whole audience is on Linux - "builds clean here" was
carrying far more weight than it had earned. A cloud builder told me, months in.

**The harness measured a workload nobody runs.** Covered above: pure decode at a 512-token
context, which is both the friendliest case for an expert cache and the one furthest from how
anyone uses a server.

The pattern: **every one of these moved the result by more than the effect being measured.**
Not one of them was a bug in the cache. The cache was fine.

---

## The finding I would keep if I could keep one

I wrote a section here claiming the per-layer host round-trip was the hard ceiling on this
design. The reasoning was that Mixtral with every expert resident - 100% hit rate, zero copies,
no PCIe traffic - still measured 0.80x, so the residual cost had to be the structural one:
stop the graph at each layer, read the router's selection back to the host, correct residency,
upload a slot table.

Then I instrumented it, which I should have done before writing the claim down.

Per layer-call on Qwen3-30B-A3B at 17 slots, 60.7% hit:

| | time | share |
|---|---|---|
| readback (device -> host sync) | 167.1 us | 11% |
| **copy (host -> device experts)** | **1356.3 us** | **87%** |
| upload (slot table) | 28.7 us | 2% |

Across 48 layers that is **74.5 ms of a 90.84 ms decode** - 82% of decode time inside the
callback, almost entirely transfers. The run moved 164.96 GiB. The round trip I had promoted to
"the ceiling" is 13% of the callback.

The correct version is duller and more useful: **the copies are the cost, so the hit rate has
to be high enough that misses are rare, which needs a large cache, which needs VRAM.** That one
mechanism explains most of the results - Qwen3-Next wins at 91% hit, and everything sitting near
60% hit does not.

It does *not* explain Mixtral at 100% hit still losing. That anomaly is still open, and my
timers cannot see it, because they only measure work inside the callback. The graph split forced
at every MoE layer costs kernel pipelining, and the cache occupies VRAM whether used or not -
both plausible, neither measured. So I have a sixth dead hypothesis and one unexplained result,
which is roughly the state this project has been in from the beginning.

Related, and equally counterintuitive: **hit rate and speed are anti-correlated** as often as
not. GPT-OSS got *slower* as its hit rate rose from 61.1% to 74.2%. A token-id predictor that
beat LRU by 20-35 points on hit rate lost every wall-clock benchmark it was in, because demand
loading moves exactly the experts a token needs while a predictor moves a superset. I deleted
the predictor. The project is named after it.

---

## What I would tell someone starting this

- **`--cpu-moe` is doing most of the work.** It is stock llama.cpp and worth ~2x over plain
  `-ngl` on MoE models. If you want to run a big MoE on a small card, start and possibly stop
  there. On a 24 GB card with 128 GB of RAM, GPT-OSS-120B runs at 13.2 tok/s on `--cpu-moe`
  alone, and every expert-caching configuration I tried made it worse.
- **Benchmark the workload you actually have.** This is the one that cost the most. A profiler
  measuring pure decode at a 512-token context flattered the technique for months, and the gap
  only appeared when I ran it as a server at a realistic context length.
- **Pair every comparison inside one run.** Never compare a number from one session against a
  number from another. This one rule would have prevented most of the errors above.
- **Re-measure anything interesting.** Every surprising result in this project was wrong the
  first time.
- **Check what your environment is deciding for you.** Core counts, RAM headroom, page-cache
  state, compiler, context length, and parameters inherited from different hardware. Each of
  those silently determined a result here.
- **Distrust proxy metrics.** Hit rate is not speed. Cache ratio is not speedup.
- **A tool that warns and proceeds is a tool that does nothing.** Mine printed "poor fit,
  expect a slowdown" and enabled itself anyway. Nobody reads startup lines. It refuses now.
- **Vary one variable.** Five hypotheses died over months of comparing uncontrolled pairs. One
  experiment changing only the core count settled the question in twenty minutes. If you find
  yourself proposing a new explanation for every new data point, stop theorising and go build
  a sweep.
- **Question whether your headline metric is a property of your system.** Mine was a ratio, and
  half of that ratio was somebody else's CPU.

The code is MIT and the measurements are reproducible. Whether the technique helps *your* model
on *your* hardware is not something I can predict - four attempts to build that predictor all
failed - but there is a profiler in the repo and it takes about ten minutes to find out.
