# What BELLS is good at

Bounded Expert LRU Loading System.

## In one sentence

BELLS moves MoE expert computation from the CPU to the GPU, by keeping the experts a model
actually uses in VRAM and streaming the rest in as needed.

That is the whole idea. Everything below is about when the trade pays off.

## The trade

Without BELLS (`--cpu-moe`, the best stock option for MoE):

```
attention on GPU  +  experts computed on CPU, reading from RAM at ~50 GB/s
```

With BELLS:

```
attention on GPU  +  experts computed on GPU, streamed over PCIe at ~12 GB/s
```

The GPU is far faster at the arithmetic. The catch is that PCIe is four times slower than
RAM, so you only win if the cache hits often enough that you are not constantly re-sending
weights. **BELLS wins when CPU expert compute costs more than the PCIe traffic needed to
avoid it.**

## The one number that decides it

```
working_set = n_layer x n_experts_active x expert_bytes
cache_ratio = usable_VRAM / working_set
```

`usable_VRAM` is your card minus attention weights, context and compute buffers. On a 6 GB
card running a 27 GB model that came to about 2.4 GB, not 6.

Measured outcomes, same machine, paired runs:

| cache ratio | model | result |
|---|---|---|
| 1.0x | GPT-OSS-120B | **1.59x slower** |
| 2.2x | Qwen3-30B-A3B | **1.52x faster** |
| 4.7x | Qwen3-Next-80B | **1.18x faster** |
| ~25x | Qwen3-Next-80B on a 24 GB card | **2.80x faster** |

**Below ~1.5x it loses. Above ~2x it wins. More VRAM keeps helping.**

Note that ratio is a screen, not a formula: 2.2x beat 4.7x here. Qwen3-30B has a slower CPU
baseline (86 ms vs 64 ms), so there was more CPU work for the GPU to take away. BELLS saves
most where the CPU path is most expensive, not simply where the cache is largest.

## Strengths

**1. It runs models that otherwise cannot run.**

On a 24 GB card, Qwen3-Next-80B is 27 GB. Plain `-ngl` fails outright - the model does not
fit. BELLS runs it at 39.9 tok/s. This is the strongest single argument for the technique:
not that it is faster, but that layer-level offload has no answer at all here while
expert-level caching does.

**2. Quality is free.**

Teacher-forced perplexity: 2.0276 with BELLS, 2.0296 without. The cache path is bit-identical
to the full expert tensor (verified on GPU across F32, Q4_K and MXFP4). You are not trading
accuracy for speed.

**3. It scales with VRAM.**

6 GB gives 1.2-1.5x. 24 GB gives 2.8x. The bigger the card, the higher the hit rate, the less
traffic, the bigger the win. Unusually, this is a technique that gets *better* on better
hardware rather than being a workaround for bad hardware.

**4. It works on mainstream models.**

Qwen3-30B-A3B - an ordinary, widely used MoE - is the largest win measured on a 6 GB card.
This is not confined to exotic architectures.

**5. It is a small, contained change.**

The cache is one extra tensor per layer plus an index rewrite inside `mul_mat_id`. No new ggml
ops, no changes to how models are loaded or quantised.

## Limits

**Needs a cache ratio of about 2x or better.** Below that, misses dominate and it is slower
than doing nothing. The tool reports this at load; believe it.

**Decode only.** A prefill batch of 512 tokens touches nearly every expert (57-64 of 64
measured), so there is no hot set to exploit. Long prompts pay full price.

**The model must fit in RAM.** Once it does not, BELLS reads cold experts from the same mmap
the baseline does, so a miss costs the same disk read *plus* a PCIe copy. Strictly worse.

**Dense models get nothing.** No experts, no cache, no benefit.

**Fat experts kill it.** GPT-OSS-120B has 12.6 MB experts against Qwen3-Next's 1.1 MB. Same
VRAM, a tenth of the cache ratio. Expert size matters more than parameter count: a 200 GB
model with small experts is an easier target than a 141 GB model with large ones.

## What did not work

**Prediction.** The project is named for a predictive loading system, and prediction does not
help. Every winning configuration is plain LRU. The offline analysis was sound - a token id
really does predict 68-77% of its experts, and a counting table really does beat LRU by 20-35
points on hit rate - but hit rate was the wrong objective. Demand loading moves exactly the
experts a token needs; prediction moves a superset, and on a bandwidth-bound path the extra
traffic costs more than the extra hits save. The predictor has since been deleted from the
runtime; the offline analysis that produced the finding is still in `bells_predict.py`.

**Dropping missing experts.** Fastest timings in the project, and a broken model: perplexity
52.97 against a baseline of 2.03, with generations collapsing into single-token loops.

**"Burning" hot experts.** There is no small permanently-hot set. Covering 80% of routing
decisions takes roughly half of all experts. Expert usage is mildly skewed, not power-law.

## How powerful is it, honestly

On a 6 GB card: **1.2-1.5x**, free of quality cost. Useful, not transformative.

On a 24 GB card: **2.8x**, and it runs models that simply do not fit otherwise. That is the
regime where it matters.

The technique is a good fit for one specific situation: **a sparse MoE that is larger than
your VRAM but fits in your RAM, on a machine where the GPU is strong relative to the CPU.**
Outside that, use `--cpu-moe` and save yourself the complexity.
