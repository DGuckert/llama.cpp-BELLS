# Running on a GTX 1050 (4 GB VRAM, 16 GB RAM)

Build: `build-1050/`, compiled for Pascal (`CMAKE_CUDA_ARCHITECTURES=61`). The `build-bells/`
binaries will not run on this GPU - they are `sm_86` only.

## Two constraints

```
model must fit in RAM            ->  <= ~14 GB
cache should hold >= 2x working set
working_set = n_layer x n_expert_used x expert_bytes
```

4 GB VRAM leaves roughly 2.5 GB after context and attention, and the auto-sizer keeps a third
back, so the usable cache is about **1.7 GB**.

## Candidates, worked through

| Model | quant | size | working set | cache ratio | verdict |
|---|---|---|---|---|---|
| **Qwen3-30B-A3B** | Q2_K | ~11 GB | ~0.58 GB | **2.9x** | **best bet, should be a win** |
| GPT-OSS-20B | MXFP4 | 12 GB | 1.25 GB | 1.4x | marginal to poor |
| OLMoE-1B-7B | Q4_K_M | 4 GB | 0.45 GB | 3.8x | fits in VRAM anyway, BELLS pointless |
| Qwen3-Next-80B | Q2_K | 27 GB | 0.51 GB | would be 3.4x | **does not fit in 16 GB RAM** |

## Expectation

**Qwen3-30B-A3B is worth trying.** On the 6 GB desktop the same model at a 2.2x ratio measured
**1.52x faster** than `--cpu-moe` - the largest win recorded anywhere in this project. The
laptop gets a slightly better 2.9x ratio, though its weaker CPU and GPU move both sides of the
comparison, so treat the desktop result as indicative rather than a prediction.

Start with the baseline so you have something to compare against:

```
llama-cli -m Qwen3-30B-A3B-Q2_K.gguf -ngl 99 --cpu-moe
```

`--cpu-moe` keeps attention on the GPU and experts on the CPU, and beat plain `-ngl` by 2x on
every MoE tested. Then add `--bells-slots -1` and compare.

## If you want to test BELLS anyway

```
llama-bells-profile -m model.gguf -f corpus.txt -o out.json -c 512 -n 128 \
    -ngl 99 --cpu-moe --bells-slots -1
```

`--bells-slots -1` sizes the cache from free VRAM and prints a fit verdict at load.

Then compare against the baseline **in the same session, alternating runs**. Absolute
throughput drifts 30%+ over tens of minutes on a warm machine; comparing a number from one
session against another is how the earlier drafts of RESULTS.md ended up overstating the
speedup by 20 points.

Check the output did not degenerate before believing any speedup:

```
python tools/bells-profile/bells_degen.py out.trace.bin
```

A collapsed generation touches almost no distinct experts, so the cache hits ~100% and the
technique looks fastest exactly when the model is broken.
