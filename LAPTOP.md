# Running on a GTX 1050 (4 GB VRAM, 16 GB RAM)

> **Read this first: BELLS is very unlikely to help on this machine, and this page originally
> said the opposite.**
>
> On a 6 GB desktop the same technique measures **0.94-0.97x through `llama-server`** - slower
> than plain `--cpu-moe`. Two reasons, both worse on 4 GB. The expert cache and the KV cache
> come out of the same VRAM and the KV cache is allocated first, so a usable context leaves
> almost nothing for a cache. And the cache holds its VRAM whether or not it is used, which on a
> small card costs ~5% unconditionally by squeezing the compute buffers.
>
> Every positive result in this project is on a 24 GB card, which can hold a 16K context and a
> large cache at the same time. Four gigabytes cannot.
>
> **Use `--cpu-moe` on its own.** It is stock llama.cpp, needs nothing from this fork, and is
> worth ~2x over plain `-ngl`. The rest of this page is kept for anyone who wants to measure it
> anyway.

Build: compiled for Pascal (`CMAKE_CUDA_ARCHITECTURES=61`). The `build-bells/` binaries will not
run on this GPU - they are `sm_86` only.

## Measured on the hardware

**Correctness passes.** `llama-bells-selftest` was built and run on a real GTX 1050 Mobile
(Debian 13, gcc 14.2, CUDA 12.4, driver 550.163.01). The cache path is bit-identical to the full
expert tensor, the mid-graph slot update is visible to the gather, 16000 residency states hold,
and the runtime loop matches exactly. **sm_61 is a supported target** - three architectures now
validated (sm_61, sm_75, sm_86).

**Host-to-device bandwidth is the problem.** Same benchmark, both machines:

| | pageable | pinned | PCIe link |
|---|---|---|---|
| RTX 2060 desktop | 9.0 GB/s | **12.25 GB/s** | 3.0 x16 |
| GTX 1050 laptop | 3.11 GB/s | **3.14 GB/s** | **3.0 x4 (downgraded from x16)** |

`lspci` confirms it: `LnkCap: Width x16` but `LnkSta: Width x4 (downgraded)`. The slot is wired
with a quarter of the lanes, which is normal for a laptop dGPU and matches the measurement
(PCIe 3.0 x4 is ~3.94 GB/s theoretical). Note also that pinned memory buys nothing here - 3.11
vs 3.14 - where it is worth ~1.4x on the desktop.

**Why that decides it.** Instrumentation shows copies are ~87% of BELLS's per-layer cost, so a
4x slower link makes the dominant term 4x worse. The selftest's own transfer projection for
Qwen3-30B-A3B at a 2x cache ratio: **81.3 ms/token on this laptop against 20.8 ms on the
desktop**, transfer alone, assuming perfect overlap. And the desktop at 6 GB already measures
0.94-0.97x through `llama-server`.

So there are now two independent reasons, both measured rather than argued: not enough VRAM to
fund a cache and a real context together, and a PCIe link a quarter as wide as the machine where
it already fails.

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

**Low.** This section previously said Qwen3-30B-A3B was "worth trying" on the strength of a
1.52x measured on the 6 GB desktop, and called it the largest win in the project. Both claims
are now retracted. That 1.52x came from `llama-bells-profile` at `-c 512` - pure decode, no
prefill, no chat template, and a large cache precisely because a 512-token context leaves VRAM
free. Through `llama-server` at `-c 4096` the same model on the same card measures 0.94-0.97x.

The cache ratios in the table above still say whether a cache is physically possible. They do
not predict a speedup - a 7.8x ratio produced the worst result in this project.

One thing does predict, and it is the reason to be pessimistic here: BELLS converges on a
**CPU-independent** throughput, because once experts are resident the work is on the GPU. So the
question is whether that throughput beats your `--cpu-moe` baseline. A GTX 1050 caps the BELLS
side hard while the CPU side keeps working, which is the wrong shape.

Start with the baseline, which is very likely also the finish:

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
