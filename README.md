# BELLS

**Run 177B models on consumer GPUs. No quality loss.**

BELLS (**B**uffered **E**xpert **L**oading with **L**ayered **S**lots) is a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork that adds a per-layer VRAM expert cache for Mixture-of-Experts models. Instead of keeping all experts in VRAM (impossible on consumer cards) or running them entirely on the CPU (slow), BELLS caches the hot experts on your GPU and streams the rest from RAM or NVMe on demand.

Every expert the router picks still gets computed on the GPU — nothing is skipped, pruned, or approximated. Output is bit-identical to full-GPU inference.

## Install

### One-command install

**Linux / macOS:**
```sh
curl -sSL https://raw.githubusercontent.com/DGuckert/llama.cpp-BELLS/bells-next/install-bells.sh | bash
```

**Windows (PowerShell):**
```powershell
iex (irm https://raw.githubusercontent.com/DGuckert/llama.cpp-BELLS/bells-next/install-bells.ps1)
```

The installer automatically:
1. Detects your GPU (NVIDIA/AMD/Intel) and available SDKs (CUDA, Vulkan)
2. Downloads precompiled binaries if available, otherwise builds from source
3. Installs the `bells` command to your PATH

After installation, run `bells system` to verify your setup.

### Requirements

- **Python 3.10+** — for the installer and tools
- **Git** — for cloning the repository
- **CMake** — only needed if building from source (no precompiled binary for your platform)
- **GPU SDK** (one of):
  - NVIDIA: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  - AMD: [ROCm](https://rocm.docs.amd.com/) (Linux) or Vulkan SDK
  - Any GPU: [Vulkan SDK](https://vulkan.lunarg.com/)

### Manual build

If you prefer to build manually or the installer doesn't cover your setup:

```sh
git clone https://github.com/DGuckert/llama.cpp-BELLS.git
cd llama.cpp-BELLS

# NVIDIA (CUDA)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# AMD (ROCm / HIP)
cmake -B build -DGGML_HIP=ON
cmake --build build --config Release

# AMD / Intel / any Vulkan GPU
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

## Benchmarks

| Model | GPU | RAM | Without BELLS | With BELLS | Speedup |
|---|---|---|---:|---:|---:|
| Qwen3.6-35B Q4 | RTX 3060 12 GB | 32 GB DDR4 | 32.5 tok/s | **60.9 tok/s** | 1.9x |
| Qwen3.6-35B Q6 | RTX 3060 12 GB | 32 GB DDR4 | 22.1 tok/s | **42.8 tok/s** | 1.9x |
| Qwen3.6-35B Q4 | RTX 2060 6 GB | 32 GB DDR4 | — | **36.3 tok/s** | — |
| Qwen3-30B-A3B Q4 | RTX 2060 6 GB | 32 GB DDR4 | 22.4 tok/s | **28.3 tok/s** | 1.3x |
| Flash-Next 177B Q2 | RTX 3060 12 GB | 32 GB DDR4 | 14.7 tok/s | **32.9 tok/s** | 2.2x |

Models streamed from NVMe via mmap. The 177B model (65 GB quantised) doesn't even fit in 32 GB RAM — the page cache handles it transparently.

"Without BELLS" means `--cpu-moe -ngl 99` (all attention on GPU, all experts on CPU). The baseline for models that don't fit in VRAM at all is CPU-only, which is far slower.

## Quick start

### One-flag setup

```sh
# --auto detects MoE models and configures everything
llama-server -m model.gguf --auto -c 4096 -fa
llama-cli    -m model.gguf --auto -c 4096
```

`--auto` enables `--cpu-moe-pinned` and `--bells` with auto-sizing. For dense (non-MoE) models it's a harmless no-op. The auto-sizer queries free VRAM, computes how many expert slots fit, then probes upward to find the actual maximum — it picks the right slot count for your hardware without any manual tuning.

### Manual setup

```sh
# auto-sized cache with explicit memory mode
llama-server -m model.gguf -ngl 99 --cpu-moe --bells -fa -c 4096

# set the cache size yourself
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots 80 -fa -c 4096

# pinned memory — faster copies when the model fits in RAM
llama-server -m model.gguf -ngl 99 --cpu-moe-pinned --bells-slots 80 -fa -c 4096
```

`--bells-slots` controls how many experts per layer stay resident on the GPU. More slots = more VRAM = fewer cache misses = faster inference. Start high and lower it if you get an OOM.

### Recommended slot counts

| VRAM | Slots | Notes |
|---|---|---|
| 6 GB | 26–30 | Use `--auto` or `--bells` for best results |
| 8 GB | 50–60 | Enough for most 30B-class models |
| 12 GB | 80–100 | Sweet spot for Qwen3.6-35B |
| 16 GB | 100–120 | Room for larger context windows |
| 24 GB | 150–200 | Can cache most active experts |

These assume the rest of the model (attention layers, KV cache) is already allocated. The auto-sizer handles this automatically.

## Supported models

BELLS works with any MoE model in GGUF format. Tested and tuned for:

| Model | Parameters | Experts | Active | File sizes |
|---|---|---|---|---|
| Qwen3-30B-A3B | 30B total, 3B active | 128 per layer | 8 | Q4: 18 GB, Q6: 24 GB |
| Qwen3.6-35B | 35B total, ~5B active | 64 per layer | 4 | Q4: 23 GB, Q6: 28 GB |
| Flash-Next 177B | 177B total | 128 per layer | 8 | IQ3: 65 GB, Q2: 70 GB |
| DeepSeek-V3 | 671B total | 256 per layer | 8 | IQ3: 100 GB, Q4: 190 GB |

Dense models (Llama, Mistral, Phi, etc.) are unaffected by BELLS flags.

## All flags

| Flag | Description |
|---|---|
| `--auto` | One-flag setup. Detects MoE models and enables `--cpu-moe-pinned` + `--bells` with auto-sizing. No-op for dense models. |
| `--cpu-moe` | Keep all MoE expert weights on the CPU (via mmap). Required for BELLS. Use this when the model is too large to fit in RAM — the OS page cache will stream from NVMe. |
| `--cpu-moe-pinned` | Like `--cpu-moe`, but pin expert weights in physical RAM (no mmap). Faster host-to-device copies because pinned memory enables async DMA. Requires the model's expert weights to fit in RAM. |
| `--bells` | Enable BELLS with auto-sized cache. Equivalent to `--bells-slots -1`. |
| `--bells-slots N` | Cache N experts per layer in VRAM. Set to `-1` to auto-size from free VRAM. More slots = higher hit rate = faster. With multiple GPUs, auto-sizing gives each device its own slot count. |
| `--bells-l2-slots N` | Use an idle secondary GPU's VRAM as an L2 victim cache. N experts per layer on GPU 2. `-1` to auto-size. Ignored on single-GPU systems and when both GPUs are actively computing. |
| `--bells-split K` | Run K of each token's experts on GPU, the rest on CPU concurrently. The MoE output is a weighted sum, so splitting is exact — no quality loss. Fewer experts need to be in the cache. Default: 0 (all through cache). |
| `--bells-refresh N` | Observe a rotating 1/N of MoE layers per token instead of every layer. Reduces graph-split overhead (~2.3 ms/token across 32 layers). Default: 1 (every layer). |
| `--bells-cache-type TYPE` | Store cached experts at a different quantisation than the model (e.g. `q2_K` when the model is `q4_K`). Fits more experts in the same VRAM at the cost of re-quantisation overhead. Experimental. |
| `--bells-passive` | Research only. Allocate the cache and perform graph splits, but don't copy any experts. Measures mechanism overhead in isolation. |

### Environment variables

| Variable | Description |
|---|---|
| `GGML_CUDA_NO_PINNED=1` | Disable CUDA pinned (page-locked) host memory allocation. Use this if you see corrupted output or NaN on systems with IOMMU/DMA issues. |

### Flag ordering

`-ot` (tensor override) rules must come **before** `--cpu-moe` on the command line. The first matching override wins, so an `-ot` placed after `--cpu-moe` is silently ignored.

```sh
# correct — -ot overrides the first 8 expert layers to GPU
llama-server -m model.gguf -ot "blk\.[0-7]\.ffn_(gate|up|down)_exps\.weight=CUDA0" --cpu-moe --bells -fa

# wrong — -ot is ignored because --cpu-moe already matched
llama-server -m model.gguf --cpu-moe -ot "blk\.[0-7]\.ffn_(gate|up|down)_exps\.weight=CUDA0" --bells -fa
```

## Multi-GPU

BELLS supports two multi-GPU modes depending on your setup.

### Active caching (both GPUs compute)

When attention layers are split across GPUs with `--device`, BELLS sizes each GPU's expert cache independently based on its own free VRAM. A 24 GB card and a 6 GB card in the same system each get the slot count that fits, instead of both being limited to the smaller card.

```sh
# 2x NVIDIA GPUs — each auto-sized independently
llama-server -m model.gguf -ngl 99 --device CUDA0,CUDA1 --cpu-moe --bells -fa

# 2x AMD GPUs
llama-server -m model.gguf -ngl 99 --device ROCm0,ROCm1 --cpu-moe --bells -fa
```

Both GPUs run attention for their assigned layers. BELLS auto-detects that the second GPU is active and skips L2 mode (which would steal VRAM from productive work).

### L2 victim cache (second GPU as storage)

When only one GPU runs compute, a second GPU can donate its VRAM as an overflow cache.

```sh
# auto-size L2 from GPU 2's free VRAM
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots 80 --bells-l2-slots -1 -fa

# or set L2 size explicitly
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots 80 --bells-l2-slots 200 -fa
```

When an expert is evicted from L1 (primary GPU), it goes to L2 (secondary GPU) instead of being discarded. Next time it's needed, it copies from GPU 2 VRAM — a fast device-to-device transfer instead of a page fault from NVMe.

## How it works

```
                ┌──────────────────────────────────────────┐
                │              BELLS cache                 │
                │                                          │
  Router      ┌──────────┐  evict   ┌──────────┐          │
  picks       │ L1 cache │ ──────▶  │ L2 cache │          │
  expert E    │ GPU 1    │ ◀──────  │ GPU 2    │          │
  ──────────▶ │(compute) │ promote  │(storage) │          │
              └──────────┘          └──────────┘          │
                   │ miss                │ miss            │
                   ▼                     ▼                 │
              ┌──────────┐         ┌──────────┐           │
              │ Host RAM │         │  (skip)  │           │
              │ or NVMe  │         │          │           │
              └──────────┘         └──────────┘           │
              (may page fault)                             │
              └────────────────────────────────────────────┘

Multi-GPU active mode (--device GPU0,GPU1):

  Router      ┌──────────┐          ┌──────────┐
  picks       │ GPU 0    │          │ GPU 1    │
  expert E    │ 80 slots │          │ 40 slots │
  ──────────▶ │ layers   │          │ layers   │
              │  0–15    │          │  16–31   │
              └──────────┘          └──────────┘
                   │ miss                │ miss
                   ▼                     ▼
              ┌──────────────────────────────────┐
              │    Host RAM / NVMe (experts)     │
              └──────────────────────────────────┘
```

### The cache hierarchy

1. **L1 (GPU VRAM)** — the working cache. Experts are loaded here for compute. Uses LRU eviction with per-layer clock counters. With multiple active GPUs, each device gets its own independent slot count.

2. **L2 (secondary GPU, optional)** — a victim cache for single-compute-GPU setups. When L1 evicts an expert, it goes to L2 instead of being discarded. L2 uses its own LRU policy. Automatically disabled when both GPUs are active.

3. **Host memory** — the cold tier. Expert weights served via mmap (or pinned allocation with `--cpu-moe-pinned`). When the model doesn't fit in RAM, the OS page cache handles NVMe paging transparently.

### Data flow for a single expert load

```
Token arrives → Router selects experts → For each expert not in L1:

  1. EVICTION:  Pick least-recently-used slot in this layer's cache
                If L2 exists: demote victim → staging buffer → write to L2 slot

  2. L2 CHECK:  Look up requested expert in L2 index
                  HIT  → read from L2 → staging → write to L1 slot → done
                  MISS → fall through to cold path

  3. COLD PATH: Read from host memory → write to L1 slot
                (may page fault from NVMe if model exceeds RAM)
```

### Key design decisions

- **Per-device slot sizing.** Each GPU gets the slot count its own free VRAM supports. A 24 GB and 6 GB card in the same system aren't bottlenecked by the smaller card.
- **Auto-sizing with upward probing.** The auto-sizer computes an initial estimate from free VRAM, then probes upward with real allocations to find the actual maximum. It checks each probe against physical VRAM to avoid over-allocation into managed memory.
- **Smart L2 detection.** L2 is automatically skipped when the candidate GPU already hosts BELLS cache layers. No manual flag juggling for multi-GPU setups.
- **Per-layer indexing.** Each layer maintains its own slot map (`expert → slot`). This matches how MoE routing works and avoids cross-layer eviction interference.
- **Victim cache semantics.** L2 only receives data evicted from L1 (or promoted back). It never loads directly from host. This keeps the L2 population naturally tuned to the model's actual access pattern.
- **Backward compatible.** Single-GPU systems behave identically to stock llama.cpp. Dense models are completely unaffected.

## Performance estimator

Estimate BELLS performance on your hardware before downloading a model:

```sh
cd tools/bells-manager

# what will an RTX 5060 Ti get on Flash-Next 177B?
python bells.py estimate run -g "5060 Ti" -m flash-next-177b -q Q2_K

# try different configurations
python bells.py estimate run -g "3060" -m qwen3.6-35b -q Q4_K_M -c 8192 --ram 64 --ram-speed ddr5-6000

# list all supported GPUs and models
python bells.py estimate gpus
python bells.py estimate models
```

The estimator models cache hit rates, transfer bandwidth (including page-cache-aware mmap behaviour), and pipeline overlap. It outputs projected tok/s with and without BELLS, plus recommended flags.

### Supported hardware in the estimator

Over 50 GPUs are in the database, including:
- **NVIDIA Desktop:** RTX 20-series through 50-series
- **NVIDIA Laptop:** RTX 3060–5080 Laptop variants (with adjusted bandwidth)
- **NVIDIA Data Center:** A10, A100, L4, L40
- **AMD Desktop:** RX 7600 through 9070 XT
- **Intel:** Arc A770, Arc B580

Custom GPUs can be specified with `--gpu-vram` and `--gpu-bw` overrides.

## Benchmarking tool

Find the optimal BELLS configuration for your system automatically:

```sh
cd tools/bells-manager

# quick benchmark — tests key configurations
python bells.py bench -m /path/to/model.gguf -p short

# thorough benchmark
python bells.py bench -m /path/to/model.gguf -p long -c 8192

# just list detected models
python bells.py models

# print system info (GPUs, RAM, detected binaries)
python bells.py system
```

The benchmark engine explores flag combinations (baseline, `--cpu-moe`, `--cpu-moe-pinned`, various `--bells-slots` values) and reports the fastest configuration. Results are saved as JSON for later comparison.

<!--
## Manager UI

BELLS includes a TUI (terminal UI) and GUI for interactive configuration, benchmarking, and server management:

```sh
cd tools/bells-manager

# terminal UI (requires: pip install textual)
python bells.py

# desktop GUI (requires: pip install pywebview)
python bells.py --gui
```

Features:
- System info dashboard (GPU, RAM, CPU detection)
- Model browser with Hugging Face search
- Interactive server launcher with log streaming
- Benchmark runner with live progress
- Configuration builder with recommended flags

### Requirements

```sh
pip install textual           # for TUI
pip install pywebview          # for desktop GUI
pip install huggingface_hub    # for model search
```
-->

## Tips and troubleshooting

### Choosing between `--cpu-moe` and `--cpu-moe-pinned`

| | `--cpu-moe` | `--cpu-moe-pinned` |
|---|---|---|
| Memory mode | mmap (virtual, backed by page cache) | Pinned in physical RAM |
| Model fits in RAM | Slightly slower copies | Fastest copies (async DMA) |
| Model exceeds RAM | Works — OS pages from NVMe transparently | Will exhaust RAM and swap, don't use |
| Use when | Model > RAM, or you want to share RAM with other apps | Model fits comfortably in RAM and you want max throughput |

### Context window vs BELLS cache

VRAM is shared between the KV cache (scales with context length) and the BELLS expert cache (scales with slot count). Longer context = less room for expert slots = lower hit rate. If you need long context, consider KV quantisation:

```sh
# halves KV cache VRAM usage, freeing room for more expert slots
llama-server -m model.gguf --auto --cache-type-k q8_0 --cache-type-v q8_0 -c 16384 -fa
```

### Common issues

**"failed to allocate expert slots"** — Not enough free VRAM after loading attention layers and KV cache. Reduce context (`-c`), use KV quantisation, or close other GPU applications.

**Output looks corrupted or produces NaN** — On some systems, CUDA pinned (page-locked) memory allocation returns bad data due to IOMMU or DMA issues. Set `GGML_CUDA_NO_PINNED=1` to disable pinned host memory.

**Server starts but BELLS isn't active** — Make sure you're using a MoE model. BELLS is silently disabled for dense models. Check stderr for `init:` lines confirming BELLS loaded with a slot count > 0.

**Lower throughput than expected** — Make sure the system is idle (no games, encoders, or other GPU-heavy apps). Background processes sharing VRAM reduce the slots available. Use `nvidia-smi` or equivalent to check free VRAM before launching.

**`-ot` overrides being ignored** — They must come before `--cpu-moe` on the command line. See [Flag ordering](#flag-ordering).

## How BELLS compares to other approaches

| Approach | Quality | Speed | VRAM needed |
|---|---|---|---|
| Full GPU offload | Lossless | Fastest | All experts must fit |
| CPU-only MoE (`--cpu-moe`) | Lossless | Slow (memory-bound) | Only attention layers |
| Expert pruning/dropping | Lossy | Fast | Reduced | 
| **BELLS** | **Lossless** | **Fast (cache-dependent)** | **Configurable (slots)** |

BELLS is the only approach that maintains full output quality while running models that exceed VRAM. It sits between "everything on GPU" (impossible for large models) and "everything on CPU" (too slow for interactive use).

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

## Quick start

```sh
# Download and run a model directly from Hugging Face
llama-cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/Qwen3.5-0.8B-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware — locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen — optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- 1.5-bit through 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for NVIDIA GPUs (AMD via HIP, Moore Threads via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than VRAM capacity

Built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [CUDA](docs/build.md#cuda) | NVIDIA GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Vulkan](docs/build.md#vulkan) | GPU (cross-vendor) |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [BLAS](docs/build.md#blas-build) | All |

See the upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) for the full backend list and build instructions.

## License

MIT — see [LICENSE](LICENSE).
