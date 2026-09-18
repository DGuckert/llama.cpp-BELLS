# BELLS

**Run 177B models on a single GPU. No quality loss.**

BELLS is a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork that adds a per-layer VRAM expert cache for Mixture-of-Experts models. Instead of keeping all experts in VRAM (impossible) or running them on the CPU (slow), BELLS caches the hot experts on your GPU and streams the rest from RAM or NVMe on demand. Every expert the router picks still gets computed — nothing is skipped or approximated.

### What you get

| model | GPU | without BELLS | with BELLS |
|---|---|---:|---:|
| Qwen3.6-35B Q4 | RTX 3060 12 GB | 32.5 tok/s | **60.9 tok/s** |
| Qwen3.6-35B Q4 | RTX 2060 6 GB | — | **36.3 tok/s** |
| Flash-Next 177B Q2 | RTX 3060 12 GB | 14.7 tok/s | **32.9 tok/s** |

32 GB DDR4, models streamed from NVMe. The 177B doesn't even fit in RAM.

### Build

```sh
# NVIDIA
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release

# AMD / Intel / any Vulkan GPU
cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release
```

### Use

```sh
# just works — auto-sizes the cache to your VRAM
llama-server -m model.gguf -ngl 99 --cpu-moe --bells -fa

# or set the cache size yourself
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots 80 -fa -c 4096

# pinned memory is faster when the model fits in RAM
llama-server -m model.gguf -ngl 99 --cpu-moe-pinned --bells-slots 80 -fa -c 4096
```

`--bells-slots` controls how many experts per layer stay resident on the GPU. More slots = more VRAM = fewer misses = faster. Start high and lower it if you run out of memory.

**Note:** `-ot` rules must come **before** `--cpu-moe` on the command line. The first matching override wins, so an `-ot` placed after `--cpu-moe` is silently ignored.

| VRAM | start with |
|---|---|
| 6 GB | `--bells-slots 30` |
| 8 GB | `--bells-slots 60` |
| 12 GB | `--bells-slots 100` |
| 16 GB | `--bells-slots 120` |
| 24 GB | `--bells-slots 200` |

### All flags

| flag | description |
|---|---|
| `--cpu-moe` | Keep all MoE expert weights on the CPU. Required for BELLS. |
| `--cpu-moe-pinned` | Like `--cpu-moe`, but pin expert weights in host memory (no mmap). Faster copies, but the model must fit in RAM. Don't use when streaming from NVMe. |
| `--bells` | Enable BELLS with auto-sized cache (equivalent to `--bells-slots -1`). |
| `--bells-slots N` | Cache N experts per layer in VRAM. `-1` to auto-size from free VRAM. More slots = more VRAM = fewer misses. |
| `--bells-l2-slots N` | Use a secondary GPU's VRAM as L2 victim cache. N experts per layer on GPU 2. `-1` to auto-size. Single-GPU systems ignore this. |
| `--bells-split K` | Run K of each token's experts on GPU, the rest on CPU concurrently. The MoE output is a weighted sum, so splitting is exact — no quality loss. Fewer experts need to be resident, so the cache covers more. Default: 0 (all experts through the cache). |
| `--bells-refresh N` | Observe a rotating 1/N of MoE layers per token instead of every layer. Reduces graph split overhead (~2.3 ms/token across 32 layers). Default: 1 (observe every layer). |
| `--bells-passive` | Research only. Allocate the cache and take the graph splits, but copy nothing and leave matmuls on the full expert stack. Measures mechanism overhead in isolation. |

BELLS only helps MoE models (Qwen3-30B-A3B, Qwen3.6-35B, DeepSeek-V3, Flash-Next, etc). Dense models are unaffected.

### Multi-GPU (L2 cache)

Got a second GPU? BELLS can use its VRAM as overflow cache space. All compute stays on GPU 1 — the second GPU just donates its memory.

```sh
# auto-size from GPU 2's free VRAM
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots 80 --bells-l2-slots -1 -fa

# or set L2 size explicitly
llama-server -m model.gguf -ngl 99 --cpu-moe --bells-slots 80 --bells-l2-slots 200 -fa
```

When an expert gets evicted from the primary cache (L1), it goes to L2 on the second GPU instead of being thrown away. Next time that expert is needed, it comes back from GPU 2 VRAM — a deterministic copy, not a page fault from NVMe. On a system where the model is streaming from disk, this is the difference between microseconds and milliseconds.

### How it works

```
                    ┌─────────────────────────────────────────┐
                    │              BELLS cache                │
                    │                                         │
  Router picks   ┌──────────┐  evict   ┌──────────┐         │
  expert E       │ L1 cache │ ──────── │ L2 cache │         │
  ─────────────▶ │  GPU 1   │ ◀─────── │  GPU 2   │         │
                 │ (compute)│ promote  │ (storage)│         │
                 └──────────┘          └──────────┘         │
                      │ miss               │ miss            │
                      │                    │                 │
                      ▼                    ▼                 │
                 ┌──────────┐        ┌──────────┐           │
                 │ Host RAM │        │  (skip)  │           │
                 │ or NVMe  │        │          │           │
                 └──────────┘        └──────────┘           │
                 (may page fault)                            │
                 └─────────────────────────────────────────┘
```

**BELLS is a per-layer VRAM expert cache.** MoE models have hundreds of expert weight matrices spread across dozens of layers, but the router only picks 2–8 per token. Most experts sit idle. BELLS keeps the hot ones in GPU VRAM and streams the rest on demand.

#### The cache hierarchy

1. **L1 (primary GPU)** — the working cache. Experts are loaded here for compute. Sized by `--bells-slots`. Uses LRU eviction with per-layer clock counters.

2. **L2 (secondary GPU)** — a victim cache. When L1 evicts an expert, it goes to L2 instead of being discarded. L2 uses its own LRU policy. Sized by `--bells-l2-slots` (or `-1` for auto, which leaves 512 MB headroom on GPU 2).

3. **Host memory** — the cold tier. Models backed by mmap. Page faults here are the most expensive operation — especially when the model doesn't fit in RAM and pages from NVMe.

#### Data flow for a single expert load

```
Token arrives → Router selects experts → For each expert not in L1:

  1. EVICTION:  Read victim from L1 slot → staging buffer → write to L2 slot
                (GPU1 VRAM → host pinned → GPU2 VRAM)

  2. L2 CHECK:  Look up requested expert in L2 index
                  HIT  → read from L2 → staging → write to L1 slot → done
                  MISS → fall through to cold path

  3. COLD PATH: Read from host mmap → write to L1 slot
                (may page fault from NVMe — this is what L2 eliminates)
```

The staging buffer is a host-side pinned allocation sized to one expert. The eviction read happens **before** the slot is overwritten, so it's always a clean VRAM read — no page faults, no blocking.

#### Key design decisions

- **No compute on GPU 2.** The matmul graph only ever references L1 slots. GPU 2 is invisible to the compute path — it's a dumb buffer with a lookup table.
- **Per-layer indexing.** Each layer maintains its own L2 slot map (`expert → slot`). This matches the L1 design and avoids cross-layer eviction interference.
- **Victim cache semantics.** L2 only receives data evicted from L1 (or promoted back). It never loads directly from host. This keeps the L2 population naturally tuned to the model's access pattern.
- **Backward compatible.** On a single-GPU system, L2 is a no-op. The `bells_copy` struct gained an `evicted` field with a default of `-1`, so all existing code paths are unchanged.
- **Stats tracking.** BELLS reports L1 hit rate, L2 hit rate, total admits, and total promotions. When L2 is enabled, the periodic timing output shows both tiers.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
