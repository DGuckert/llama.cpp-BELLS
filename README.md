# BELLS — a per-expert VRAM cache for MoE models

> This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). The upstream README follows
> below. Everything in this section is specific to the fork.

A Mixture-of-Experts model activates a few experts per token but has to *store* all of them. When
the expert weights do not fit in VRAM, llama.cpp keeps them in host memory (`--cpu-moe`) and the
expert matmuls run on the CPU. BELLS instead keeps the **N hottest experts per layer resident in
VRAM** and rewrites the routing ids to index that small cache, so most expert work runs on the GPU
and only misses are fetched.

Routing is read back at each MoE layer, the cache is updated, and an expert→slot table is uploaded
for the graph to index through. Non-resident experts are redirected to a spare zeroed slot rather
than out of bounds.

## Measured results

RTX 3060 12 GB, 32 GB RAM, Qwen3.6-35B-A3B Q4_K_M (40 layers, 256 experts, 8 active, 21 GB of
weights of which 19.5 GB are experts):

| configuration | decode | prefill |
|---|---|---|
| `--cpu-moe-pinned` alone | 35.7 tok/s | — |
| **+ BELLS 88 slots + 8 layers via `-ot`** | **~70 tok/s** | **~545 tok/s** |

**BELLS is worth ~2x on this model, losslessly** — quality is identical to running without it,
because every routed expert is made resident before the matmul reads it.

It is **not** universally a win. On Qwen3.8-Flash-Next (177B, 512 experts per layer) a 12 GB card
can only cover ~12% of the routing space, so nearly every token misses and pays a host→device copy
that plain `--cpu-moe` never pays. BELLS measured *slower* than baseline there at every slot count
tested. **The dividing line is cache coverage as a fraction of the routing space**, not model size.

## Flags

| flag | what it does |
|---|---|
| `--bells-slots N` | keep N experts per layer resident. Use with `--cpu-moe` / `--cpu-moe-pinned` |
| `--bells` | same, sized automatically from free VRAM |
| `--bells-passive` | allocate the cache and take the readback, but never use it — isolates the fixed cost of the mechanism |
| `--moe-stats FILE` | write a CSV of how often, and with how much routing weight, each expert is used |
| `--pin-experts FILE` | seat the measured-hottest experts permanently, from a `--moe-stats` CSV |
| `--bells-refresh N` | **research only, degrades output** — observe only every Nth layer |

Working configuration for the model above:

```sh
llama-server -m model.gguf -ngl 99 -c 65536 -t 8 \
  -ot "blk\.[0-7]\.ffn_.*_exps=CUDA0" --cpu-moe-pinned --bells-slots 88 \
  -fa on -ctk q8_0 -ctv q8_0
```

`-ot` **must come before** `--cpu-moe`: the first matching tensor-override rule wins, so an `-ot`
placed after it is silently ignored.

## Things worth knowing

- **Context allocation is nearly free, until it isn't.** `-c` of 8k/32k/64k all keep the full cache
  and decode identically. At `-c 131072` the KV allocation squeezes the cache out entirely and
  decode halves. After changing `-c`, check the `slots/layer of` line still appears.
- **Decode degrades with *used* context, prefill barely does** — 69 tok/s empty, 61 at 16k, 51 at
  40k, against prefill 545 → 517. Decode re-reads the whole KV every token.
- **Prefill takes no graph splits.** BELLS only serves small ubatches, so the routing read during
  prefill was discarded; skipping it is worth ~20% prefill and costs nothing.
- **Prefetching cannot help much here.** Instrumented per layer-call: readback ~13 µs, copy ~7 µs,
  upload ~13 µs. Copy is the only part that scales with misses and it is ~0.25 ms/token of a
  ~14.7 ms budget, so a perfect predictor buys under 2%.
- **A better expert-selection heuristic does not exist.** Ranking a static table by summed routing
  weight instead of by occurrence count gains +0.9%; the two are near-perfectly correlated because
  every pick averages ~1/8 of the weight.

Set `GGML_SCHED_DEBUG=2` with `-lv 10` to see per-node backend assignment — the fastest way to
answer "where is this op actually running".

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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

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
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
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
