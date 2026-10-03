"""BELLS performance estimator — predict tok/s and recommend flags for any hardware.

Calibrated against measured benchmarks on RTX 3060/2060. Estimates are ±20%
at short context, wider at very long context (attention modeling is approximate).
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# GPU database — memory bandwidth is the primary decode bottleneck
# ---------------------------------------------------------------------------

@dataclass
class GpuSpec:
    name: str
    vram_mb: int
    bw_gbs: float  # memory bandwidth GB/s (theoretical peak)

GPUS: dict[str, GpuSpec] = {
    # NVIDIA — Desktop
    "rtx 2060":            GpuSpec("RTX 2060",          6144,   336),
    "rtx 2060 super":      GpuSpec("RTX 2060 Super",    8192,   448),
    "rtx 2070":            GpuSpec("RTX 2070",           8192,   448),
    "rtx 2070 super":      GpuSpec("RTX 2070 Super",     8192,   448),
    "rtx 2080":            GpuSpec("RTX 2080",           8192,   448),
    "rtx 2080 ti":         GpuSpec("RTX 2080 Ti",       11264,   616),
    "rtx 3060":            GpuSpec("RTX 3060",          12288,   360),
    "rtx 3060 ti":         GpuSpec("RTX 3060 Ti",        8192,   448),
    "rtx 3070":            GpuSpec("RTX 3070",           8192,   448),
    "rtx 3070 ti":         GpuSpec("RTX 3070 Ti",        8192,   504),
    "rtx 3080 10gb":       GpuSpec("RTX 3080 10GB",     10240,   760),
    "rtx 3080 12gb":       GpuSpec("RTX 3080 12GB",     12288,   912),
    "rtx 3090":            GpuSpec("RTX 3090",          24576,   936),
    "rtx 3090 ti":         GpuSpec("RTX 3090 Ti",       24576,  1008),
    "rtx 4060":            GpuSpec("RTX 4060",           8192,   272),
    "rtx 4060 ti 8gb":     GpuSpec("RTX 4060 Ti 8GB",   8192,   288),
    "rtx 4060 ti 16gb":    GpuSpec("RTX 4060 Ti 16GB", 16384,   288),
    "rtx 4070":            GpuSpec("RTX 4070",          12288,   504),
    "rtx 4070 super":      GpuSpec("RTX 4070 Super",    12288,   504),
    "rtx 4070 ti":         GpuSpec("RTX 4070 Ti",       12288,   504),
    "rtx 4070 ti super":   GpuSpec("RTX 4070 Ti Super", 16384,   672),
    "rtx 4080":            GpuSpec("RTX 4080",          16384,   717),
    "rtx 4080 super":      GpuSpec("RTX 4080 Super",    16384,   736),
    "rtx 4090":            GpuSpec("RTX 4090",          24576,  1008),
    "rtx 5060":            GpuSpec("RTX 5060",           8192,   448),
    "rtx 5060 ti":         GpuSpec("RTX 5060 Ti",       16384,   448),
    "rtx 5070":            GpuSpec("RTX 5070",          12288,   672),
    "rtx 5070 ti":         GpuSpec("RTX 5070 Ti",       16384,   896),
    "rtx 5080":            GpuSpec("RTX 5080",          16384,   960),
    "rtx 5090":            GpuSpec("RTX 5090",          32768,  1792),
    # NVIDIA — Laptop (lower bandwidth due to power limits)
    "rtx 3060 laptop":     GpuSpec("RTX 3060 Laptop",   6144,   264),
    "rtx 4060 laptop":     GpuSpec("RTX 4060 Laptop",   8192,   256),
    "rtx 4070 laptop":     GpuSpec("RTX 4070 Laptop",   8192,   256),
    "rtx 4080 laptop":     GpuSpec("RTX 4080 Laptop",  12288,   432),
    "rtx 4090 laptop":     GpuSpec("RTX 4090 Laptop",  16384,   576),
    "rtx 5060 ti laptop":  GpuSpec("RTX 5060 Ti Laptop", 12288, 384),
    "rtx 5070 laptop":     GpuSpec("RTX 5070 Laptop",   8192,   384),
    "rtx 5070 ti laptop":  GpuSpec("RTX 5070 Ti Laptop", 12288, 504),
    "rtx 5080 laptop":     GpuSpec("RTX 5080 Laptop",   16384,  576),
    # NVIDIA — Data Center
    "a10":                 GpuSpec("A10",               24576,   600),
    "a100 40gb":           GpuSpec("A100 40GB",         40960,  1555),
    "a100 80gb":           GpuSpec("A100 80GB",         81920,  2039),
    "l4":                  GpuSpec("L4",                24576,   300),
    "l40":                 GpuSpec("L40",               49152,   864),
    # AMD — Desktop
    "rx 7600":             GpuSpec("RX 7600",            8192,   288),
    "rx 7700 xt":          GpuSpec("RX 7700 XT",        12288,   432),
    "rx 7800 xt":          GpuSpec("RX 7800 XT",        16384,   624),
    "rx 7900 gre":         GpuSpec("RX 7900 GRE",       16384,   624),
    "rx 7900 xt":          GpuSpec("RX 7900 XT",        20480,   800),
    "rx 7900 xtx":         GpuSpec("RX 7900 XTX",       24576,   960),
    "rx 9060 xt":          GpuSpec("RX 9060 XT",        16384,   448),
    "rx 9070":             GpuSpec("RX 9070",           12288,   608),
    "rx 9070 xt":          GpuSpec("RX 9070 XT",        16384,   672),
    # Intel
    "arc a770":            GpuSpec("Arc A770",          16384,   560),
    "arc b580":            GpuSpec("Arc B580",          12288,   456),
}


# ---------------------------------------------------------------------------
# Model database — MoE architecture params
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_experts: int           # experts per MoE layer
    n_active: int            # routed experts per token
    n_kv_heads: int          # GQA key/value heads
    d_head: int              # head dimension
    expert_ratio: float      # fraction of file size that is expert weights
    quants: dict[str, float] # quant_name -> model file size in GB

MODELS: dict[str, ModelSpec] = {
    "qwen3-30b-a3b": ModelSpec(
        name="Qwen3-30B-A3B", n_layers=48, n_experts=128, n_active=8,
        n_kv_heads=4, d_head=128, expert_ratio=0.82,
        quants={"Q4_K_M": 18.0, "Q6_K": 24.0, "Q8_0": 32.0, "IQ3_XXS": 12.0},
    ),
    "qwen3.6-35b": ModelSpec(
        name="Qwen3.6-35B", n_layers=40, n_experts=64, n_active=4,
        n_kv_heads=8, d_head=128, expert_ratio=0.75,
        quants={"Q4_K_M": 23.0, "Q6_K": 28.0, "Q8_0": 37.0, "IQ3_XXS": 15.0},
    ),
    "flash-next-177b": ModelSpec(
        name="Flash-Next 177B", n_layers=62, n_experts=128, n_active=8,
        n_kv_heads=8, d_head=128, expert_ratio=0.92,
        quants={"IQ3_XXS": 65.0, "Q2_K": 70.0, "Q4_K_M": 107.0},
    ),
    "deepseek-v3": ModelSpec(
        name="DeepSeek-V3", n_layers=61, n_experts=256, n_active=8,
        n_kv_heads=8, d_head=128, expert_ratio=0.90,
        quants={"IQ3_XXS": 100.0, "Q2_K": 131.0, "Q4_K_M": 190.0},
    ),
}


# ---------------------------------------------------------------------------
# RAM bandwidth reference (dual-channel, GB/s theoretical)
# ---------------------------------------------------------------------------

RAM_SPEEDS: dict[str, float] = {
    "ddr4-2400":  38.4,
    "ddr4-2666":  42.7,
    "ddr4-3200":  51.2,
    "ddr4-3600":  57.6,
    "ddr5-4800":  76.8,
    "ddr5-5200":  83.2,
    "ddr5-5600":  89.6,
    "ddr5-6000":  96.0,
    "ddr5-6400": 102.4,
    "ddr5-7200": 115.2,
    "ddr5-8000": 128.0,
}

STORAGE_SPEEDS: dict[str, float] = {
    "nvme-gen3": 3.5,
    "nvme-gen4": 7.0,
    "nvme-gen5": 14.0,
    "sata-ssd":  0.55,
}


# ---------------------------------------------------------------------------
# Calibration constants — derived from measured benchmarks
# ---------------------------------------------------------------------------

# Reference GPU: RTX 3060 (360 GB/s)
# Qwen3.6-35B Q4, BELLS 80 slots, ~4k context → 60.9 tok/s (16.4ms)
# Without BELLS → 32.5 tok/s (30.8ms)
# Flash-Next 177B Q2, BELLS, ~4k context → 32.9 tok/s (30.4ms)
# Without BELLS → 14.7 tok/s (68ms)

REF_GPU_BW = 360.0  # GB/s
REF_BASE_MS = 14.0  # non-expert GPU compute at reference bandwidth (short ctx)
GRAPH_SPLIT_MS = 2.3
BW_EFFICIENCY = 0.75  # real-world fraction of peak bandwidth
PIPELINE_OVERLAP = 0.5 # CPU→GPU transfer overlaps with GPU compute
CACHE_LOCALITY_K = 160.0  # cache_efficiency = K / n_experts (calibrated from 3060 benches)


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

@dataclass
class Hardware:
    gpu: GpuSpec
    gpu_count: int = 1
    ram_mb: int = 32768
    ram_bw_gbs: float = 51.2  # dual-channel theoretical
    storage_bw_gbs: float = 7.0
    backend: str = "auto"      # auto, cuda, rocm, vulkan

@dataclass
class Estimate:
    tok_s: float
    tok_s_no_bells: float
    attention_ms: float
    expert_ms: float
    overhead_ms: float
    bells_slots: int
    max_bells_slots: int
    hit_rate: float
    kv_cache_gb: float
    free_vram_gb: float
    model_fits_ram: bool
    flags: list[str]
    notes: list[str]


def kv_cache_gb(model: ModelSpec, ctx: int, kv_quant: str = "f16") -> float:
    bpe = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}.get(kv_quant, 2.0)
    return 2 * model.n_kv_heads * model.d_head * model.n_layers * ctx * bpe / (1024**3)


def expert_slot_gb(model: ModelSpec, quant: str) -> float:
    size = model.quants.get(quant, 0)
    total_expert = size * model.expert_ratio
    return total_expert / (model.n_layers * model.n_experts)


def vram_budget(model: ModelSpec, quant: str, hw: Hardware,
                ctx: int, kv_quant: str) -> tuple[float, float, float]:
    """(non_expert_gb, kv_gb, free_for_bells_gb) on primary GPU."""
    size = model.quants.get(quant, 0)
    non_expert = size * (1.0 - model.expert_ratio)
    kv = kv_cache_gb(model, ctx, kv_quant)
    primary_vram = hw.gpu.vram_mb / 1024

    if hw.gpu_count > 1:
        per_gpu_model = non_expert / hw.gpu_count
        per_gpu_kv = kv / hw.gpu_count
    else:
        per_gpu_model = non_expert
        per_gpu_kv = kv

    runtime_overhead = 0.5  # CUDA/ROCm runtime
    free = primary_vram - per_gpu_model - per_gpu_kv - runtime_overhead
    return non_expert, kv, max(0.0, free)


def cache_hit_rate(slots: int, model: ModelSpec) -> float:
    """Estimate cache hit rate using calibrated exponential model.

    MoE routing is heavily skewed — a small fraction of expert-layer pairs
    handle most traffic. Models with more experts have flatter distributions.
    Calibrated against RTX 3060 benchmarks for Qwen3.6-35B and Flash-Next 177B.
    """
    if slots <= 0:
        return 0.0
    active_per_token = model.n_active * model.n_layers
    cache_efficiency = CACHE_LOCALITY_K / model.n_experts
    rate = 1.0 - math.exp(-cache_efficiency * slots / active_per_token)
    return min(1.0, rate)


def _transfer_bw(model: ModelSpec, quant: str, hw: Hardware) -> float:
    """Effective expert transfer bandwidth in GB/s.

    Page cache keeps hot expert weights in RAM even when the full model
    exceeds physical memory. Up to ~2.5x RAM, page cache is nearly as
    fast as pinned host memory. Beyond that, NVMe page faults increase.
    """
    size = model.quants.get(quant, 0)
    ram_gb = hw.ram_mb / 1024
    expert_gb = size * model.expert_ratio
    if expert_gb <= ram_gb:
        return hw.ram_bw_gbs * BW_EFFICIENCY
    ratio = size / ram_gb
    if ratio <= 2.5:
        penalty = (ratio - 1.0) * 0.03
        return hw.ram_bw_gbs * BW_EFFICIENCY * (1 - penalty)
    excess = min(1.0, (ratio - 2.5) / 5.0)
    bw = (1 - excess) * hw.ram_bw_gbs + excess * hw.storage_bw_gbs
    return bw * BW_EFFICIENCY


VULKAN_BW_PENALTY = 0.90  # Vulkan achieves ~90% of native CUDA/ROCm bandwidth


def _resolve_backend(hw: Hardware) -> str:
    """Return the device prefix for --device flags."""
    if hw.backend != "auto":
        return {"cuda": "CUDA", "rocm": "ROCm", "vulkan": "Vulkan"}[hw.backend]
    name = hw.gpu.name.lower()
    if any(k in name for k in ("rx ", "radeon")):
        return "ROCm"
    if any(k in name for k in ("arc ", "intel")):
        return "Vulkan"
    return "CUDA"


def estimate(
    model_key: str,
    quant: str,
    hw: Hardware,
    ctx: int = 4096,
    kv_quant: str = "f16",
    bells_slots: int | None = None,
) -> Estimate:
    model = MODELS[model_key]
    notes: list[str] = []
    size = model.quants.get(quant)
    if size is None:
        available = ", ".join(model.quants.keys())
        raise ValueError(f"Unknown quant '{quant}' for {model.name}. Available: {available}")

    # --- VRAM budget ---
    non_exp, kv, free = vram_budget(model, quant, hw, ctx, kv_quant)
    slot_gb = expert_slot_gb(model, quant)
    max_slots = int(free / slot_gb) if slot_gb > 0 else 0

    if bells_slots is None:
        bells_slots = int(max_slots * 0.85)
    bells_slots = max(0, min(bells_slots, max_slots))

    if max_slots < 5:
        if kv_quant == "f16":
            notes.append("Almost no VRAM for cache. Try --cache-type-k q8_0 --cache-type-v q8_0.")
        elif kv_quant == "q8_0":
            notes.append("Almost no VRAM for cache. Try q4_0 KV or shorter context.")
        else:
            notes.append("Almost no VRAM for cache. Try shorter context.")

    # --- Hit rate ---
    hr = cache_hit_rate(bells_slots, model)

    # --- Attention / base compute ---
    backend = _resolve_backend(hw)
    gpu_bw = hw.gpu.bw_gbs
    if backend == "Vulkan":
        gpu_bw *= VULKAN_BW_PENALTY
    if hw.gpu_count > 1:
        gpu_bw_effective = gpu_bw  # each GPU handles its layers in parallel-ish
    else:
        gpu_bw_effective = gpu_bw

    base_ms = REF_BASE_MS * (REF_GPU_BW / gpu_bw_effective)

    kv_bytes = kv_cache_gb(model, ctx, kv_quant) * (1024**3)
    if hw.gpu_count > 1:
        kv_per_gpu = kv_bytes / hw.gpu_count
    else:
        kv_per_gpu = kv_bytes
    kv_read_ms = (kv_per_gpu / (gpu_bw * BW_EFFICIENCY * 1e9)) * 1000

    attention_ms = max(base_ms, kv_read_ms)
    if hw.gpu_count > 1:
        attention_ms *= 1.08  # inter-GPU sync overhead

    # --- Expert loading ---
    # Each token activates n_active experts in EACH of n_moe_layers layers.
    # Transfer cost = misses * expert_size / bandwidth, reduced by pipeline overlap
    # (CPU→GPU transfer partially overlaps with GPU compute on cached experts).
    xfer_bw = _transfer_bw(model, quant, hw)
    expert_bytes = slot_gb * (1024**3)
    misses_per_token = model.n_active * model.n_layers * (1.0 - hr)
    expert_ms_bells = (misses_per_token * expert_bytes / (xfer_bw * 1e9)) * 1000 * PIPELINE_OVERLAP

    # Without BELLS: all active experts from CPU every layer every token
    all_loads = model.n_active * model.n_layers
    expert_ms_no_bells = (all_loads * expert_bytes / (xfer_bw * 1e9)) * 1000 * PIPELINE_OVERLAP

    # --- Totals ---
    overhead = GRAPH_SPLIT_MS
    total_bells = attention_ms + expert_ms_bells + overhead
    total_no_bells = attention_ms + expert_ms_no_bells + overhead

    tok_s = 1000.0 / total_bells if total_bells > 0 else 0
    tok_s_no = 1000.0 / total_no_bells if total_no_bells > 0 else 0

    # --- Flags ---
    expert_gb = size * model.expert_ratio
    model_fits = (expert_gb * 1024) < hw.ram_mb
    flags: list[str] = []

    if hw.gpu_count > 1:
        devs = ",".join(f"{backend}{i}" for i in range(hw.gpu_count))
        flags.append(f"--device {devs}")

    flags.append("-ngl 99")
    flags.append("-fa")

    if model_fits:
        flags.append("--cpu-moe-pinned")
    else:
        flags.append("--cpu-moe")

    if bells_slots > 0:
        flags.append(f"--bells-slots {bells_slots}")

    if kv_quant != "f16":
        flags.append(f"--cache-type-k {kv_quant}")
        flags.append(f"--cache-type-v {kv_quant}")

    flags.append(f"-c {ctx}")

    if not model_fits:
        flags.append("--load-mode mmap")

    # --- Notes ---
    if not model_fits:
        notes.append("Model exceeds RAM — streaming from storage.")
    if kv_quant == "f16" and ctx > 16000:
        notes.append("Consider --cache-type-k q8_0 --cache-type-v q8_0 to save VRAM.")
    if bells_slots == 0 and max_slots == 0:
        notes.append("No VRAM for BELLS cache — attention dominates at this context length.")
    elif hr < 0.5:
        notes.append(f"Low hit rate ({hr:.0%}). More VRAM or fewer slots needed.")
    speedup = tok_s / tok_s_no if tok_s_no > 0 else 0
    if speedup > 1.1:
        notes.append(f"BELLS speedup: {speedup:.1f}x over baseline.")

    return Estimate(
        tok_s=round(tok_s, 1),
        tok_s_no_bells=round(tok_s_no, 1),
        attention_ms=round(attention_ms, 1),
        expert_ms=round(expert_ms_bells, 1),
        overhead_ms=round(overhead, 1),
        bells_slots=bells_slots,
        max_bells_slots=max_slots,
        hit_rate=round(hr, 3),
        kv_cache_gb=round(kv, 1),
        free_vram_gb=round(free, 1),
        model_fits_ram=model_fits,
        flags=flags,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

def format_result(est: Estimate, model: ModelSpec, quant: str, hw: Hardware, ctx: int) -> str:
    backend = _resolve_backend(hw)
    lines = [
        f"  Model:       {model.name} {quant}",
        f"  GPU:         {hw.gpu_count}x {hw.gpu.name} ({hw.gpu.vram_mb} MB each)",
        f"  Backend:     {backend}",
        f"  RAM:         {hw.ram_mb // 1024} GB ({hw.ram_bw_gbs:.0f} GB/s)",
        f"  Context:     {ctx:,} tokens",
        "",
        f"  KV cache:    {est.kv_cache_gb:.1f} GB",
        f"  Free VRAM:   {est.free_vram_gb:.1f} GB (for BELLS cache)",
        f"  BELLS slots: {est.bells_slots} / {est.max_bells_slots} max",
        f"  Hit rate:    {est.hit_rate:.0%}",
        "",
        f"  Decode breakdown:",
        f"    Attention:   {est.attention_ms:.1f} ms",
        f"    Expert load: {est.expert_ms:.1f} ms",
        f"    Overhead:    {est.overhead_ms:.1f} ms",
        f"    Total:       {est.attention_ms + est.expert_ms + est.overhead_ms:.1f} ms",
        "",
        f"  Estimated tok/s:",
        f"    With BELLS:    {est.tok_s:.1f} tok/s",
        f"    Without BELLS: {est.tok_s_no_bells:.1f} tok/s",
    ]
    if est.tok_s_no_bells > 0:
        lines.append(f"    Speedup:       {est.tok_s / est.tok_s_no_bells:.1f}x")
    lines.append("")
    lines.append("  Recommended flags:")
    lines.append(f"    {' '.join(est.flags)}")
    if est.notes:
        lines.append("")
        for n in est.notes:
            lines.append(f"  * {n}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers for CLI
# ---------------------------------------------------------------------------

def find_gpu(query: str) -> GpuSpec | None:
    q = query.lower().strip()
    if q in GPUS:
        return GPUS[q]
    for k, v in GPUS.items():
        if q in k or q in v.name.lower():
            return v
    return None


def find_model(query: str) -> tuple[str, ModelSpec] | None:
    q = query.lower().strip()
    if q in MODELS:
        return q, MODELS[q]
    for k, v in MODELS.items():
        if q in k or q in v.name.lower():
            return k, v
    return None


def parse_ram_speed(spec: str) -> float | None:
    s = spec.lower().strip().replace(" ", "-")
    if s in RAM_SPEEDS:
        return RAM_SPEEDS[s]
    for k, v in RAM_SPEEDS.items():
        if s in k:
            return v
    try:
        return float(s)
    except ValueError:
        return None


def list_gpus() -> str:
    lines = []
    current_cat = ""
    for key, g in GPUS.items():
        if "laptop" in key:
            cat = "Laptop"
        elif "rx " in key or "arc " in key:
            cat = "AMD/Intel"
        elif any(x in key for x in ("a10", "a100", "l4", "l40")):
            cat = "Data Center"
        else:
            cat = "NVIDIA Desktop"
        if cat != current_cat:
            if lines:
                lines.append("")
            lines.append(f"  {cat}:")
            current_cat = cat
        lines.append(f"    {g.name:<28} {g.vram_mb:>6} MB   {g.bw_gbs:>6.0f} GB/s")
    return "\n".join(lines)


def list_models() -> str:
    lines = []
    for key, m in MODELS.items():
        quants = ", ".join(f"{q} ({s:.0f}GB)" for q, s in m.quants.items())
        lines.append(f"  {m.name:<24} {m.n_experts} experts, top-{m.n_active}")
        lines.append(f"    Quants: {quants}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="bells estimate",
        description="Estimate BELLS performance and recommend flags",
    )
    sub = parser.add_subparsers(dest="action")

    # List GPUs
    sub.add_parser("gpus", help="List known GPUs")
    sub.add_parser("models", help="List known models")

    # Estimate
    est_p = sub.add_parser("run", help="Run estimation")
    est_p.add_argument("-g", "--gpu", required=True, help="GPU name (e.g. 'rtx 3060')")
    est_p.add_argument("-n", "--gpu-count", type=int, default=1, help="Number of GPUs")
    est_p.add_argument("-m", "--model", required=True, help="Model name (e.g. 'qwen3.6-35b')")
    est_p.add_argument("-q", "--quant", required=True, help="Quantization (e.g. 'Q4_K_M')")
    est_p.add_argument("-c", "--context", type=int, default=4096, help="Context length")
    est_p.add_argument("-r", "--ram", type=int, default=32, help="RAM in GB")
    est_p.add_argument("--ram-speed", default="ddr4-3200", help="RAM speed (e.g. 'ddr5-6000')")
    est_p.add_argument("--kv-quant", default="f16", help="KV cache quant: f16, q8_0, q4_0")
    est_p.add_argument("--slots", type=int, default=None, help="Override BELLS slots")
    est_p.add_argument("--storage", default="nvme-gen4", help="Storage type for mmap")
    est_p.add_argument("--gpu-vram", type=int, default=None, help="Override GPU VRAM (MB)")
    est_p.add_argument("--gpu-bw", type=float, default=None, help="Override GPU bandwidth (GB/s)")
    est_p.add_argument("--backend", choices=["auto", "cuda", "rocm", "vulkan"],
                       default="auto", help="GPU backend (default: auto-detect from GPU name)")

    parsed = parser.parse_args(args)

    if parsed.action == "gpus":
        print("\nKnown GPUs:\n")
        print(list_gpus())
        print()
        return

    if parsed.action == "models":
        print("\nKnown models:\n")
        print(list_models())
        print()
        return

    if parsed.action != "run":
        parser.print_help()
        return

    # Resolve GPU
    gpu = find_gpu(parsed.gpu)
    if gpu is None:
        if parsed.gpu_vram and parsed.gpu_bw:
            gpu = GpuSpec(name=parsed.gpu, vram_mb=parsed.gpu_vram, bw_gbs=parsed.gpu_bw)
        else:
            print(f"Unknown GPU: {parsed.gpu}")
            print("Use --gpu-vram and --gpu-bw for custom GPUs, or 'bells estimate gpus' to list known ones.")
            return
    if parsed.gpu_vram:
        gpu = GpuSpec(name=gpu.name, vram_mb=parsed.gpu_vram, bw_gbs=gpu.bw_gbs)
    if parsed.gpu_bw:
        gpu = GpuSpec(name=gpu.name, vram_mb=gpu.vram_mb, bw_gbs=parsed.gpu_bw)

    # Resolve model
    result = find_model(parsed.model)
    if result is None:
        print(f"Unknown model: {parsed.model}")
        print("Use 'bells estimate models' to list known ones.")
        return
    model_key, model = result

    # RAM bandwidth
    ram_bw = parse_ram_speed(parsed.ram_speed)
    if ram_bw is None:
        print(f"Unknown RAM speed: {parsed.ram_speed}")
        return

    # Storage bandwidth
    storage_bw = STORAGE_SPEEDS.get(parsed.storage, 7.0)

    hw = Hardware(
        gpu=gpu,
        gpu_count=parsed.gpu_count,
        ram_mb=parsed.ram * 1024,
        ram_bw_gbs=ram_bw,
        storage_bw_gbs=storage_bw,
        backend=parsed.backend,
    )

    est = estimate(
        model_key=model_key,
        quant=parsed.quant,
        hw=hw,
        ctx=parsed.context,
        kv_quant=parsed.kv_quant,
        bells_slots=parsed.slots,
    )

    print(f"\n{'=' * 50}")
    print(f"  BELLS Performance Estimate")
    print(f"{'=' * 50}\n")
    print(format_result(est, model, parsed.quant, hw, parsed.context))
    print()


if __name__ == "__main__":
    main()
