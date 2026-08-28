"""Will BELLS help my model? A calculator.

    python bells_calc.py --vram 6 --ram 32 --preset qwen3-next-80b
    python bells_calc.py --vram 12 --ram 64 --layers 48 --experts 512 --active 10 \
                         --hidden 2048 --ffn 512 --quant q2_k --model-gb 27

Works out the per-token expert working set, how many of them your VRAM can cache, and
whether that lands in the range where BELLS wins. Thresholds are calibrated against measured
outcomes, not guessed - see the table at the bottom of the output.
"""

import argparse
import sys

# bytes per parameter, including quantisation overhead
QUANT_BYTES = {
    "f16":   2.0,
    "q8_0":  1.06,
    "q6_k":  0.82,
    "q5_k":  0.69,
    # checked against expert sizes the runtime reports: q4_k_m gives 2.92 MB for
    # 3x2048x768, q2_k gives 1.09 MB for 3x2048x512, mxfp4 gives 12.61 MB for 3x2880x2880
    "q4_k":  0.62,
    "q4_k_m": 0.62,
    "mxfp4": 0.51,
    "q3_k":  0.44,
    "q2_k":  0.35,
    "iq2":   0.28,
    "iq1":   0.22,
}

# layers, experts, active, hidden, moe_ffn, quant, model GB at that quant
#
# Architecture figures are taken from each model's config.json. Model sizes are approximate.
PRESETS = {
    "qwen3-next-80b":    (48, 512, 10, 2048,   512, "q2_k",   27),
    "qwen3-30b-a3b":     (48, 128,  8, 2048,   768, "q4_k",   18),
    "qwen3-235b-a22b":   (94, 128,  8, 4096,  1536, "q2_k",   70),
    "gpt-oss-120b":      (36, 128,  4, 2880,  2880, "mxfp4",  59),
    "gpt-oss-20b":       (24,  32,  4, 2880,  2880, "mxfp4",  12),
    "olmoe-1b-7b":       (16,  64,  8, 2048,  1024, "q4_k",    4),
    "deepseek-v4-flash": (43, 256,  6, 4096,  2048, "q2_k",  103),
    "deepseek-v3":       (61, 256,  8, 7168,  2048, "q2_k",  200),
    "kimi-k2":           (61, 384,  8, 7168,  2048, "q2_k",  380),
    "mixtral-8x7b":      (32,   8,  2, 4096, 14336, "q4_k",   26),
    "mixtral-8x22b":     (56,   8,  2, 6144, 16384, "q4_k",   80),
}


def analyse(layers, experts, active, hidden, ffn, quant, vram_gb, ram_gb, model_gb):
    bpp = QUANT_BYTES[quant]

    # gate + up + down, each hidden x ffn
    expert_bytes = 3 * hidden * ffn * bpp
    working_set  = layers * active * expert_bytes
    all_experts  = layers * experts * expert_bytes

    if model_gb is None:
        model_gb = all_experts/1e9 * 1.1   # experts plus roughly 10% for attention etc

    # Non-expert weights land in VRAM under --cpu-moe, and context and compute buffers need
    # room too. The auto-sizer keeps a third of what is left as headroom, because over
    # allocating the cache measurably slows decode.
    # The +2.0 is calibrated, not guessed: on a 6 GB card running Qwen3-Next the runtime saw
    # 3.0-3.3 GiB free at cache-allocation time, implying ~2.7-3.0 GB already gone. Using
    # +1.0 predicted 56 slots where the runtime actually chose 43.
    non_expert_gb = max(0.5, model_gb - all_experts/1e9)
    overhead_gb   = non_expert_gb + 2.0
    free_gb       = max(0.0, vram_gb - overhead_gb)
    usable_gb     = free_gb * 2/3

    slots = int(usable_gb*1e9 / expert_bytes / layers) if expert_bytes else 0
    ratio = slots/active if active else 0

    # Active expert parameters per token. This is the CPU work BELLS offloads, and it turns
    # out to matter more than the cache ratio: Mixtral-8x7B measured 3.85x faster at a 1.0x
    # ratio, because ~13B active parameters make the CPU baseline enormous. Qwen3-Next gains
    # only 1.18x at 4.7x, because 3B active is barely any work to take away.
    active_params = layers * active * 3 * hidden * ffn

    return {
        "expert_mb":    expert_bytes/1e6,
        "working_gb":   working_set/1e9,
        "all_gb":       all_experts/1e9,
        "model_gb":     model_gb,
        "sparsity":     experts/active if active else 0,
        "overhead_gb":  overhead_gb,
        "usable_gb":    usable_gb,
        "slots":        slots,
        "ratio":        ratio,
        "active_b":     active_params/1e9,
        "fits_ram":     model_gb <= ram_gb * 0.9,
        # if the whole model fits on the card there is nothing to stream and BELLS is moot
        "fits_vram":    model_gb <= vram_gb * 0.85,
    }


def verdict(r):
    """Whether a cache is POSSIBLE. Not whether it will help.

    This function used to grade the cache ratio - "workable" at 2x, "good fit" at 4x - and it
    was wrong. GPT-OSS-120B on a 24 GB card with 128 GB of RAM rates 7.8x here and measures
    6.1x SLOWER than --cpu-moe, the worst result in the project. It also had a special case for
    models with many active expert parameters, on the strength of Mixtral measuring 5.88x; that
    number was an artifact of an 8-core cloud baseline and is 1.06x on a normal CPU.

    Four separate rules for predicting the speedup from static properties were tried and all
    four were falsified. The one that survives cannot be computed from a config.json, because it
    depends on your CPU:

        BELLS converges on a throughput set by the GPU and the model. Measure your --cpu-moe
        baseline. Below that number BELLS wins by the difference; above it, BELLS loses.

    So this only rules things out. A low ratio reliably means no; a high ratio means "measure".
    """
    if r["fits_vram"]:
        return ("N/A", "the whole model fits in VRAM - just load it, BELLS has nothing to do")
    if not r["fits_ram"]:
        return ("NO", "model does not fit in RAM - cold experts come off disk, and BELLS "
                      "pays that read plus a PCIe copy. Strictly worse than --cpu-moe.")
    if r["slots"] < 1:
        return ("NO", "no VRAM left for a cache after attention and buffers")

    if r["ratio"] < 1.0:
        return ("NO", "cache cannot hold even one token's experts")
    if r["ratio"] < 1.5:
        return ("NO", "cache below 1.5x the working set - it will thrash. Measured 0.83x "
                      "against --cpu-moe at 1.2x, and the runtime now refuses to enable here.")
    return ("MEASURE", "a cache is possible. Whether it helps depends on your CPU, which this "
                       "cannot know - benchmark --cpu-moe against BELLS and compare")


def matrix():
    """Compatibility table across every preset and a range of hardware."""
    configs = [(4, 16), (6, 32), (8, 32), (12, 32), (12, 64),
               (16, 64), (24, 64), (24, 128), (48, 256), (80, 512)]

    print("\nBELLS compatibility. The cell is the cache ratio: whether a cache FITS.")
    print("It does not predict a speedup - 7.8x here measured 6.1x slower in practice.\n")
    print("  N.Nx = a cache is possible, measure it   .  = cache too small to help")
    print("  vram = model fits on the card, load it normally   RAM = model exceeds RAM\n")

    head = "  " + f"{'model':<20}" + "".join(f"{v}/{r}".rjust(9) for v, r in configs)
    print(head)
    print("  " + "-"*(len(head) - 2))

    for name in PRESETS:
        layers, experts, active, hidden, ffn, quant, model_gb = PRESETS[name]
        row = f"  {name:<20}"
        for vram, ram in configs:
            r = analyse(layers, experts, active, hidden, ffn, quant, vram, ram, model_gb)
            ans, _ = verdict(r)
            if ans == "N/A":
                cell = "vram"
            elif not r["fits_ram"]:
                cell = "RAM"
            elif ans == "MEASURE":
                cell = f"{r['ratio']:.1f}x"
            else:
                cell = "."
            row += cell.rjust(9)
        print(row)

    print("\n  columns are VRAM/RAM in GB\n")
    print("  Measured calibration:")
    print("    RTX 2060 6 GB, paired runs")
    print("      gpt-oss-120b   1.0x ratio, 1.4B active  ->  1.59x SLOWER")
    print("      qwen3-30b-a3b  2.2x ratio, 1.8B active  ->  1.52x faster")
    print("      qwen3-next-80b 4.7x ratio, 1.5B active  ->  1.18x faster")
    print("    A10G 24 GB with only 8 vCPU, which inflates these")
    print("      mixtral-8x7b   1.0x ratio, 5.1B active  ->  3.85x faster")
    print("      mixtral-8x7b   2.0x ratio, 5.1B active  ->  5.88x faster")
    print("  Ratio screens candidates, it does not rank them. Active parameters drive the")
    print("  size of the win, and a slow CPU makes every result look better.")
    print("  BELLS accelerates generation only, never prompt processing.\n")


def main():
    ap = argparse.ArgumentParser(description="Will BELLS help my model?")
    ap.add_argument("--matrix", action="store_true",
                    help="print the full model x hardware compatibility table")
    ap.add_argument("--vram", type=float, help="GPU memory, GB")
    ap.add_argument("--ram",  type=float, help="system memory, GB")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="known model")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--experts", type=int, help="total routed experts")
    ap.add_argument("--active", type=int, help="experts used per token")
    ap.add_argument("--hidden", type=int, help="hidden_size")
    ap.add_argument("--ffn", type=int, help="moe_intermediate_size")
    ap.add_argument("--quant", choices=sorted(QUANT_BYTES), default="q4_k")
    ap.add_argument("--model-gb", type=float, help="model file size, GB")
    a = ap.parse_args()

    if a.matrix:
        matrix()
        return

    if a.vram is None or a.ram is None:
        ap.error("need --vram and --ram (or --matrix)")

    if a.preset:
        layers, experts, active, hidden, ffn, quant, model_gb = PRESETS[a.preset]
        name = a.preset
        if a.quant != "q4_k":
            quant = a.quant
            model_gb = None
    else:
        missing = [f for f in ("layers", "experts", "active", "hidden", "ffn")
                   if getattr(a, f) is None]
        if missing:
            ap.error("need --preset, or all of: --" + ", --".join(missing))
        layers, experts, active = a.layers, a.experts, a.active
        hidden, ffn, quant = a.hidden, a.ffn, a.quant
        model_gb = a.model_gb
        name = "custom"

    r = analyse(layers, experts, active, hidden, ffn, quant, a.vram, a.ram, model_gb)
    ans, why = verdict(r)

    print(f"\n{name}  ({quant})   on {a.vram:g} GB VRAM / {a.ram:g} GB RAM\n")
    print(f"  expert size          {r['expert_mb']:8.2f} MB")
    print(f"  sparsity             {r['sparsity']:8.1f}:1   ({experts} experts, {active} active)")
    print(f"  per-token working set{r['working_gb']:8.2f} GB   ({layers} layers x {active} x expert)")
    print(f"  active expert params {r['active_b']:8.1f} B    (the CPU work BELLS offloads)")
    print(f"  all experts          {r['all_gb']:8.1f} GB")
    print(f"  model                {r['model_gb']:8.1f} GB   {'fits RAM' if r['fits_ram'] else 'EXCEEDS RAM'}")
    print()
    print(f"  VRAM overhead        {r['overhead_gb']:8.1f} GB   (attention + context + buffers)")
    print(f"  usable for cache     {r['usable_gb']:8.1f} GB")
    print(f"  cache slots/layer    {r['slots']:8d}")
    print(f"  cache ratio          {r['ratio']:8.1f}x")
    print()
    print(f"  --> {ans}: {why}\n")

    if not r["fits_ram"]:
        need = r["model_gb"]/0.9
        print(f"  Need about {need:.0f} GB of RAM for this model.\n")
    elif r["ratio"] < 2.0 and r["slots"] >= 1:
        # how much VRAM would reach 2x
        want_gb = (2.0*active*r['expert_mb']*1e6*layers)/1e9 * 1.5 + r['overhead_gb']
        print(f"  About {want_gb:.0f} GB of VRAM would reach a 2x ratio.\n")

    print("  Measured (A10G 24 GB, 16 vCPU, 128 GB RAM, paired within one run):")
    print("    Qwen3-Next-80B    ratio  27x  ->  2.08x faster")
    print("    Qwen3-235B-A22B   ratio 1.9x  ->  1.81x faster")
    print("    Mixtral-8x7B      ratio 2.0x  ->  1.06x")
    print("    GPT-OSS-120B      ratio 7.8x  ->  6.1x SLOWER")
    print()
    print("  The ratio does not order those. What does, and what this tool cannot compute:")
    print("    BELLS converges on a throughput set by your GPU and model, not your CPU.")
    print("    Measure --cpu-moe. Below that number BELLS wins; above it, BELLS loses.")
    print("    Same Mixtral, same cache: 1.63x at 8 cores, 0.72x at 32.")
    print()
    print("  BELLS speeds up generation only; long prompts are unaffected.")
    print("  The cache also competes with the KV cache for VRAM - a shorter -c leaves more.\n")


if __name__ == "__main__":
    main()
