#!/usr/bin/env python3
"""Should every layer get the same number of cache slots?

BELLS gives each MoE layer an identical n_slot. That is only the right answer if an extra slot
is worth the same in every layer, which nobody has checked. If routing is concentrated in some
layers and near-uniform in others, slots moved from the flat layers to the skewed ones buy hits
for free - the same VRAM, a better hit rate, and hit rate is the only lever left after eviction
policy and transfer tricks both failed.

Method: measure each layer's hit-rate curve against slot count, then compare

    uniform    every layer gets the same S
    greedy     slots handed out one at a time to whichever layer gains most

Greedy is optimal here because each layer's curve is concave in practice (diminishing returns),
which is the condition that makes marginal allocation exact.

    python bells_alloc.py models/bells/qwen3-big.trace.bin
"""

import argparse

import numpy as np

from bells_policy import load, sim_lru


def curve(seq, n_expert, slots):
    """hits at each slot count, for one layer"""
    out = {}
    for s in slots:
        h, m = sim_lru(seq, s, n_expert)
        out[s] = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--mult", type=int, default=2,
                    help="uniform budget per layer, as a multiple of n_used")
    a = ap.parse_args()

    t = load(a.trace, a.limit)
    n_used, n_expert = t["n_used"], t["n_expert"]

    step = max(1, t["n_layer"]//a.layers)
    layers = list(range(0, t["n_layer"], step))[:a.layers]

    uniform = min(a.mult*n_used, n_expert)
    budget = uniform*len(layers)

    # grid from the minimum legal cache up to twice the uniform size
    grid = sorted({n_used, *range(n_used, min(2*uniform, n_expert) + 1, max(1, n_used//2))})

    print(f"\n{a.trace}")
    print(f"{t['n_rec']} records, {n_expert} experts, {n_used} active")
    print(f"layers {layers}")
    print(f"budget {budget} slots total ({uniform} per layer uniform)\n")

    curves = {}
    for il in layers:
        curves[il] = curve(t["rows"][:, il, :], n_expert, grid)

    per_layer_accesses = t["n_rec"]*n_used

    print(f"{'layer':>6} " + "".join(f"{s:>7}" for s in grid))
    print("       " + "  hit% at each slot count")
    print("-"*(7 + 7*len(grid)))
    for il in layers:
        row = f"{il:>6} "
        for s in grid:
            row += f"{100.0*curves[il][s]/per_layer_accesses:>6.1f} "
        print(row)

    # spread at the uniform point: if every layer is the same, there is nothing to reallocate
    at_u = [100.0*curves[il][uniform]/per_layer_accesses for il in layers]
    print(f"\nat {uniform} slots: best layer {max(at_u):.1f}%, worst {min(at_u):.1f}%, "
          f"spread {max(at_u) - min(at_u):.1f} points")

    # ---- greedy reallocation under the same total budget
    alloc = {il: n_used for il in layers}          # every layer needs at least n_used
    spent = n_used*len(layers)

    def hits_at(il, s):
        s = min(s, max(grid))
        if s in curves[il]:
            return curves[il][s]
        lo = max(g for g in grid if g <= s)
        return curves[il][lo]

    while spent < budget:
        best, gain = None, 0.0
        for il in layers:
            nxt = [g for g in grid if g > alloc[il]]
            if not nxt:
                continue
            s2 = nxt[0]
            if spent + (s2 - alloc[il]) > budget:
                continue
            g = (hits_at(il, s2) - hits_at(il, alloc[il]))/(s2 - alloc[il])
            if g > gain:
                best, gain = (il, s2), g
        if best is None:
            break
        il, s2 = best
        spent += s2 - alloc[il]
        alloc[il] = s2

    tot_u = sum(curves[il][uniform] for il in layers)
    tot_g = sum(hits_at(il, alloc[il]) for il in layers)
    tot_access = per_layer_accesses*len(layers)

    print(f"\nuniform : {100.0*tot_u/tot_access:.2f}% hit  ({uniform} slots x {len(layers)})")
    print(f"greedy  : {100.0*tot_g/tot_access:.2f}% hit  ({spent} slots) -> "
          + ", ".join(f"L{il}:{alloc[il]}" for il in layers))

    d = 100.0*(tot_g - tot_u)/tot_access
    if d <= 0.05:
        print(f"\nno gain ({d:+.2f} points). Uniform allocation is already right - the layers")
        print("do not differ enough for reallocation to buy anything.")
    else:
        cut = 100.0*(tot_g - tot_u)/max(1e-9, tot_access - tot_u)
        print(f"\ngain {d:+.2f} points = {cut:.0f}% fewer misses, at identical VRAM.")


if __name__ == "__main__":
    main()
