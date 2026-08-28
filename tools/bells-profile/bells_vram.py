#!/usr/bin/env python3
"""Can better eviction buy the same hit rate from a SMALLER cache?

The claim being tested, which is not the same as "lookahead improves eviction": if you knew
which experts were coming you would not need to hold as many *just in case*, so a smaller cache
should suffice. That trades VRAM rather than misses, and VRAM is the binding constraint on every
card where BELLS currently fails - a 6 GB board cannot fund a large cache and a real context at
the same time.

So the question here is not "how much better is lookahead at a fixed size" but "how many slots
can be given back while holding hit rate constant".

    python bells_vram.py models/bells/qwen3-big.trace.bin
"""

import argparse

from bells_policy import load, sim_lru
from bells_lookahead import sim_lookahead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--target-mult", type=int, default=4,
                    help="the LRU configuration to match")
    a = ap.parse_args()

    t = load(a.trace, a.limit)
    nu, ne = t["n_used"], t["n_expert"]
    step = max(1, t["n_layer"]//a.layers)
    layers = list(range(0, t["n_layer"], step))[:a.layers]

    def tot(fn, slots, *extra):
        h = m = 0
        for il in layers:
            x, y = fn(t["rows"][:, il, :], slots, ne, *extra)
            h += x
            m += y
        return 100.0*h/max(1, h + m)

    mults = [1, 2, 3, 4, 6, 8]

    print(f"\n{a.trace}")
    print(f"{t['n_rec']} records, {ne} experts, {nu} active, layers {layers}\n")

    hdr = f"{'slots':>7} {'ratio':>6} {'LRU':>9} {'look-2':>9} {'look-4':>9} {'look-8':>9}"
    print(hdr)
    print("-"*len(hdr))

    rows = {}
    for mult in mults:
        s = min(mult*nu, ne)
        r = (tot(sim_lru, s),
             tot(sim_lookahead, s, 2),
             tot(sim_lookahead, s, 4),
             tot(sim_lookahead, s, 8))
        rows[mult] = r
        print(f"{s:>7} {mult:>5}x {r[0]:>8.1f}% {r[1]:>8.1f}% {r[2]:>8.1f}% {r[3]:>8.1f}%")

    target = rows[a.target_mult][0]
    tslots = min(a.target_mult*nu, ne)
    print(f"\nLRU at {a.target_mult}x ({tslots} slots) achieves {target:.1f}% hit.")
    print("Smallest cache each lookahead policy needs to match it:\n")

    for name, idx in (("look-2", 1), ("look-4", 2), ("look-8", 3)):
        for mult in mults:
            if rows[mult][idx] >= target:
                s = min(mult*nu, ne)
                saving = f"{100*(1 - s/tslots):.0f}% less VRAM" if s < tslots else "no saving"
                print(f"  {name:>7}: {mult}x ({s} slots) -> {saving}")
                break
        else:
            print(f"  {name:>7}: never reaches it within {mults[-1]}x")

    print("\nVRAM is what decides whether BELLS can be used at all on a small card, so slots")
    print("given back here matter more than hit rate gained at a fixed size.")


if __name__ == "__main__":
    main()
