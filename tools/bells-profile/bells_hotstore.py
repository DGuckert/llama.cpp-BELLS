#!/usr/bin/env python3
"""Does reserving slots for globally hot experts beat plain LRU?

BELLS uses pure LRU. The predictor work in this repo asked a different question - "which experts
does THIS TOKEN need" - and died on it: precision fell from 82.2% in-distribution to 49.5% on
unfamiliar text, because token-id routing statistics are corpus-specific.

Frequency pinning asks something else. "Which experts does this MODEL use most" is a property of
the trained weights rather than of the text, so it should survive a domain change that kills
token-id prediction. Belady says there are 16.4 points over LRU at a 2x cache ratio, and LRU
leaves all of it.

The policy: reserve `hot` of the `cap` slots for the most-frequently-routed experts, counted on a
training prefix, and never evict them. The remaining slots run LRU as now. hot=0 is exactly
today's behaviour, which is the control.

    python bells_hotstore.py models/bells/big.trace.bin --caps 16,32,64 --layers 8

Counting the hot set on a prefix and evaluating on the rest is what makes this a fair test - a
hot set counted over the whole trace would be cheating in the same way the 82.2% figure was.
"""

import argparse
from collections import defaultdict

import numpy as np

from bells_policy import load


def simulate(rows, il, cap, hot_n, hot_set):
    """Replay one layer. `hot_set` slots are pinned; the rest are LRU."""
    pinned = set(hot_set[:hot_n])

    resident = dict(pinned and {})   # expert -> last use
    for e in pinned:
        resident[e] = -1             # -1 marks never-evictable

    lru_cap = cap - len(pinned)
    tick = 0
    hits = misses = 0

    for i in range(rows.shape[0]):
        tick += 1
        for e in rows[i, il, :]:
            e = int(e)
            if e in resident:
                hits += 1
                if resident[e] != -1:
                    resident[e] = tick
                continue

            misses += 1

            # evict the oldest non-pinned entry
            if len(resident) - len(pinned) >= lru_cap:
                victim, oldest = None, None
                for k, v in resident.items():
                    if v == -1:
                        continue
                    if oldest is None or v < oldest:
                        oldest, victim = v, k
                if victim is None:
                    continue          # cache is all pinned, nothing to give
                del resident[victim]

            resident[e] = tick

    return hits/max(1, hits + misses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole trace")
    ap.add_argument("--layers", type=int, default=8, help="how many layers to sample")
    ap.add_argument("--caps", default="16,32,64")
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="prefix used to count the hot set; the rest is evaluated")
    a = ap.parse_args()

    t = load(a.trace, a.limit or None)
    rows, n_layer = t["rows"], t["n_layer"]

    step = max(1, n_layer//a.layers)
    layers = list(range(0, n_layer, step))[:a.layers]

    n_train = int(rows.shape[0]*a.train_frac)
    train, test = rows[:n_train], rows[n_train:]

    print(f"\n{a.trace}")
    print(f"{t['n_rec']} records, {t['n_expert']} experts, {t['n_used']} active")
    print(f"hot set counted on the first {n_train}, evaluated on the remaining "
          f"{rows.shape[0] - n_train}\n")

    caps = [int(x) for x in a.caps.split(",") if x.strip()]

    hdr = f"{'cap':>5} {'hot':>5} {'hit':>8} {'vs LRU':>9}"
    print(hdr)
    print("-"*len(hdr))

    for cap in caps:
        base = None
        for hot_frac in (0.0, 0.125, 0.25, 0.5):
            hot_n = int(cap*hot_frac)

            agg = 0.0
            for il in layers:
                counts = np.bincount(train[:, il, :].ravel().astype(np.int64),
                                     minlength=t["n_expert"])
                hot_set = list(np.argsort(-counts))
                agg += simulate(test, il, cap, hot_n, hot_set)

            hit = agg/len(layers)
            if base is None:
                base = hit
            delta = 100.0*(hit - base)

            print(f"{cap:>5} {hot_n:>5} {hit:>7.1%} {delta:>+8.1f}pt"
                  f"{'   <- LRU, control' if hot_n == 0 else ''}")
        print()

    print("A hot store only earns its place if it beats its own cap's LRU row. Reserving slots")
    print("costs LRU capacity, so a wash means it is strictly worse - same hit rate, more code.")


if __name__ == "__main__":
    main()
