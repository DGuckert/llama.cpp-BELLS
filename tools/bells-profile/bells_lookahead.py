#!/usr/bin/env python3
"""How much of the Belady gap does a K-token lookahead capture?

The one lever that survives the single-stream problem. Prefetching cannot reduce bytes - it
moves the same experts through the same bottleneck, just earlier - but *eviction* can. Belady's
optimum sits 12 points above LRU at the ratios that matter, worth ~39% fewer misses at 4x, and a
miss avoided is a transfer that never happens. That helps whether or not copies can overlap
compute.

The catch is that Belady needs the future. To evict well at token t you need to know what tokens
t+1, t+2 ... will ask for, and a token-id predictor only describes the *current* token. What
does provide future token ids is speculative decoding: a draft model proposes the next few
tokens, and those proposals are exactly the lookahead this needs.

So the question is not "is Belady better" - it is - but **how far ahead you have to see**. A
draft model gives 2-4 tokens. If most of the gap closes by K=4 this is buildable on top of
speculative decoding. If it needs K=50 it is unreachable and the gap stays theoretical.

    python bells_lookahead.py models/bells/qwen3-big.trace.bin --mult 4
"""

import argparse

from bells_policy import load, sim_lru, sim_belady


def sim_lookahead(seq, n_slot, n_expert, k):
    """Belady truncated to a window of k future tokens.

    Evict the resident expert whose next use is furthest away *within the window*; anything not
    appearing in the window at all is fair game and goes first. Ties fall back to LRU, which is
    what a real implementation would do once the window runs out of information.
    """
    n = len(seq)
    rows = [set(map(int, r)) for r in seq]

    resident = {}          # expert -> last used tick
    hits = misses = 0
    tick = 0

    for i in range(n):
        tick += 1
        want = rows[i]

        for e in want:
            if e in resident:
                resident[e] = tick
                hits += 1

        for e in want:
            if e in resident:
                continue
            misses += 1

            if len(resident) >= n_slot:
                # distance to next use inside the window, n+1 if it never appears
                best_d, best_e = -1, None
                for x in resident:
                    if x in want:
                        continue
                    d = n + 1
                    for j in range(i + 1, min(i + 1 + k, n)):
                        if x in rows[j]:
                            d = j
                            break
                    # further away is a better victim; break ties by LRU
                    if d > best_d or (d == best_d and resident[x] < resident[best_e]):
                        best_d, best_e = d, x

                if best_e is None:
                    continue
                del resident[best_e]

            resident[e] = tick

    return hits, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--mult", type=int, default=4)
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    a = ap.parse_args()

    t = load(a.trace, a.limit)
    n_slot = min(a.mult*t["n_used"], t["n_expert"])
    step = max(1, t["n_layer"]//a.layers)
    layers = list(range(0, t["n_layer"], step))[:a.layers]

    print(f"\n{a.trace}")
    print(f"{t['n_rec']} records, {t['n_expert']} experts, {t['n_used']} active, "
          f"cache {n_slot} ({a.mult}x)")
    print(f"layers {layers}\n")

    acc = t["n_rec"]*t["n_used"]*len(layers)

    def total(fn, *extra):
        h = m = 0
        for il in layers:
            a_, b_ = fn(t["rows"][:, il, :], n_slot, t["n_expert"], *extra)
            h += a_
            m += b_
        return 100.0*h/max(1, h + m)

    lru = total(sim_lru)
    opt = total(sim_belady)

    print(f"{'lookahead':>10} {'hit%':>8} {'vs LRU':>9} {'of the gap':>12}")
    print("-"*42)
    print(f"{'LRU (0)':>10} {lru:>8.1f} {'-':>9} {'-':>12}")

    for k in [int(x) for x in a.ks.split(",") if x.strip()]:
        r = total(sim_lookahead, k)
        gap = 100.0*(r - lru)/max(1e-9, opt - lru)
        print(f"{k:>10} {r:>8.1f} {r - lru:>+8.1f}p {gap:>11.0f}%")

    print(f"{'Belady':>10} {opt:>8.1f} {opt - lru:>+8.1f}p {100:>11.0f}%")

    print("\nA draft model realistically supplies 2-4 tokens of lookahead. Read that row: it is")
    print("the fraction of the theoretical gain that speculative decoding could actually deliver.")


if __name__ == "__main__":
    main()
