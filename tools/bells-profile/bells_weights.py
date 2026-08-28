#!/usr/bin/env python3
"""Would skipping low-weight experts save enough transfers to be worth the error?

The premise. Copies are ~87% of BELLS's per-layer cost, eviction policy cannot reduce misses
(see bells_policy.py), and PCIe cannot be widened. What is left is to not copy at all on some
misses. A missing expert has to be transferred before its matmul can run - but an expert the
router weighted at 0.02 contributes almost nothing to that layer's output. Skip its copy, drop
it from the sum, renormalise the survivors, and the transfer disappears.

That is a quality-for-speed trade, so it needs two numbers before anyone writes the runtime:

  - how many copies it saves      (the win)
  - how much probability mass it discards  (the price)

There is a reason to expect the trade to be favourable: low-weight experts should be the ones
the cache is least likely to hold, because they are chosen less often. If so, the experts worth
skipping and the experts that cost a transfer are disproportionately the same experts.

Needs a .trace.bin and its .weights.bin sidecar, both written by llama-bells-profile.

    python bells_weights.py models/bells/wt4.trace.bin
"""

import argparse
import struct
import sys

import numpy as np

from bells_policy import load as load_trace


def load_weights(path):
    with open(path, "rb") as f:
        blob = f.read()

    if blob[:8] != b"BELLSWT1":
        sys.exit(f"{path}: not a BELLS weights file")

    version, n_layer, n_used = struct.unpack_from("<3I", blob, 8)
    (n_rec,) = struct.unpack_from("<Q", blob, 20)

    off = 28 + n_layer*4
    need = n_rec*n_layer*n_used
    have = (len(blob) - off)//4
    n_rec = min(n_rec, have//(n_layer*n_used))

    w = np.frombuffer(blob, dtype=np.float32, count=n_rec*n_layer*n_used, offset=off)
    return w.reshape(n_rec, n_layer, n_used), n_layer, n_used


def lru_miss_mask(seq, n_slot):
    """Replay LRU and return a bool array marking which (record, k) accesses missed."""
    n, k = seq.shape
    miss = np.zeros((n, k), dtype=bool)

    resident = {}
    tick = 0
    for i in range(n):
        row = seq[i]
        tick += 1
        want = set(row.tolist())
        for j in range(k):
            e = int(row[j])
            if e in resident:
                resident[e] = tick
                continue
            miss[i, j] = True
            if len(resident) >= n_slot:
                v, best = None, None
                for x, t in resident.items():
                    if x in want:
                        continue
                    if best is None or t < best:
                        best, v = t, x
                if v is None:
                    continue
                del resident[v]
            resident[e] = tick
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--weights", default=None, help="defaults to <trace base>.weights.bin")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--mult", type=int, default=2, help="cache size as a multiple of n_used")
    a = ap.parse_args()

    wpath = a.weights or a.trace.replace(".trace.bin", ".weights.bin")

    t = load_trace(a.trace, a.limit)
    w_all, n_layer_w, n_used_w = load_weights(wpath)

    if n_used_w != t["n_used"]:
        sys.exit(f"n_expert_used mismatch: trace {t['n_used']}, weights {n_used_w}")

    n = min(t["n_rec"], w_all.shape[0])
    rows = t["rows"][:n]
    w_all = w_all[:n]

    n_slot = min(a.mult*t["n_used"], t["n_expert"])
    step = max(1, t["n_layer"]//a.layers)
    layers = list(range(0, min(t["n_layer"], n_layer_w), step))[:a.layers]

    print(f"\n{a.trace}")
    print(f"{n} records, {t['n_expert']} experts, {t['n_used']} active, "
          f"cache {n_slot} slots ({a.mult}x)")
    print(f"layers {layers}\n")

    # --- what the router's weights actually look like, by rank
    wl = w_all[:, layers, :].reshape(-1, t["n_used"])
    print("mean router weight by rank (rank 0 = router's first choice):")
    means = wl.mean(axis=0)
    print("  " + "  ".join(f"r{i}:{m:.3f}" for i, m in enumerate(means)))
    print(f"  weights sum to {wl.sum(axis=1).mean():.3f} on average\n")

    # --- are low-weight experts also the ones that miss?
    miss_w, hit_w = [], []
    for il in layers:
        m = lru_miss_mask(rows[:, il, :], n_slot)
        wv = w_all[:, il, :]
        miss_w.append(wv[m])
        hit_w.append(wv[~m])

    miss_w = np.concatenate(miss_w)
    hit_w = np.concatenate(hit_w)

    print(f"mean weight of a HIT  {hit_w.mean():.4f}   ({len(hit_w)} accesses)")
    print(f"mean weight of a MISS {miss_w.mean():.4f}   ({len(miss_w)} accesses)")
    ratio = miss_w.mean()/max(1e-9, hit_w.mean())
    print(f"  -> misses carry {ratio:.2f}x the weight of hits"
          f"{'  (GOOD: cheap to skip)' if ratio < 0.9 else ''}"
          f"{'  (BAD: the misses are the important experts)' if ratio > 1.1 else ''}\n")

    # --- the actual trade
    total_mass = wl.sum()
    n_access = miss_w.size + hit_w.size

    print(f"{'threshold':>10} {'copies saved':>14} {'of all copies':>15} {'mass dropped':>14}")
    print("-"*56)
    for thr in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        skip = miss_w < thr
        saved = int(skip.sum())
        mass = float(miss_w[skip].sum())
        print(f"{thr:>10.2f} {saved:>14} {100.0*saved/max(1,miss_w.size):>14.1f}% "
              f"{100.0*mass/max(1e-9,total_mass):>13.2f}%")

    print(f"\ntotal accesses {n_access}, misses {miss_w.size} "
          f"({100.0*miss_w.size/max(1,n_access):.1f}%)")
    print("\nReading this: 'copies saved' is the fraction of transfers that disappear, which is")
    print("the speed. 'mass dropped' is the share of total router probability thrown away,")
    print("which is the error - and it compounds over every layer, so small is essential.")


if __name__ == "__main__":
    main()
