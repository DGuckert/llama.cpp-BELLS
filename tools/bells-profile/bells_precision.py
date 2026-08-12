#!/usr/bin/env python3
"""Does confidence-gated prefetching pay once the bus has spare capacity?

The predictor this project removed was judged on RECALL - "does my top-N contain the experts
this token needs?" It scored 68-77% and still lost every wall-clock benchmark, and the
conclusion drawn was that prediction does not work.

That conclusion is too broad, for two reasons.

**Recall is the wrong metric.** What decides whether prefetching pays is PRECISION: of the
experts fetched, what fraction get used. Recall rewards fetching more; precision rewards
fetching less. Optimising recall is precisely how you end up moving a superset, which is the
mechanism that killed it.

**The regime was wrong too.** At the ~60% hit rates measured then, the bus was saturated with
demand traffic, so every speculative byte displaced a needed one. At a high hit rate that is no
longer true. Qwen3-Next at 192 slots:

    48 layers x 10 active x 9% miss x 1.09 MB = 47 MB/token
    47 MB / 7.09 GB/s = 6.6 ms, against 24.2 ms/token of decode
    -> the bus is idle ~73% of the time

Spare bandwidth means a wrong prefetch costs idle capacity rather than displacing a needed
transfer, while a right one removes a stall from the critical path. Small experts make each
wrong guess cheaper still.

So this measures, per confidence threshold:
  - precision  : of what we would prefetch, how much is used
  - coverage   : of the misses, how many we would have converted to hits
  - bytes      : whether the speculative traffic fits inside the idle capacity

    python bells_precision.py models/bells/qwen3-big.trace.bin
"""

import argparse
from collections import defaultdict

import numpy as np

from bells_policy import load, _victim


def build_table(rows, tokens, layers, n_expert, train_frac=0.6):
    """token id -> per-layer expert frequencies, counted on a training prefix."""
    n_train = int(len(tokens)*train_frac)
    table = {il: defaultdict(lambda: np.zeros(n_expert, dtype=np.int32)) for il in layers}
    seen = defaultdict(int)

    for i in range(n_train):
        tok = int(tokens[i])
        seen[tok] += 1
        for il in layers:
            for e in rows[i, il, :]:
                table[il][tok][int(e)] += 1

    return table, seen, n_train


def simulate(rows, tokens, il, table, seen, n_slot, n_expert, start, thresh, expert_mb):
    """Replay LRU, and at each token also consider prefetching predicted experts.

    A prefetch is only issued when the predictor's confidence for that expert clears `thresh`,
    which is the knob that trades recall for precision. Prefetched experts are admitted like
    any other, so they can evict - the cost of a bad guess is real, not hypothetical.
    """
    resident = {}
    tick = 0

    demand_miss = 0          # misses that would happen without prefetching
    pf_issued = 0            # experts speculatively fetched
    pf_used = 0              # ...that the token actually wanted
    saved = 0                # demand misses converted to hits by a prefetch

    for i in range(start, len(tokens)):
        tick += 1
        tok = int(tokens[i])
        want = list(map(int, rows[i, il, :]))
        wset = set(want)

        # ---- speculative admission, before the token is served
        counts = table[il].get(tok)
        if counts is not None and seen.get(tok, 0) > 0:
            conf = counts/float(seen[tok])
            cand = np.nonzero(conf >= thresh)[0]
            for e in cand:
                e = int(e)
                if e in resident:
                    continue
                pf_issued += 1
                if e in wset:
                    pf_used += 1
                    saved += 1
                if len(resident) >= n_slot:
                    v = _victim(resident, wset, lambda x: resident[x])
                    if v is None:
                        continue
                    del resident[v]
                resident[e] = tick

        # ---- demand path
        for e in want:
            if e in resident:
                resident[e] = tick
                continue
            demand_miss += 1
            if len(resident) >= n_slot:
                v = _victim(resident, wset, lambda x: resident[x])
                if v is None:
                    continue
                del resident[v]
            resident[e] = tick

    n_tok = len(tokens) - start
    return {
        "demand_miss": demand_miss,
        "pf_issued": pf_issued,
        "pf_used": pf_used,
        "saved": saved,
        "n_tok": n_tok,
        "precision": pf_used/max(1, pf_issued),
        "pf_mb_per_tok": pf_issued*expert_mb/max(1, n_tok),
        "demand_mb_per_tok": demand_miss*expert_mb/max(1, n_tok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--mult", type=int, default=8, help="cache size; high = the regime that matters")
    ap.add_argument("--expert-mb", type=float, default=1.09)
    ap.add_argument("--pcie", type=float, default=7.09, help="GB/s measured")
    ap.add_argument("--decode-ms", type=float, default=24.2, help="ms/token at this cache size")
    a = ap.parse_args()

    t = load(a.trace, a.limit)
    n_slot = min(a.mult*t["n_used"], t["n_expert"])
    step = max(1, t["n_layer"]//a.layers)
    layers = list(range(0, t["n_layer"], step))[:a.layers]

    print(f"\n{a.trace}")
    print(f"{t['n_rec']} records, {t['n_expert']} experts, {t['n_used']} active, "
          f"cache {n_slot} ({a.mult}x)")
    print(f"layers {layers}, expert {a.expert_mb} MB, PCIe {a.pcie} GB/s, "
          f"decode {a.decode_ms} ms/token\n")

    table, seen, n_train = build_table(t["rows"], t["tokens"], layers, t["n_expert"])
    print(f"predictor trained on {n_train} tokens, evaluated on {t['n_rec'] - n_train} held out\n")

    # Only a few layers are simulated, so their traffic must be scaled to the whole model
    # before being compared against a per-token bandwidth budget. Getting this wrong made the
    # bus look 7% busy when it is nearer 87%, which is the difference between "prefetching is
    # free" and "prefetching displaces the transfers you actually need".
    scale = t["n_layer"]/len(layers)
    budget_mb = a.pcie*1000.0*(a.decode_ms/1000.0)   # MB movable per token at full bus
    print(f"bus budget {budget_mb:.0f} MB/token at 100% utilisation")
    print(f"scaling {len(layers)} sampled layers to all {t['n_layer']} (x{scale:.0f})\n")

    hdr = f"{'thresh':>7} {'precision':>10} {'misses saved':>13} {'demand MB':>10} {'prefetch MB':>12} {'bus use':>9}"
    print(hdr)
    print("-"*len(hdr))

    base = None
    for thr in (1.01, 0.9, 0.7, 0.5, 0.3, 0.15):
        agg = defaultdict(float)
        for il in layers:
            r = simulate(t["rows"], t["tokens"], il, table, seen, n_slot,
                         t["n_expert"], n_train, thr, a.expert_mb)
            for k, v in r.items():
                agg[k] += v

        prec = agg["pf_used"]/max(1, agg["pf_issued"])
        dmb = agg["demand_mb_per_tok"]*scale
        pmb = agg["pf_mb_per_tok"]*scale
        if base is None:
            base = dmb                       # thresh 1.01 issues nothing: the no-prefetch case
        savedpct = 100.0*(base - dmb)/max(1e-9, base)
        use = 100.0*(dmb + pmb)/budget_mb

        label = "none" if thr > 1.0 else f"{thr:.2f}"
        print(f"{label:>7} {prec:>9.1%} {savedpct:>12.1f}% {dmb:>10.1f} {pmb:>12.1f} {use:>8.0f}%")

    print("\nReading this. 'misses saved' is demand traffic removed, which is the win.")
    print("'prefetch MB' is speculative traffic added, which is the cost. It is only free")
    print("while total bus use stays under 100% - past that, prefetch displaces demand")
    print("traffic and the old failure mode returns.")


if __name__ == "__main__":
    main()
