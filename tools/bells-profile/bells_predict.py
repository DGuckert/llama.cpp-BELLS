"""Evaluates expert predictors against the reactive floor and the oracle ceiling.

Builds a token-id -> per-layer expert count table on a training split, then measures how
well it fills a VRAM expert cache on held-out tokens. Reported next to LRU (no prediction)
and Belady (perfect prediction) at identical capacities, so the three are comparable.

    python bells_predict.py olmoe-big.trace.bin

The number that matters is where "predict" lands between "LRU" and "Belady". That gap is
the entire prize; if the table captures most of it, a trained model has little left to win.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bells_trace import load


def build_table(token_id, experts, n_expert):
    """token id -> (n_layer, n_expert) counts, plus the global prior for unseen tokens."""
    n_layer = experts.shape[1]
    uniq, inv = np.unique(token_id, return_inverse=True)

    table = np.zeros((len(uniq), n_layer, n_expert), dtype=np.float32)
    layers = np.arange(n_layer)[None, :, None]

    for start in range(0, len(token_id), 4096):
        sl = slice(start, start + 4096)
        np.add.at(table, (inv[sl][:, None, None], layers, experts[sl]), 1.0)

    prior = table.sum(0)
    prior /= max(prior.max(), 1.0)

    return uniq, table, prior


def topk_mask(scores, cap):
    """Boolean mask of the `cap` highest-scoring experts along the last axis."""
    idx = np.argpartition(-scores, cap - 1, axis=-1)[..., :cap]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=-1)
    return mask


def predicted_ids(uniq, table, prior, token_id, n_take, alpha=0.5):
    """(n_records, n_layer, n_take) expert ids the table ranks highest for each token.

    Raw counts, not per-row frequencies: a token seen once must still outrank the global
    prior, which is only here to break ties among experts with equal counts.
    """
    scores = table + alpha*prior[None]

    order = np.argpartition(-scores, n_take - 1, axis=-1)[..., :n_take]
    fallback = np.argpartition(-prior, n_take - 1, axis=-1)[..., :n_take]

    pos = np.searchsorted(uniq, token_id)
    pos[pos >= len(uniq)] = 0
    known = uniq[pos] == token_id

    out = np.where(known[:, None, None], order[pos], fallback[None])

    return out, known.mean()


def sim_predict_only(pred, experts, n_expert):
    """Cache holds exactly the predicted set. Hit rate == recall@len(pred)."""
    hits = 0
    for start in range(0, len(experts), 2048):
        sl = slice(start, start + 2048)
        p = pred[sl]
        e = experts[sl]
        m = np.zeros((len(e), e.shape[1], n_expert), dtype=bool)
        r = np.arange(len(e))[:, None, None]
        l = np.arange(e.shape[1])[None, :, None]
        m[r, l, p] = True
        hits += m[r, l, e].sum()
    return hits/experts.size


def sim_hybrid(pred, actual, n_expert, cap):
    """Prefetch predictions into an LRU cache, then serve real accesses from it.

    This is the policy a real system runs: prediction supplies lookahead, recency covers
    what prediction misses. Misses are loaded on demand.
    """
    resident = np.zeros(n_expert, bool)
    last = np.full(n_expert, -1, np.int64)
    live = []
    hits = 0
    total = 0

    def admit(e, i, protected):
        if resident[e]:
            return
        if len(live) >= cap:
            j = None
            oldest = None
            for k, x in enumerate(live):
                if x in protected:
                    continue
                if oldest is None or last[x] < oldest:
                    oldest = last[x]
                    j = k
            if j is None:
                return
            resident[live[j]] = False
            live[j] = live[-1]
            live.pop()
        resident[e] = True
        live.append(e)

    for i in range(len(actual)):
        want = set(int(x) for x in pred[i])
        for e in pred[i]:
            admit(int(e), i, want)

        for e in actual[i]:
            e = int(e)
            total += 1
            if resident[e]:
                hits += 1
            else:
                admit(e, i, want)
            last[e] = i

    return hits/total


def sim_lru(seq, n_expert, cap):
    last = np.full(n_expert, -1, np.int64)
    resident = np.zeros(n_expert, bool)
    live = []
    hits = 0

    for i, e in enumerate(seq):
        if resident[e]:
            hits += 1
            last[e] = i
            continue
        if len(live) >= cap:
            j = min(range(len(live)), key=lambda k: last[live[k]])
            resident[live[j]] = False
            live[j] = live[-1]
            live.pop()
        resident[e] = True
        last[e] = i
        live.append(e)

    return hits/len(seq)


def sim_opt(seq, n_expert, cap):
    occ = [[] for _ in range(n_expert)]
    for i, e in enumerate(seq):
        occ[e].append(i)
    ptr = [0]*n_expert
    INF = len(seq) + 1

    def nxt(x, i):
        o = occ[x]
        p = ptr[x]
        while p < len(o) and o[p] <= i:
            p += 1
        ptr[x] = p
        return o[p] if p < len(o) else INF

    resident = np.zeros(n_expert, bool)
    live = []
    hits = 0

    for i, e in enumerate(seq):
        nxt(e, i)
        if resident[e]:
            hits += 1
            continue
        if len(live) >= cap:
            j = max(range(len(live)), key=lambda k: nxt(live[k], i))
            resident[live[j]] = False
            live[j] = live[-1]
            live.pop()
        resident[e] = True
        live.append(e)

    return hits/len(seq)


def main(path, max_records=None):
    tr = load(path)

    n = len(tr.token_id)
    if max_records is not None:
        n = min(n, max_records)
    cut = int(n*0.8)

    print(tr)
    print(f"train {cut} records, test {n - cut} records\n")

    uniq, table, prior = build_table(tr.token_id[:cut], tr.experts[:cut], tr.n_expert)

    te_tok = tr.token_id[cut:n]
    te_exp = tr.experts[cut:n]

    # capacity in multiples of the per-token working set. A cache smaller than n_expert_used
    # cannot even hold one token's experts, so 1x is the hard floor.
    mults = [1, 2, 4, 8]

    print(f"{'cap':>5} {'xK':>4} {'%exp':>6} {'LRU':>8} {'pred':>8} {'hybrid':>8} {'Belady':>8}")

    seen = 0.0

    for mult in mults:
        cap = mult*tr.n_expert_used
        if cap > tr.n_expert:
            break

        ids, seen = predicted_ids(uniq, table, prior, te_tok, cap)

        only = sim_predict_only(ids, te_exp, tr.n_expert)

        # the hybrid must prefetch less than the full cache, otherwise predictions evict
        # every historical entry and it degenerates into predict-only
        n_pre = max(tr.n_expert_used, cap//2)
        ids_pre, _ = predicted_ids(uniq, table, prior, te_tok, n_pre)

        hyb = np.mean([sim_hybrid(ids_pre[:, l, :], te_exp[:, l, :], tr.n_expert, cap)
                       for l in range(tr.n_layer_moe)])
        lru = np.mean([sim_lru(te_exp[:, l, :].ravel(), tr.n_expert, cap)
                       for l in range(tr.n_layer_moe)])
        opt = np.mean([sim_opt(te_exp[:, l, :].ravel(), tr.n_expert, cap)
                       for l in range(tr.n_layer_moe)])

        pct = 100.0*cap/tr.n_expert

        print(f"{cap:>5} {mult:>3}x {pct:>5.1f}% {100*lru:>7.1f}% {100*only:>7.1f}% "
              f"{100*hyb:>7.1f}% {100*opt:>7.1f}%")

    print(f"\ntest tokens seen during training: {100*seen:.1f}%")
    print("pred   = cache holds only the predicted set (recall@capacity)")
    print("hybrid = half the cache prefetched from predictions, half kept by LRU")
    print("Belady = optimal DEMAND policy: it may only cache what has already been asked")
    print("         for, so prefetching can and does exceed it. Not a universal ceiling.")
    print("captured = share of the LRU->Belady gap that hybrid closes; >100% means")
    print("         prefetching beat the best any reactive policy could do")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
