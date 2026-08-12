#!/usr/bin/env python3
"""Compare cache eviction policies on a recorded routing trace.

Why this exists. Instrumenting the runtime showed copies are ~87% of the per-layer cost
(1356 us of 1552 us on Qwen3-30B at 17 slots), so the only lever that matters is moving fewer
experts. Two ways to do that: raise the hit rate, or transfer more cheaply. This measures the
first, offline and for free.

The important constraint: a policy that only changes *which* expert is evicted adds no PCIe
traffic at all. That is what killed the predictor - prefetching raised hit rate while moving a
superset of what was needed, and lost on wall clock. A better eviction rule has no such cost, so
a hit-rate gain here should convert almost directly into a speedup.

Semantics match bells_cache::ensure exactly:
  - one independent cache per layer
  - every expert a token asks for must be resident afterwards
  - an expert wanted by the current token is never evicted to make room for one of its peers

Usage:
    python bells_policy.py trace.bin
    python bells_policy.py trace.bin --mults 1,2,4,8 --limit 20000
"""

import argparse
import struct
import sys
from collections import defaultdict

import numpy as np

MAGIC = b"BELLSTR1"


def load(path, limit=None):
    with open(path, "rb") as f:
        blob = f.read()

    if blob[:8] != MAGIC:
        sys.exit(f"{path}: not a BELLS trace")

    version, n_layer, n_expert, n_used, n_vocab = struct.unpack_from("<5I", blob, 8)
    (n_rec,) = struct.unpack_from("<Q", blob, 28)

    layer_ids = np.frombuffer(blob, dtype=np.uint32, count=n_layer, offset=36).copy()

    off = 36 + n_layer*4                       # header + layer id list
    rec_bytes = 12 + n_layer*n_used*2          # token,pos,gen,pad + u16 rows

    avail = (len(blob) - off)//rec_bytes
    n_rec = min(n_rec, avail)
    if limit:
        n_rec = min(n_rec, limit)

    raw = np.frombuffer(blob, dtype=np.uint8, count=n_rec*rec_bytes, offset=off)
    raw = raw.reshape(n_rec, rec_bytes)

    tokens = raw[:, 0:4].copy().view(np.uint32).ravel()
    rows = raw[:, 12:].copy().view(np.uint16).reshape(n_rec, n_layer, n_used)

    return {
        "n_layer": n_layer, "n_expert": n_expert, "n_used": n_used,
        "n_rec": n_rec, "tokens": tokens, "rows": rows.astype(np.int32),
        "layer_ids": layer_ids,
    }


# ---------------------------------------------------------------- policies
#
# Each returns (hits, misses) for one layer's access sequence. `seq` is
# (n_rec, n_used) and every row must be fully resident after it is served.


def _victim(candidates, want, key):
    """Least-valuable resident that the current token does not need.

    Returns None when every resident is wanted, which the real runtime treats as ensure()
    failing. Only reachable when the usable region is smaller than n_expert_used.
    """
    best, chosen = None, None
    for x in candidates:
        if x in want:
            continue
        k = key(x)
        if best is None or k < best:
            best, chosen = k, x
    return chosen


def sim_lru(seq, n_slot, n_expert):
    resident = {}                      # expert -> last used tick
    hits = misses = 0
    tick = 0
    for row in seq:
        tick += 1
        want = set(row.tolist())
        for e in want:
            if e in resident:
                resident[e] = tick
                hits += 1
        for e in want:
            if e in resident:
                continue
            misses += 1
            if len(resident) >= n_slot:
                v = _victim(resident, want, lambda x: resident[x])
                if v is None:
                    continue
                del resident[v]
            resident[e] = tick
    return hits, misses


def sim_lfu(seq, n_slot, n_expert):
    """Evict least frequently used. Frequency counted over the whole run."""
    resident = {}
    freq = defaultdict(int)
    hits = misses = 0
    for row in seq:
        want = set(row.tolist())
        for e in want:
            freq[e] += 1
            if e in resident:
                hits += 1
        for e in want:
            if e in resident:
                continue
            misses += 1
            if len(resident) >= n_slot:
                v = _victim(resident, want, lambda x: freq[x])
                if v is None:
                    continue
                del resident[v]
            resident[e] = True
    return hits, misses


def sim_lfu_aged(seq, n_slot, n_expert, halflife=2000):
    """LFU with exponential decay, so old popularity fades.

    Plain LFU cannot forget: an expert that was hot early keeps its count forever and squats.
    """
    resident = {}
    score = defaultdict(float)
    hits = misses = 0
    step = 0
    decay = 0.5 ** (1.0/halflife)
    boost = 1.0
    for row in seq:
        step += 1
        boost /= decay                 # cheaper than decaying every counter
        want = set(row.tolist())
        for e in want:
            score[e] += boost
            if e in resident:
                hits += 1
        for e in want:
            if e in resident:
                continue
            misses += 1
            if len(resident) >= n_slot:
                v = _victim(resident, want, lambda x: score[x])
                if v is None:
                    continue
                del resident[v]
            resident[e] = True
    return hits, misses


def sim_static(seq, n_slot, n_expert, warmup=0.1):
    """Pin the globally hottest experts, no eviction at all.

    Routing is only mildly skewed, so this should lose - but it costs nothing to check, and if
    it were close it would be far simpler than LRU in the runtime.
    """
    n_warm = max(1, int(len(seq)*warmup))
    counts = np.bincount(seq[:n_warm].ravel(), minlength=n_expert)
    pinned = set(np.argsort(-counts)[:n_slot].tolist())

    hits = misses = 0
    for row in seq[n_warm:]:
        for e in set(row.tolist()):
            if e in pinned:
                hits += 1
            else:
                misses += 1
    return hits, misses


def sim_hybrid(seq, n_slot, n_expert, pin_frac=0.5, warmup=0.1):
    """Pin the hottest experts in part of the cache, run LRU in the rest.

    The idea: a hot core that never gets evicted by a burst of cold traffic, plus an LRU region
    that adapts. This is ARC's intuition without ARC's bookkeeping.
    """
    n_pin = int(n_slot*pin_frac)
    n_lru = n_slot - n_pin
    if n_lru < 1:
        return sim_static(seq, n_slot, n_expert, warmup)

    n_warm = max(1, int(len(seq)*warmup))
    counts = np.bincount(seq[:n_warm].ravel(), minlength=n_expert)
    pinned = set(np.argsort(-counts)[:n_pin].tolist())

    resident = {}
    hits = misses = 0
    tick = 0
    for row in seq[n_warm:]:
        tick += 1
        want = set(row.tolist())
        for e in want:
            if e in pinned:
                hits += 1
            elif e in resident:
                resident[e] = tick
                hits += 1
        for e in want:
            if e in pinned or e in resident:
                continue
            misses += 1
            if len(resident) >= n_lru:
                v = _victim(resident, want, lambda x: resident[x])
                if v is None:
                    continue
                del resident[v]
            resident[e] = tick
    return hits, misses


def sim_belady(seq, n_slot, n_expert):
    """Offline optimum: evict whatever is needed furthest in the future.

    A bound, not a policy - it needs the future. Anything close to this has nothing left to win
    from smarter eviction, and the remaining cost has to be attacked elsewhere.
    """
    n = len(seq)
    nxt = [None]*n
    last_seen = {}
    for i in range(n - 1, -1, -1):
        cur = {}
        for e in seq[i].tolist():
            cur[e] = last_seen.get(e, n)
        nxt[i] = cur
        for e in seq[i].tolist():
            last_seen[e] = i

    resident = set()
    hits = misses = 0
    for i, row in enumerate(seq):
        want = set(row.tolist())
        for e in want:
            if e in resident:
                hits += 1
        for e in want:
            if e in resident:
                continue
            misses += 1
            if len(resident) >= n_slot:
                # furthest next use among residents not wanted right now
                far, victim = -1, None
                for x in resident:
                    if x in want:
                        continue
                    d = nxt[i].get(x)
                    if d is None:
                        d = _next_use(seq, x, i + 1, n)
                    if d > far:
                        far, victim = d, x
                if victim is None:
                    break
                resident.discard(victim)
            resident.add(e)
    return hits, misses


_nu_cache = {}


def _next_use(seq, expert, start, n):
    key = (id(seq), expert, start)
    hit = _nu_cache.get(key)
    if hit is not None:
        return hit
    for j in range(start, min(start + 4096, n)):
        if expert in seq[j]:
            _nu_cache[key] = j
            return j
    _nu_cache[key] = n
    return n


def sim_lru2(seq, n_slot, n_expert):
    """LRU-K with K=2: evict on the *second* most recent use.

    Classic result - one-off touches cannot promote an expert to the front, so a burst of cold
    traffic does not evict the working set. This is the cheapest known way to capture part of
    the Belady gap, and it needs one extra integer per resident expert.
    """
    last1 = {}
    last2 = {}
    hits = misses = 0
    tick = 0
    for row in seq:
        tick += 1
        want = set(row.tolist())
        for e in want:
            if e in last1:
                last2[e] = last1[e]
                last1[e] = tick
                hits += 1
        for e in want:
            if e in last1:
                continue
            misses += 1
            if len(last1) >= n_slot:
                # oldest second-reference; never-twice-used experts sort oldest
                v = _victim(last1, want, lambda x: last2.get(x, -1))
                if v is None:
                    continue
                del last1[v]
                last2.pop(v, None)
            last1[e] = tick
    return hits, misses


def sim_rank_lru(seq, n_slot, n_expert, decay=0.9):
    """Recency, but weighted by how strongly the router wanted the expert.

    The trace stores each token's experts in topk order, so column 0 is the highest-weight
    choice. LRU discards that entirely: an expert scraped in at rank 7 is treated exactly like
    one the router put first. Here each touch adds a value that decays with rank, and eviction
    takes the lowest score. Free - the ordering is already in the routing tensor the runtime
    reads anyway.
    """
    score = {}
    hits = misses = 0
    for row in seq:
        ranked = row.tolist()
        want = set(ranked)
        for r, e in enumerate(ranked):
            w = decay ** r
            if e in score:
                score[e] = score[e]*0.98 + w
                hits += 1
        for r, e in enumerate(ranked):
            if e in score:
                continue
            misses += 1
            if len(score) >= n_slot:
                v = _victim(score, want, lambda x: score[x])
                if v is None:
                    continue
                del score[v]
            score[e] = decay ** r
    return hits, misses


POLICIES = [
    ("LRU (current)", sim_lru),
    ("LRU-2", sim_lru2),
    ("rank-LRU", sim_rank_lru),
    ("LFU", sim_lfu),
    ("LFU aged", sim_lfu_aged),
    ("static pinned", sim_static),
    ("hybrid 50/50", sim_hybrid),
    ("Belady (bound)", sim_belady),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--mults", default="1,2,4,8",
                    help="cache sizes as multiples of n_expert_used")
    ap.add_argument("--limit", type=int, default=8000, help="records to replay")
    ap.add_argument("--layers", type=int, default=8, help="layers to sample")
    ap.add_argument("--no-belady", action="store_true", help="skip the slow optimum")
    a = ap.parse_args()

    t = load(a.trace, a.limit)
    print(f"\n{a.trace}")
    print(f"{t['n_rec']} records, {t['n_layer']} layers, {t['n_expert']} experts, "
          f"{t['n_used']} active")

    uniq = len(set(t["tokens"].tolist()))
    print(f"token diversity {uniq}/{t['n_rec']} ({100*uniq/max(1,t['n_rec']):.0f}%)"
          f"{'  <-- DEGENERATE, results meaningless' if uniq < 0.2*t['n_rec'] else ''}")

    # sample layers evenly rather than replaying all of them
    step = max(1, t["n_layer"]//a.layers)
    layers = list(range(0, t["n_layer"], step))[:a.layers]
    print(f"replaying layers {layers}\n")

    policies = [p for p in POLICIES if not (a.no_belady and "Belady" in p[0])]
    mults = [int(x) for x in a.mults.split(",")]

    print(f"{'cache':>8} {'ratio':>6} " + "".join(f"{n:>16}" for n, _ in policies))
    print("-"*(15 + 16*len(policies)))

    results = {}
    for m in mults:
        n_slot = min(m*t["n_used"], t["n_expert"])
        line = f"{n_slot:>8} {m:>5}x "
        for name, fn in policies:
            hits = misses = 0
            for il in layers:
                h, mi = fn(t["rows"][:, il, :], n_slot, t["n_expert"])
                hits += h
                misses += mi
            rate = 100.0*hits/max(1, hits + misses)
            results[(m, name)] = rate
            line += f"{rate:>15.1f}%"
        print(line)

    # what a policy change is actually worth
    print("\nmiss reduction vs LRU (this is the part that becomes speed):")
    for m in mults:
        base = results.get((m, "LRU (current)"))
        if base is None:
            continue
        best_name, best = None, base
        for name, _ in policies:
            if "Belady" in name or name == "LRU (current)":
                continue
            r = results.get((m, name), 0)
            if r > best:
                best_name, best = name, r
        n_slot = min(m*t["n_used"], t["n_expert"])
        if best_name:
            cut = 100.0*(best - base)/max(1e-9, 100.0 - base)
            print(f"  {n_slot:>4} slots ({m}x): {best_name} beats LRU "
                  f"{base:.1f}% -> {best:.1f}%, {cut:.0f}% fewer misses")
        else:
            print(f"  {n_slot:>4} slots ({m}x): nothing beats LRU ({base:.1f}%)")


if __name__ == "__main__":
    main()
