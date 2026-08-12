#!/usr/bin/env python3
"""Build a confidence-carrying predictor table from a routing trace.

The table the old predictor used stored a ranked list of experts and nothing else, which is why
it could only be used as "prefetch the top N" - and prefetching the top N is what made it move a
superset of what was needed. This one stores a confidence with every candidate, so the runtime
can prefetch only where it is actually sure.

Format (little endian), read by bells_conf::load in src/llama-bells.cpp:

    magic "BELLSCF1", u32 version, u32 n_layer, u32 n_expert, u32 max_k, u64 n_token
    u32 layer_ids[n_layer]
    i32 tokens[n_token]                       sorted, for binary search
    per token, per layer: i32 expert[max_k], f32 conf[max_k]     expert -1 = empty
    then the same block once more as a global fallback for unseen tokens

Confidence is P(expert used | this token id, this layer), counted over the trace.

    python bells_build_conf.py trace.bin out.conf --min-count 2 --max-k 8
"""

import argparse
import struct
import sys
from collections import defaultdict

import numpy as np

from bells_policy import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--max-k", type=int, default=8,
                    help="candidates kept per token per layer")
    ap.add_argument("--min-count", type=int, default=2,
                    help="ignore tokens seen fewer times than this; one sighting is noise")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole trace")
    a = ap.parse_args()

    t = load(a.trace, a.limit or None)
    n_layer, n_expert = t["n_layer"], t["n_expert"]
    rows, tokens = t["rows"], t["tokens"]

    seen = defaultdict(int)
    for tok in tokens.tolist():
        seen[int(tok)] += 1

    keep = sorted(tok for tok, c in seen.items() if c >= a.min_count)
    if not keep:
        sys.exit("no token appears often enough to be worth a table entry")

    index = {tok: i for i, tok in enumerate(keep)}

    counts = np.zeros((len(keep), n_layer, n_expert), dtype=np.int32)
    globals_ = np.zeros((n_layer, n_expert), dtype=np.int64)

    for i in range(t["n_rec"]):
        tok = int(tokens[i])
        gi = index.get(tok)
        for il in range(n_layer):
            for e in rows[i, il, :]:
                globals_[il, int(e)] += 1
                if gi is not None:
                    counts[gi, il, int(e)] += 1

    print(f"{t['n_rec']} records, {len(keep)} tokens kept "
          f"(of {len(seen)} distinct, min-count {a.min_count})")

    max_k = a.max_k

    with open(a.out, "wb") as f:
        f.write(b"BELLSCF1")
        f.write(struct.pack("<4I", 1, n_layer, n_expert, max_k))
        f.write(struct.pack("<Q", len(keep)))
        f.write(np.array(t["layer_ids"] if "layer_ids" in t else range(n_layer),
                         dtype=np.uint32).tobytes())
        f.write(np.array(keep, dtype=np.int32).tobytes())

        # per token
        for gi, tok in enumerate(keep):
            n = seen[tok]
            for il in range(n_layer):
                c = counts[gi, il]
                top = np.argsort(-c)[:max_k]
                ex = np.full(max_k, -1, dtype=np.int32)
                cf = np.zeros(max_k, dtype=np.float32)
                for j, e in enumerate(top):
                    if c[e] <= 0:
                        break
                    ex[j] = e
                    cf[j] = c[e]/float(n)
                f.write(ex.tobytes())
                f.write(cf.tobytes())

        # global fallback
        total = max(1, t["n_rec"])
        for il in range(n_layer):
            c = globals_[il]
            top = np.argsort(-c)[:max_k]
            ex = np.full(max_k, -1, dtype=np.int32)
            cf = np.zeros(max_k, dtype=np.float32)
            for j, e in enumerate(top):
                if c[e] <= 0:
                    break
                ex[j] = e
                cf[j] = c[e]/float(total)
            f.write(ex.tobytes())
            f.write(cf.tobytes())

    import os
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB, max_k {max_k})")

    # what the runtime will actually see at each threshold
    print("\ncandidates per layer that clear each threshold (mean over kept tokens):")
    for thr in (0.9, 0.7, 0.5, 0.3):
        n_pass = 0
        for gi, tok in enumerate(keep):
            n = seen[tok]
            n_pass += int(((counts[gi]/float(n)) >= thr).sum())
        print(f"  {thr:.2f}: {n_pass/max(1, len(keep)*n_layer):.2f}")


if __name__ == "__main__":
    main()
