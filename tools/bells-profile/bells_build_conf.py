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

    # Counting is done with bincount rather than nested loops. At the trace sizes this is meant
    # for - 200k records, 48 layers, 8 ids each - the loop version is 79M Python iterations and
    # takes the better part of a day, which is how it escaped notice on a 7k-record trace.
    uniq, first_count = np.unique(tokens, return_counts=True)

    keep_mask = first_count >= a.min_count
    keep = uniq[keep_mask].astype(np.int64)
    if keep.size == 0:
        sys.exit("no token appears often enough to be worth a table entry")

    n_keep = int(keep.size)
    keep_n = first_count[keep_mask].astype(np.float64)   # sightings per kept token, aligned to keep

    # record -> row in the table, or -1 for tokens that did not clear min-count
    slot = np.searchsorted(keep, tokens)
    slot = np.where((slot < n_keep) & (keep[np.minimum(slot, n_keep - 1)] == tokens), slot, -1)
    kept_rec = slot >= 0
    slot_kept = slot[kept_rec]

    counts   = np.zeros((n_keep, n_layer, n_expert), dtype=np.int32)
    globals_ = np.zeros((n_layer, n_expert), dtype=np.int64)

    # One bincount per layer keeps the flat index under n_keep*n_expert instead of
    # n_keep*n_layer*n_expert, which for a 20k-token table is 2.6M instead of 123M.
    for il in range(n_layer):
        ids = rows[:, il, :].astype(np.int64)
        globals_[il] = np.bincount(ids.ravel(), minlength=n_expert)[:n_expert]

        flat = (slot_kept[:, None]*n_expert + ids[kept_rec]).ravel()
        counts[:, il, :] = np.bincount(
            flat, minlength=n_keep*n_expert).reshape(n_keep, n_expert).astype(np.int32)

    print(f"{t['n_rec']} records, {n_keep} tokens kept "
          f"(of {uniq.size} distinct, min-count {a.min_count})")

    max_k = a.max_k

    with open(a.out, "wb") as f:
        f.write(b"BELLSCF1")
        f.write(struct.pack("<4I", 1, n_layer, n_expert, max_k))
        f.write(struct.pack("<Q", len(keep)))
        f.write(np.array(t["layer_ids"] if "layer_ids" in t else range(n_layer),
                         dtype=np.uint32).tobytes())
        f.write(np.array(keep, dtype=np.int32).tobytes())

        # per token. The on-disk block is ex[max_k] then cf[max_k] for each (token, layer), and
        # both are 4 bytes wide, so the whole thing is built as one uint32 array with the pair
        # axis in the middle and written in a single call - 960k argsorts otherwise.
        block = np.empty((n_keep, n_layer, 2, max_k), dtype=np.uint32)

        for il in range(n_layer):
            c   = counts[:, il, :]                                  # (n_keep, n_expert)
            top = np.argsort(-c, axis=1, kind="stable")[:, :max_k]   # ties -> lowest expert id
            val = np.take_along_axis(c, top, axis=1)

            ex = np.where(val > 0, top, -1).astype(np.int32)
            cf = np.where(val > 0, val/keep_n[:, None], 0.0).astype(np.float32)

            block[:, il, 0, :] = ex.view(np.uint32)
            block[:, il, 1, :] = cf.view(np.uint32)

        f.write(block.tobytes())

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
    thresholds = (0.9, 0.7, 0.5, 0.3)
    n_pass = dict.fromkeys(thresholds, 0)

    for il in range(n_layer):
        conf = counts[:, il, :]/keep_n[:, None]
        for thr in thresholds:
            n_pass[thr] += int((conf >= thr).sum())

    for thr in thresholds:
        print(f"  {thr:.2f}: {n_pass[thr]/max(1, n_keep*n_layer):.2f}")


if __name__ == "__main__":
    main()
