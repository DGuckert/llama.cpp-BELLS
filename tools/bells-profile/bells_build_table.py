"""Builds the runtime predictor table from a routing trace.

    python bells_build_table.py olmoe-big.trace.bin olmoe.bells

Counts, per token id and per layer, which experts the router picked, then stores the top
n_take of them ranked. The runtime looks this up on the token id -- which is known before
layer 0 runs -- and prefetches the result, buying a full forward pass of lead time.

Format (little endian), matching bells_predictor::load in src/llama-bells.cpp:

    "BELLSPR1", u32 version, u32 n_layer, u32 n_take, u32 n_token
    i32 tokens[n_token]                    sorted, for binary search
    i32 fallback[n_layer*n_take]           global prior, for unseen tokens
    i32 ranked[n_token*n_layer*n_take]
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bells_trace import load


def build(tr, n_take, alpha=0.5):
    uniq, inv = np.unique(tr.token_id, return_inverse=True)
    n_layer = tr.n_layer_moe

    counts = np.zeros((len(uniq), n_layer, tr.n_expert), dtype=np.float32)
    layers = np.arange(n_layer)[None, :, None]

    for start in range(0, len(tr.token_id), 4096):
        sl = slice(start, start + 4096)
        np.add.at(counts, (inv[sl][:, None, None], layers, tr.experts[sl]), 1.0)

    prior = counts.sum(0)
    prior /= max(prior.max(), 1.0)

    # raw counts, so a token seen once still outranks the prior; the prior only breaks ties
    scores = counts + alpha*prior[None]

    ranked = np.argsort(-scores, axis=-1)[..., :n_take].astype(np.int32)
    fallback = np.argsort(-prior, axis=-1)[..., :n_take].astype(np.int32)

    return uniq.astype(np.int32), ranked, fallback


def main(trace_path, out_path, n_take=None):
    tr = load(trace_path)

    if n_take is None:
        n_take = 2*tr.n_expert_used

    n_take = min(n_take, tr.n_expert)

    print(tr)
    print(f"building table: n_take={n_take}")

    tokens, ranked, fallback = build(tr, n_take)

    with open(out_path, "wb") as f:
        f.write(b"BELLSPR1")
        f.write(struct.pack("<IIII", 1, tr.n_layer_moe, n_take, len(tokens)))
        f.write(tokens.tobytes())
        f.write(fallback.tobytes())
        f.write(ranked.tobytes())

    size = 24 + tokens.nbytes + fallback.nbytes + ranked.nbytes
    print(f"wrote {out_path}: {len(tokens)} tokens, {tr.n_layer_moe} layers, {size/1e6:.1f} MB")

    # sanity: how much of the real routing this table would have covered, in-sample
    hit = 0
    for start in range(0, len(tr.token_id), 2048):
        sl = slice(start, start + 2048)
        pos = np.searchsorted(tokens, tr.token_id[sl])
        r = ranked[pos]
        m = np.zeros((r.shape[0], tr.n_layer_moe, tr.n_expert), dtype=bool)
        idx = np.arange(r.shape[0])[:, None, None]
        lay = np.arange(tr.n_layer_moe)[None, :, None]
        m[idx, lay, r] = True
        hit += m[idx, lay, tr.experts[sl]].sum()

    print(f"in-sample recall@{n_take}: {100.0*hit/tr.experts.size:.1f}% "
          f"(optimistic, held-out is lower)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else None)
