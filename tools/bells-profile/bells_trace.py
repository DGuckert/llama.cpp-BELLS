"""Reader for the .trace.bin files written by llama-bells-profile.

Each record is one token: its id, its position, whether it was generated or fed in,
and the experts every MoE layer routed it to.

    from bells_trace import load
    tr = load("olmoe-profile.trace.bin")
    tr.experts.shape   # (n_records, n_layer_moe, n_expert_used)
    tr.token_id.shape  # (n_records,)
"""

import numpy as np


class Trace:
    def __init__(self, n_expert, n_expert_used, n_vocab, layer_ids,
                 token_id, pos, generated, experts):
        self.n_expert = n_expert
        self.n_expert_used = n_expert_used
        self.n_vocab = n_vocab
        self.layer_ids = layer_ids
        self.token_id = token_id
        self.pos = pos
        self.generated = generated
        self.experts = experts

    @property
    def n_layer_moe(self):
        return len(self.layer_ids)

    def onehot(self, layer):
        """(n_records, n_expert) float32 multi-hot of the experts used at `layer`."""
        out = np.zeros((len(self.token_id), self.n_expert), dtype=np.float32)
        rows = np.arange(len(self.token_id))[:, None]
        out[rows, self.experts[:, layer, :]] = 1.0
        return out

    def __repr__(self):
        return (f"Trace(records={len(self.token_id)}, layers={self.n_layer_moe}, "
                f"n_expert={self.n_expert}, n_expert_used={self.n_expert_used}, "
                f"generated={int(self.generated.sum())})")


def load(path):
    with open(path, "rb") as f:
        buf = f.read()

    if buf[:8] != b"BELLSTR1":
        raise ValueError(f"{path}: not a BELLS trace file")

    head = np.frombuffer(buf, dtype=np.uint32, count=5, offset=8)
    version, n_layer_moe, n_expert, n_expert_used, n_vocab = (int(x) for x in head)

    if version != 1:
        raise ValueError(f"{path}: unsupported version {version}")

    n_records = int(np.frombuffer(buf, dtype=np.uint64, count=1, offset=28)[0])
    layer_ids = np.frombuffer(buf, dtype=np.uint32, count=n_layer_moe, offset=36).astype(np.int32)

    offset = 36 + 4*n_layer_moe
    n_ids = n_layer_moe*n_expert_used

    rec = np.dtype([
        ("token_id",  "<u4"),
        ("pos",       "<u4"),
        ("generated", "u1"),
        ("pad",       "u1", 3),
        ("experts",   "<u2", n_ids),
    ])

    if rec.itemsize*n_records + offset != len(buf):
        raise ValueError(
            f"{path}: size mismatch, header implies "
            f"{rec.itemsize*n_records + offset} bytes but file is {len(buf)}"
        )

    data = np.frombuffer(buf, dtype=rec, count=n_records, offset=offset)

    return Trace(
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        n_vocab=n_vocab,
        layer_ids=layer_ids,
        token_id=data["token_id"].astype(np.int32),
        pos=data["pos"].astype(np.int32),
        generated=data["generated"].astype(bool),
        experts=data["experts"].reshape(n_records, n_layer_moe, n_expert_used).astype(np.int32),
    )


if __name__ == "__main__":
    import sys

    tr = load(sys.argv[1])
    print(tr)
    print("layers:", tr.layer_ids.tolist())

    # how much of layer L+1's expert set is already implied by layer L
    for l in range(min(4, tr.n_layer_moe - 1)):
        a = tr.experts[:, l, :]
        b = tr.experts[:, l + 1, :]
        overlap = np.array([len(set(x) & set(y)) for x, y in zip(a[:2000], b[:2000])])
        print(f"layer {tr.layer_ids[l]} -> {tr.layer_ids[l+1]}: "
              f"mean shared experts {overlap.mean():.2f} / {tr.n_expert_used}")
