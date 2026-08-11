"""Check whether a generation degenerated into a loop.

A repetitive output would make the expert cache look brilliant for the wrong reason: the
same few experts get used every token, hit rate goes to ~100%, and the speedup is an
artifact of the model breaking rather than the cache working.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bells_trace import load

for path in sys.argv[1:]:
    tr = load(path)
    g = tr.token_id[tr.generated]

    uniq = len(set(g.tolist()))
    frac_unique = uniq/max(1, len(g))

    # longest run of a single repeated token
    longest = best = 1
    for i in range(1, len(g)):
        best = best + 1 if g[i] == g[i-1] else 1
        longest = max(longest, best)

    # does a short cycle explain the tail?
    tail = g[-60:].tolist() if len(g) >= 60 else g.tolist()
    cyc = 0
    for period in range(1, 13):
        if len(tail) > 2*period and tail[-period:] == tail[-2*period:-period]:
            cyc = period
            break

    # distinct experts actually used, layer 0, as a routing-diversity proxy
    e0 = tr.experts[tr.generated][:, 0, :].ravel()
    print(f"{os.path.basename(path):<22} tokens={len(g):>4} unique={uniq:>4} "
          f"({100*frac_unique:>5.1f}%)  max_run={longest:>3}  cycle={cyc or '-'}  "
          f"distinct_experts_L0={len(set(e0.tolist())):>4}")
    print(f"  last 24: {g[-24:].tolist()}")
