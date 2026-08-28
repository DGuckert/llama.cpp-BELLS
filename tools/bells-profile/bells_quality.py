"""Compare generated token sequences with alignment.

Position-by-position comparison is wrong here: a single inserted token shifts everything
after it and scores as total divergence when the text is nearly identical. Use a sequence
matcher so insertions cost one token, not the whole tail.
"""

import difflib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bells_trace import load

ref = load(sys.argv[1])
got = load(sys.argv[2])

r = ref.token_id[ref.generated].tolist()
g = got.token_id[got.generated].tolist()

sm = difflib.SequenceMatcher(a=r, b=g, autojunk=False)

matched = sum(block.size for block in sm.get_matching_blocks())
ratio = sm.ratio()

# strict positional, for contrast with the naive metric
n = min(len(r), len(g))
positional = sum(1 for i in range(n) if r[i] == g[i])

print(f"reference tokens : {len(r)}")
print(f"bells tokens     : {len(g)}")
print()
print(f"aligned similarity : {100*ratio:.1f}%   ({matched} tokens matched)")
print(f"positional match   : {100*positional/n:.1f}%   (misleading after any insertion)")
print()

ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
print(f"edits: {len(ops)}")
for tag, i1, i2, j1, j2 in ops[:8]:
    print(f"  {tag:<8} ref[{i1}:{i2}]={r[i1:i2]}  bells[{j1}:{j2}]={g[j1:j2]}")
