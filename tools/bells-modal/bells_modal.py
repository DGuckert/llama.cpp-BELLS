"""BELLS benchmark on Modal.

The question this exists to answer: does an expert cache beat plain layer offloading at
the SAME VRAM budget? Locally we only ever compared against --cpu-moe, because a 6 GB card
can't hold enough of these models for -ngl to be a real option. On a 24 GB card it is, and
it's the honest competitor.

Caveat that applies to every number this produces: the A10G instance has 8 vCPU. A slow CPU
inflates every BELLS result, because BELLS wins by moving work off the CPU. Cloud numbers
are not desktop numbers.

    modal run bells_modal.py::build          # warm the image, no GPU
    modal run bells_modal.py::fetch_model    # download into a Volume, no GPU
    modal run bells_modal.py::bench          # the actual measurement, GPU

Image builds are CPU-only on purpose: nvcc cross-compiles fine without a GPU present,
and GPU minutes are the expensive part.
"""

import modal

UPSTREAM = "https://github.com/antirez/llama.cpp-deepseek-v4-flash"
BASE_REV = "2f2d44052b7d15c9c4dd6610f6e14a5f7b2d5f3f"

# sm_86 covers A10G and RTX 3090, which is the only target that matters here. Building for
# four architectures compiled every .cu file four times and put the build at ~100 minutes.
CUDA_ARCHS = "86"

app = modal.App("bells-bench")

models = modal.Volume.from_name("bells-models", create_if_missing=True)
results = modal.Volume.from_name("bells-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "cmake", "ninja-build", "build-essential", "curl", "libcurl4-openssl-dev")
    .pip_install("huggingface_hub[hf_transfer]", "numpy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Point at the overlay subdirectory only. If this included the directory holding this
    # script, every edit to the benchmark would invalidate the layer and force a full
    # CUDA rebuild.
    .add_local_dir(
        "C:/Users/Daniel/AppData/Local/Temp/claude/C--Users-Daniel/dc34c62a-4c48-46b9-9018-044226134f7f/scratchpad/modal/new",
        remote_path="/bells-src/new",
        copy=True,
    )
    .run_commands(
        f"git clone {UPSTREAM} /llama && cd /llama && git checkout {BASE_REV}",
        # Overlay whole files rather than applying a diff: the patch was generated on
        # Windows and CRLF vs LF makes `git apply` reject it against a Linux clone.
        "cp -rv /bells-src/new/. /llama/",
        "cd /llama && git status --short | head -30",
        # ggml-cuda calls the CUDA *driver* API (cuGetErrorString and friends). libcuda.so
        # only exists where a driver is installed, so a CPU-only builder fails at link time.
        # Point the linker at the stubs, and attach a cheap GPU so a real driver is present.
        f'cd /llama && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release '
        f'-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="{CUDA_ARCHS}" '
        f'-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF '
        f'-DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" '
        f'-DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs"',
        "cd /llama && LIBRARY_PATH=/usr/local/cuda/lib64/stubs cmake --build build -j $(nproc)",
        gpu="T4",
    )
)


@app.function(image=image, timeout=3600)
def build():
    """Force the image to build and report what came out."""
    import subprocess
    out = subprocess.run(["ls", "-la", "/llama/build/bin"], capture_output=True, text=True)
    print(out.stdout[-3000:])
    return "ok"


@app.function(image=image, volumes={"/models": models}, timeout=14400)
def fetch_model(repo: str, filename: str):
    """Fetch one GGUF, or every shard of a split one.

    Anything over HF's 50 GB per-file limit ships as `-00001-of-000NN.gguf` shards. llama.cpp
    opens the set by being handed shard 1, but only if the others sit next to it under their
    original names, so flatten to basename and keep them together.

    Pass a comma-separated list to grab several shards.
    """
    import os
    import shutil
    from huggingface_hub import hf_hub_download

    saved = []

    for fn in [x.strip() for x in filename.split(",") if x.strip()]:
        dest = f"/models/{os.path.basename(fn)}"

        if os.path.exists(dest):
            print(f"already present: {dest} ({os.path.getsize(dest)/1e9:.1f} GB)")
            saved.append(dest)
            continue

        print(f"downloading {repo}/{fn}")
        os.makedirs("/models", exist_ok=True)

        # download straight into the volume: /tmp and /models are different filesystems, so
        # os.rename across them fails with EXDEV
        path = hf_hub_download(repo_id=repo, filename=fn, local_dir="/models/_dl")
        shutil.move(path, dest)
        models.commit()

        print(f"saved {dest} ({os.path.getsize(dest)/1e9:.1f} GB)")
        saved.append(dest)

    total = sum(os.path.getsize(p) for p in saved)
    print(f"\n{len(saved)} file(s), {total/1e9:.1f} GB total")

    return saved


BIN = "/llama/build/bin/llama-bells-profile"


def _run(args, tag):
    """Run one configuration and pull the decode timing out of the log."""
    import re
    import subprocess
    import time

    t0 = time.time()
    p = subprocess.run([BIN] + args, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0

    blob = p.stdout + p.stderr

    ms = None
    m = re.search(r"decode ([\d.]+) ms/token, ([\d.]+) tok/s", blob)
    if m:
        ms, tps = float(m.group(1)), float(m.group(2))
    else:
        tps = None

    hit = None
    h = re.search(r"free: hit ([\d.]+)%", blob)
    if h:
        hit = float(h.group(1))

    # per layer-call: readback X us, copy Y us, upload Z us (N calls, T ms total)
    rdbk = copy_us = layer_ms = None
    r = re.search(r"readback ([\d.]+) us, copy ([\d.]+) us, upload [\d.]+ us "
                  r"\(\d+ calls, ([\d.]+) ms total\)", blob)
    if r:
        rdbk     = float(r.group(1))
        copy_us  = float(r.group(2))
        layer_ms = float(r.group(3))

    # init: N slots/layer of M experts, X.XX GiB VRAM (Y% of the card), serves ubatch <= Z
    slots = cache_gib = cache_pct = None
    c = re.search(r"init: (\d+) slots/layer of \d+ experts, ([\d.]+) GiB VRAM \((\d+)% of the card\)", blob)
    if c:
        slots     = int(c.group(1))
        cache_gib = float(c.group(2))
        cache_pct = int(c.group(3))

    print(f"\n=== {tag} ===")
    print(f"  ms/token {ms}   tok/s {tps}   cache hit {hit}   wall {wall:.0f}s")
    if cache_gib is not None:
        print(f"  cache {slots} slots = {cache_gib:.2f} GiB ({cache_pct}% of card)")
    if ms is None:
        print("  NO TIMING FOUND, tail follows:")
        print(blob[-2500:])

    return {"tag": tag, "ms": ms, "tps": tps, "hit": hit,
            "rdbk_us": rdbk, "copy_us": copy_us, "layer_ms": layer_ms,
            "slots": slots, "cache_gib": cache_gib, "cache_pct": cache_pct, "args": args}


@app.function(
    image=image,
    volumes={"/models": models, "/results": results},
    gpu="A10G",
    cpu=16.0,
    memory=131072,
    timeout=7200,
)
def readback_scaling(model: str, slots: str = "8,16,32,64,128,192", n_gen: int = 96):
    """Is the per-layer readback fixed overhead, or is it waiting on GPU compute?

    It is now the largest item in BELLS' per-layer accounting - ~230 us against ~150 us of copy
    once the source is pinned - and the second-stream result showed it is also the barrier that
    prevents copies overlapping compute. So it is the obvious next target.

    But the timer wraps ggml_backend_tensor_get, which blocks until the GPU has produced the
    routing tensor. Part of that 230 us may be the GPU computing the layer rather than
    synchronisation overhead, and removing the sync would not remove GPU work.

    Sweeping the cache size varies the miss rate over a wide range while holding the model,
    the layer count and the graph identical. If readback stays flat while copy time moves with
    the miss rate, it is fixed overhead and worth attacking. If readback tracks the misses, it
    is mostly GPU wait and the ceiling on any fix is much lower.

    Everything runs pinned, so copy time is at its floor and readback is not hidden behind it.
    """
    import json
    import os
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    path = f"/models/{model}"
    corpus = "/results/corpus.txt"
    if not os.path.exists(corpus):
        subprocess.run(
            ["bash", "-c", f"cat /llama/docs/*.md /llama/*.md > {corpus} 2>/dev/null || true"],
            check=False,
        )

    common = ["-m", path, "-f", corpus, "--chunks", "1", "-c", "512", "-n", str(n_gen),
              "-ngl", "99", "--cpu-moe-pinned"]
    out = []

    want = [int(x) for x in slots.split(",") if x.strip()]
    _run(common + ["-o", "/tmp/rw.json", "--bells-slots", str(want[0])], "warmup")

    for s in want:
        out.append(_run(common + ["-o", f"/tmp/rb{s}.json", "--bells-slots", str(s)],
                        f"{s} slots"))

    with open("/results/readback_scaling.json", "w") as f:
        json.dump(out, f, indent=2)
    results.commit()

    print("\n\n===== SUMMARY =====")
    print(f"{'slots':>6} {'hit':>7} {'readback':>10} {'copy':>10} {'layer tot':>11} {'ms/token':>9}")
    for r in out:
        if r["rdbk_us"] is None:
            print(f"{str(r['slots']):>6}   no counters")
            continue
        print(f"{str(r['slots']):>6} {r['hit']:>6.1f}% {r['rdbk_us']:>9.1f}us "
              f"{r['copy_us']:>9.1f}us {r['layer_ms']:>10.0f}ms {r['ms']:>9.2f}")
    print("\nIf readback is flat while copy falls with the hit rate, it is fixed overhead")
    print("and removing the sync is worth roughly its full value. If it falls too, it is")
    print("largely GPU wait and the fix buys only the overlap, not the time.")

    return out


@app.local_entrypoint()
def cpu_scaling(model: str = "Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf",
                slots: str = "192",
                cpus: str = "4,8,16,32"):
    """Vary ONLY the core count and watch what moves.

    The hypothesis under test: BELLS puts expert compute on the GPU, so its own throughput
    should barely respond to core count, while the --cpu-moe baseline scales with it. If that
    holds, every speedup ratio this project has quoted is really a statement about how weak the
    baseline's CPU was, and the ratio is not a property of BELLS at all.

    Same GPU, same model file, same slot count, same container image - only cpu= changes.
    """
    rows = []
    for c in [float(x) for x in cpus.split(",") if x.strip()]:
        print(f"\n########## cpu={c} ##########")
        res = bench.with_options(cpu=c).remote(model=model, slots=slots)
        base = next((r for r in res if r["tag"] == "cpu-moe baseline"), None)
        bell = next((r for r in res if r["tag"].startswith("BELLS")), None)
        rows.append((c, base and base["tps"], bell and bell["tps"]))

    print("\n\n===== CPU SCALING =====")
    print(f"{'cpu':>5} {'baseline':>10} {'BELLS':>10} {'ratio':>8}")
    for c, b, l in rows:
        r = f"{l/b:.2f}x" if (b and l) else "-"
        print(f"{c:>5} {str(b):>10} {str(l):>10} {r:>8}")


@app.function(
    image=image,
    volumes={"/models": models, "/results": results},
    gpu="A10G",
    cpu=16.0,
    memory=131072,
    timeout=7200,
)
def skip_sweep(model: str, slots: int = 16, skips: str = "0,8,16,24", n_gen: int = 128):
    """Does per-layer gating put a floor under BELLS?

    The claim to test. A layer whose cache misses too often costs more in PCIe traffic than the
    CPU work it replaces, so caching it is a loss; gate it off and that layer runs plain
    --cpu-moe. If that is right, the floor of the whole technique should be the baseline rather
    than the 0.16x GPT-OSS-120B currently measures - which is the configuration where the claim
    is worth the most and where it is most likely to fail, because the break-even assumes a
    memory-bound CPU path and MXFP4 dequantises per element.

    On desktop this was worth 1.144x and turned 0.944x into 1.080x. Here the interesting number
    is not the speedup but how close the worst case gets to 1.0.
    """
    import os
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    path = f"/models/{model}"
    corpus = "/results/corpus.txt"
    if not os.path.exists(corpus):
        subprocess.run(["bash", "-c", f"cat /llama/docs/*.md /llama/*.md > {corpus} 2>/dev/null || true"],
                       check=False)

    common = ["-m", path, "-f", corpus, "--chunks", "1", "-c", "512", "-n", str(n_gen),
              "-ngl", "99", "--cpu-moe"]

    os.environ.pop("BELLS_SKIP_LAYERS", None)

    # Warm the volume before timing anything. A first run in a fresh container reads the model
    # off network storage, and for a 59 GB model that dominates everything: an earlier attempt
    # measured the baseline at 0.80 tok/s against 13.21 for the same model on the same instance
    # type, and the resulting "6.70x" was page cache filling up across run order, not gating.
    _run(common + ["-o", "/tmp/warm.json", "-n", "16"], "WARMUP (discarded)")

    base = _run(common + ["-o", "/tmp/k0.json"], "baseline --cpu-moe")

    rows = []
    for sk in [int(x) for x in skips.split(",") if x.strip()]:
        os.environ["BELLS_SKIP_LAYERS"] = str(sk)
        rows.append((sk, _run(common + ["-o", f"/tmp/k{sk}.json", "--bells-slots", str(slots)],
                              f"BELLS {slots} slots, skip {sk}")))
    os.environ.pop("BELLS_SKIP_LAYERS", None)

    print(f"\n\n===== PER-LAYER GATING: {model} =====")
    print(f"{'skip':>6} {'tok/s':>9} {'ms/token':>10} {'vs baseline':>12}")
    b = base["tps"]
    print(f"{'none':>6} {b:>9.2f} {base['ms']:>10.2f} {1.0:>11.2f}x   (--cpu-moe)")
    for sk, r in rows:
        if r["tps"]:
            print(f"{sk:>6} {r['tps']:>9.2f} {r['ms']:>10.2f} {r['tps']/b:>11.2f}x")

    # Re-measure the baseline last. If the two baselines disagree the container was still
    # warming and every ratio above is drift rather than effect - which is exactly how the first
    # attempt at this produced a meaningless 6.70x.
    os.environ.pop("BELLS_SKIP_LAYERS", None)
    base2 = _run(common + ["-o", "/tmp/k99.json"], "baseline again (drift check)")

    if base["tps"] and base2["tps"]:
        drift = abs(base2["tps"] - base["tps"])/base["tps"]
        print(f"\nbaseline drift across the sweep: {100*drift:.1f}% "
              f"({base['tps']:.2f} -> {base2['tps']:.2f} tok/s)"
              f"{'   <-- TOO LARGE, results are drift not effect' if drift > 0.15 else '   (acceptable)'}")

    return [base] + [r for _, r in rows] + [base2]


@app.function(
    image=image,
    volumes={"/models": models, "/results": results},
    gpu="A10G",
    cpu=16.0,
    memory=131072,
    timeout=7200,
)
def slot_sweep(model: str, slots: str, n_gen: int = 128):
    """Full slot curve for one model, with the cache size the runtime actually allocated.

    The auto-sizer takes a third of free VRAM, a heuristic fitted to one model on a 6 GB card -
    a configuration since measured at 0.94-0.97x, i.e. one where BELLS does not help at all. To
    replace it with something defensible, the optimum has to be expressible in terms of a
    quantity available at load time. This collects the curve so that can be checked rather than
    guessed, which is what went wrong with the four predictors already retracted.

    Everything runs in one container so the baseline and every slot count share a warm volume.
    """
    import os
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    path = f"/models/{model}"
    corpus = "/results/corpus.txt"
    if not os.path.exists(corpus):
        subprocess.run(["bash", "-c", f"cat /llama/docs/*.md /llama/*.md > {corpus} 2>/dev/null || true"],
                       check=False)

    common = ["-m", path, "-f", corpus, "--chunks", "1", "-c", "512", "-n", str(n_gen),
              "-ngl", "99", "--cpu-moe"]

    base = _run(common + ["-o", "/tmp/s0.json"], "baseline")
    rows = [base]
    for s in [int(x) for x in slots.split(",") if x.strip()]:
        rows.append(_run(common + ["-o", f"/tmp/s{s}.json", "--bells-slots", str(s)],
                         f"BELLS {s} slots"))

    print(f"\n\n===== SLOT CURVE: {model} =====")
    print(f"{'slots':>6} {'cacheGiB':>9} {'%card':>6} {'hit%':>6} {'tok/s':>8} {'vs base':>8}")
    b = base["tps"]
    for r in rows:
        if r["tps"] is None:
            continue
        sl  = r["slots"] if r["slots"] is not None else 0
        cg  = f"{r['cache_gib']:.2f}" if r["cache_gib"] is not None else "-"
        cp  = f"{r['cache_pct']}" if r["cache_pct"] is not None else "-"
        hit = f"{r['hit']:.1f}" if r["hit"] is not None else "-"
        rel = f"{r['tps']/b:.2f}x" if b else "-"
        print(f"{sl:>6} {cg:>9} {cp:>6} {hit:>6} {r['tps']:>8.2f} {rel:>8}")

    best = max((r for r in rows if r["tps"] is not None and r["slots"]), key=lambda r: r["tps"], default=None)
    if best and b:
        print(f"\npeak: {best['slots']} slots, {best['cache_gib']:.2f} GiB "
              f"({best['cache_pct']}% of card), {best['tps']:.2f} tok/s = {best['tps']/b:.2f}x")

    return rows


@app.function(
    image=image,
    volumes={"/models": models, "/results": results},
    gpu="A10G",
    cpu=16.0,
    memory=131072,
    timeout=7200,
)
def decompose(model: str, slots: int = 4, n_gen: int = 128):
    """Split BELLS's cost into "what the mechanism costs" and "what the cache is worth".

    Motivated by the one result nothing explains: Mixtral with every expert resident hits 100%,
    copies nothing, and still measures 0.80x. Three configurations, same container, back to back:

        baseline   no cache at all
        passive    cache allocated, graph split and id readback still taken, but the matmuls
                   stay on the full expert stack and nothing is ever copied
        bells      the real thing

    baseline -> passive isolates VRAM occupancy plus the per-layer graph split plus the readback.
    passive -> bells is everything the cache actually buys.
    """
    import os
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    print(f"cpu count: {os.cpu_count()}")

    path = f"/models/{model}"
    corpus = "/results/corpus.txt"
    if not os.path.exists(corpus):
        subprocess.run(["bash", "-c", f"cat /llama/docs/*.md /llama/*.md > {corpus} 2>/dev/null || true"],
                       check=False)

    common = ["-m", path, "-f", corpus, "--chunks", "1", "-c", "512", "-n", str(n_gen),
              "-ngl", "99", "--cpu-moe"]

    base = _run(common + ["-o", "/tmp/d0.json"], "baseline (no cache)")
    pas  = _run(common + ["-o", "/tmp/d1.json", "--bells-slots", str(slots), "--bells-passive"],
                f"passive ({slots} slots allocated, unused)")
    act  = _run(common + ["-o", "/tmp/d2.json", "--bells-slots", str(slots)],
                f"BELLS ({slots} slots, active)")

    print("\n\n===== DECOMPOSITION =====")
    for r in (base, pas, act):
        print(f"{r['tag']:<40} {str(r['ms']):>9} ms/token  {str(r['tps']):>7} tok/s")

    if base["ms"] and pas["ms"] and act["ms"]:
        print(f"\nmechanism costs   {pas['ms'] - base['ms']:+.1f} ms/token "
              f"({100*(pas['ms']-base['ms'])/base['ms']:+.1f}%)  <- VRAM + graph split + readback")
        print(f"cache is worth    {act['ms'] - pas['ms']:+.1f} ms/token "
              f"({100*(act['ms']-pas['ms'])/base['ms']:+.1f}%)  <- the actual caching")
        print(f"net               {act['ms'] - base['ms']:+.1f} ms/token "
              f"({100*(act['ms']-base['ms'])/base['ms']:+.1f}%)")

    return [base, pas, act]


@app.function(
    image=image,
    volumes={"/models": models, "/results": results},
    gpu="A10G",          # 24 GB, sm_86 - the same class as an RTX 3090, which is the point
    # 16 cores, not the 8 the earlier runs used. BELLS wins by moving work off the CPU, so a
    # starved CPU inflates every result: the 5.88x on Mixtral and 2.80x on Qwen3-Next were
    # measured against a baseline no desktop would have. A 3090 sits next to a real CPU, and
    # the whole point of these runs is a number a 3090 owner can believe.
    cpu=16.0,
    memory=131072,       # 128 GB, so even a 100 GB model stays resident and disk never enters
    timeout=7200,
)
def bench(model: str, n_gen: int = 128, corpus_chunks: int = 1, slots: str = "16,32"):
    """The decisive comparison: BELLS vs plain -ngl vs --cpu-moe at the same VRAM."""
    import json
    import os
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    print(f"cpu count: {os.cpu_count()}")

    path = f"/models/{model}"
    corpus = "/results/corpus.txt"

    if not os.path.exists(corpus):
        # any large-ish text works; the routing statistics are what matter
        subprocess.run(
            ["bash", "-c", f"cat /llama/docs/*.md /llama/*.md > {corpus} 2>/dev/null || true"],
            check=False,
        )
    print(f"corpus {os.path.getsize(corpus)/1e3:.0f} KB")

    common = ["-m", path, "-f", corpus, "--chunks", str(corpus_chunks),
              "-c", "512", "-n", str(n_gen)]

    out = []

    # 1. experts on CPU, the baseline we beat locally
    out.append(_run(common + ["-o", "/tmp/a.json", "-ngl", "99", "--cpu-moe"], "cpu-moe baseline"))

    # 2. plain layer offload, no BELLS. the real competitor at this VRAM
    out.append(_run(common + ["-o", "/tmp/b.json", "-ngl", "99"], "plain -ngl (all layers)"))

    # 3. BELLS, exact output, across the requested cache sizes
    for s in [int(x) for x in slots.split(",") if x.strip()]:
        out.append(_run(
            common + ["-o", f"/tmp/c{s}.json", "-ngl", "99", "--cpu-moe",
                      "--bells-slots", str(s)],
            f"BELLS exact, {s} slots"))

    with open("/results/bench.json", "w") as f:
        json.dump(out, f, indent=2)
    results.commit()

    print("\n\n===== SUMMARY =====")
    for r in out:
        print(f"{r['tag']:<28} {str(r['ms']):>10} ms/token  {str(r['tps']):>8} tok/s")

    return out


@app.function(
    image=image,
    volumes={"/models": models, "/results": results},
    gpu="A10G",
    cpu=16.0,
    memory=131072,
    timeout=7200,
)
def stream_ab(model: str, slots: str = "16", n_gen: int = 96, passes: int = 3):
    """Does a second CUDA stream help now that the source is pinned?

    It was built and measured before and changed nothing: copies came out of pageable memory,
    where cudaMemcpyAsync blocks the caller for 99.7% of the transfer, so the host never got far
    enough ahead for stream ordering to matter. That premise is gone. Pinned copies return to the
    caller in 4.7% of the transfer time, so there is finally something for a second stream to
    overlap - and the 2.25x on the 235B suggests some overlap is already happening implicitly.

    Everything here runs pinned; the only variable is BELLS_COPY_STREAM.
    """
    import json
    import os
    import subprocess
    import statistics as st

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    path = f"/models/{model}"
    corpus = "/results/corpus.txt"
    if not os.path.exists(corpus):
        subprocess.run(
            ["bash", "-c", f"cat /llama/docs/*.md /llama/*.md > {corpus} 2>/dev/null || true"],
            check=False,
        )

    common = ["-m", path, "-f", corpus, "--chunks", "1", "-c", "512", "-n", str(n_gen),
              "-ngl", "99", "--cpu-moe-pinned"]
    out = []

    for s in [int(x) for x in slots.split(",") if x.strip()]:
        os.environ.pop("BELLS_COPY_STREAM", None)
        _run(common + ["-o", "/tmp/sw.json", "--bells-slots", str(s)], f"warmup {s} slots")

        for p in range(1, passes + 1):
            os.environ.pop("BELLS_COPY_STREAM", None)
            out.append(_run(common + ["-o", f"/tmp/s1_{s}_{p}.json", "--bells-slots", str(s)],
                            f"{s} slots one stream p{p}"))

            os.environ["BELLS_COPY_STREAM"] = "1"
            out.append(_run(common + ["-o", f"/tmp/s2_{s}_{p}.json", "--bells-slots", str(s)],
                            f"{s} slots TWO streams p{p}"))
            os.environ.pop("BELLS_COPY_STREAM", None)

    with open("/results/stream_ab.json", "w") as f:
        json.dump(out, f, indent=2)
    results.commit()

    print("\n\n===== SUMMARY =====")
    for s in [int(x) for x in slots.split(",") if x.strip()]:
        one = [r["ms"] for r in out if r["tag"].startswith(f"{s} slots one") and r["ms"]]
        two = [r["ms"] for r in out if r["tag"].startswith(f"{s} slots TWO") and r["ms"]]
        if one and two:
            a, b = st.mean(one), st.mean(two)
            print(f"{s:>4} slots   one stream {a:7.2f} ms   two streams {b:7.2f} ms   {a/b:.3f}x")
        else:
            print(f"{s:>4} slots   INCOMPLETE one={one} two={two}")

    return out


@app.function(
    image=image,
    volumes={"/models": models, "/results": results},
    gpu="A10G",
    cpu=16.0,
    memory=131072,       # the point of running this here: locally a 944 MiB tail would not pin
    timeout=7200,
)
def pinned_ab(model: str, slots: str = "48", n_gen: int = 128, passes: int = 3,
              ctx: int = 512, baseline: bool = False):
    """Pageable vs pinned expert source, alternating inside one session.

    Locally this is worth 1.59x on Qwen3-30B at 17 slots and 1.15x on Qwen3-Next-80B at 48,
    the difference being how much each configuration was transferring - a pageable
    cudaMemcpyAsync blocks the caller for 99.7% of the transfer, so removing the stall pays in
    proportion to the miss rate. See PINNED.md.

    Two things only a big machine answers. Locally a 944 MiB tail always failed to pin, so the
    measured figure is 94-96% of the technique rather than all of it. And 11.3 MB experts have
    never been tried pinned at all: the isolated H2D probe says block size matters a lot
    (1.44x at 1.09 MB against 1.12x at 2.92 MB), which predicts the 235B gains least. If that
    holds it confirms the model; if it does not, the model is wrong somewhere.

    Configurations alternate within a pass because absolute throughput drifts 30%+ over tens of
    minutes on any machine - see the methodology warning in RESULTS.md. Only paired ratios mean
    anything.
    """
    import json
    import os
    import subprocess
    import statistics as st

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    print(f"cpu count: {os.cpu_count()}, "
          f"RAM {os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9:.0f} GB")

    path = f"/models/{model}"
    corpus = "/results/corpus.txt"
    if not os.path.exists(corpus):
        subprocess.run(
            ["bash", "-c", f"cat /llama/docs/*.md /llama/*.md > {corpus} 2>/dev/null || true"],
            check=False,
        )

    common = ["-m", path, "-f", corpus, "--chunks", "1", "-c", str(ctx), "-n", str(n_gen),
              "-ngl", "99"]
    out = []

    # DeepSeek V4 sizes its compressed attention cache from kv_size, so a small -c can make the
    # graph assert before anything runs. A no-BELLS run first separates "the model does not work
    # here" from "BELLS breaks it".
    if baseline:
        out.append(_run(common + ["-o", "/tmp/base.json", "--cpu-moe"], "baseline, no BELLS"))

    for s in [int(x) for x in slots.split(",") if x.strip()]:
        # one throwaway first: the page cache is cold on a fresh volume mount, and a cold
        # first run is what produced a fake 6.70x earlier in this project
        _run(common + ["-o", "/tmp/w.json", "--cpu-moe", "--bells-slots", str(s)],
             f"warmup {s} slots")

        for p in range(1, passes + 1):
            out.append(_run(common + ["-o", f"/tmp/pg{s}_{p}.json", "--cpu-moe",
                                      "--bells-slots", str(s)],
                            f"{s} slots pageable p{p}"))
            out.append(_run(common + ["-o", f"/tmp/pn{s}_{p}.json", "--cpu-moe-pinned",
                                      "--bells-slots", str(s)],
                            f"{s} slots PINNED p{p}"))

    with open("/results/pinned_ab.json", "w") as f:
        json.dump(out, f, indent=2)
    results.commit()

    print("\n\n===== SUMMARY =====")
    for s in [int(x) for x in slots.split(",") if x.strip()]:
        pg = [r["ms"] for r in out if r["tag"].startswith(f"{s} slots pageable") and r["ms"]]
        pn = [r["ms"] for r in out if r["tag"].startswith(f"{s} slots PINNED") and r["ms"]]
        if pg and pn:
            a, b = st.mean(pg), st.mean(pn)
            print(f"{s:>4} slots   pageable {a:7.2f} ms   pinned {b:7.2f} ms   "
                  f"{a/b:.2f}x   (n={len(pg)})")
        else:
            print(f"{s:>4} slots   INCOMPLETE  pageable={pg} pinned={pn}")

    return out
