"""BELLS vs Strata head-to-head comparison on the same Modal A10.

Builds both engines from source, downloads the same model, sets up MTP
for both, and runs identical benchmark prompts through each engine's
OpenAI-compatible API.

Strata's MTP uses its own format (fetched from the BF16 checkpoint and
converted), while BELLS uses the llama.cpp MTP GGUF sidecar.
"""
import modal
import subprocess
import os
import signal
import time
import json
import glob as globmod
import urllib.request
import urllib.error

app = modal.App("bells-vs-strata")

vol = modal.Volume.from_name("bells-benchmark-models", create_if_missing=True)
MODEL_DIR = "/models"

comparison_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "build-essential", "git", "wget", "curl", "ninja-build", "unzip")
    # --- Build BELLS ---
    .run_commands("git clone --branch bells-next https://github.com/DGuckert/llama.cpp-BELLS.git /opt/bells")
    .run_commands("cd /opt/bells && git pull origin bells-next && git log --oneline -3")  # bust:cmake324
    .run_commands(
        "mkdir -p /opt/bells/build && cd /opt/bells/build && "
        "cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='86'"
    )
    .run_commands("cd /opt/bells/build && cmake --build . --target ggml-cuda -j$(nproc)")
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && "
        "cd /opt/bells/build && cmake --build . --target llama-server -j$(nproc)"
    )
    # --- Build Strata ---
    # Strata needs CMake >= 3.24, Ubuntu 22.04 ships 3.22
    .run_commands("pip install cmake --upgrade && cmake --version")
    .run_commands("git clone https://github.com/Niko1221/Strata.git /opt/strata")
    .run_commands("cd /opt/strata && git log --oneline -3")
    # Strata's FetchContent grabs its own pinned llama.cpp/ggml if STRATA_GGML_DIR is unset
    .run_commands(
        "mkdir -p /opt/strata/build && cd /opt/strata/build && "
        "cmake -DCMAKE_BUILD_TYPE=Release "
        "-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF "
        "-DCMAKE_CUDA_ARCHITECTURES='86' "
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
        ".. || echo 'WARNING: Strata cmake configure failed'"
    )
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && "
        "cd /opt/strata/build && cmake --build . --target strata -j$(nproc) || "
        "echo 'WARNING: Strata build failed, will skip Strata benchmarks'"
    )
    .pip_install("huggingface_hub", "numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil")
)


def download_models():
    """Download Flash-Next GGUF model + llama.cpp MTP sidecar."""
    from huggingface_hub import snapshot_download

    model_pattern = f"{MODEL_DIR}/**/*Q2_K_XL*.gguf"
    existing = sorted(globmod.glob(model_pattern, recursive=True))
    if existing:
        total_gb = sum(os.path.getsize(f) for f in existing) / (1024**3)
        print(f"Main model already cached: {len(existing)} shards ({total_gb:.1f} GiB)")
        model_path = existing[0]
    else:
        print("Downloading Qwen3.8-Flash-Next UD-Q2_K_XL (~79 GB, 3 shards)...")
        snapshot_download(
            repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
            allow_patterns=["UD-Q2_K_XL/*"],
            local_dir=MODEL_DIR,
        )
        shards = sorted(globmod.glob(model_pattern, recursive=True))
        if not shards:
            raise RuntimeError("Model download failed")
        model_path = shards[0]

    mtp_pattern = f"{MODEL_DIR}/**/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"
    existing_mtp = sorted(globmod.glob(mtp_pattern, recursive=True))
    if existing_mtp:
        mtp_path = existing_mtp[0]
    else:
        print("Downloading MTP sidecar (standalone Q4_K_M, ~2.8 GB)...")
        snapshot_download(
            repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
            allow_patterns=["MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"],
            local_dir=MODEL_DIR,
        )
        mtp_files = sorted(globmod.glob(mtp_pattern, recursive=True))
        mtp_path = mtp_files[0] if mtp_files else None

    return model_path, mtp_path


def setup_strata_tokenizer(model_path):
    """Extract tokenizer from GGUF for Strata's server.py."""
    tok_dir = "/tmp/strata_tokenizer"
    if os.path.exists(os.path.join(tok_dir, "vocab.json")):
        print("Tokenizer already extracted")
        return tok_dir
    print("Extracting tokenizer from GGUF...")
    r = subprocess.run(
        ["python3", "/opt/strata/tools/strata_tokenizer.py",
         "--gguf", model_path, "--out", "/tmp/strata_pack"],
        capture_output=True, text=True, cwd="/opt/strata"
    )
    if r.returncode != 0:
        print(f"strata_tokenizer.py failed: {r.stderr[-500:]}")
        return None
    tok_dir = "/tmp/strata_pack/tokenizer"
    if os.path.exists(os.path.join(tok_dir, "vocab.json")):
        print(f"Tokenizer extracted to {tok_dir}")
        return tok_dir
    print(f"Tokenizer extraction produced no vocab.json")
    return None


def setup_strata_mtp():
    """Fetch and convert MTP tensors for Strata's draft layer."""
    rt_dir = "/tmp/strata_mtp/rt"
    if os.path.exists(os.path.join(rt_dir, "experts.bin")):
        print("Strata MTP runtime already set up")
        return "/tmp/strata_mtp"

    mtp_dir = "/tmp/strata_mtp"
    os.makedirs(mtp_dir, exist_ok=True)

    print("Fetching MTP tensors from BF16 checkpoint (~5 GB)...")
    r = subprocess.run(
        ["python3", "/opt/strata/tools/mtp_fetch.py", "fetch", "--out", mtp_dir],
        capture_output=True, text=True, cwd="/opt/strata", timeout=1800
    )
    if r.returncode != 0:
        print(f"mtp_fetch.py failed (exit {r.returncode}):")
        print(r.stderr[-1000:] if r.stderr else "no stderr")
        print(r.stdout[-1000:] if r.stdout else "no stdout")
        return None

    print("Packing MTP tensors into GGUF...")
    gguf_path = os.path.join(mtp_dir, "mtp-q2_0.gguf")
    r = subprocess.run(
        ["python3", "/opt/strata/tools/mtp_pack.py",
         "--src", mtp_dir, "--experts", "q2_0", "--out", gguf_path],
        capture_output=True, text=True, cwd="/opt/strata", timeout=600
    )
    if r.returncode != 0:
        print(f"mtp_pack.py failed: {r.stderr[-500:]}")
        return None

    print("Converting MTP to Strata runtime format...")
    os.makedirs(rt_dir, exist_ok=True)
    r = subprocess.run(
        ["python3", "/opt/strata/tools/mtp_rt.py",
         "--gguf", gguf_path, "--out", rt_dir],
        capture_output=True, text=True, cwd="/opt/strata", timeout=600
    )
    if r.returncode != 0:
        print(f"mtp_rt.py failed: {r.stderr[-500:]}")
        return None

    if os.path.exists(os.path.join(rt_dir, "experts.bin")):
        print(f"Strata MTP runtime ready at {rt_dir}")
        return mtp_dir
    print("MTP runtime conversion produced no experts.bin")
    return None


def find_strata_binary():
    """Locate the compiled strata binary."""
    candidates = [
        "/opt/strata/build/strata",
        "/opt/strata/build/bin/strata",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    r = subprocess.run(
        ["find", "/opt/strata/build", "-name", "strata", "-type", "f", "-executable"],
        capture_output=True, text=True
    )
    if r.stdout.strip():
        return r.stdout.strip().split('\n')[0]
    return None


def api_call(method, path, body=None, timeout=120, port=8080):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def wait_healthy(timeout=600, port=8080):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = api_call("GET", "/health", port=port)
            if r.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def chat(prompt, max_tokens=256, port=8080):
    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": f"/no_think {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.time()
    r = api_call("POST", "/v1/chat/completions", body, timeout=600, port=port)
    elapsed = time.time() - t0
    if "error" in r:
        return {"error": r["error"], "elapsed": elapsed}
    choice = r.get("choices", [{}])[0]
    text = choice.get("message", {}).get("content", "")
    usage = r.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tps = comp_tokens / elapsed if elapsed > 0 else 0
    return {
        "text": text[:100],
        "decode_tps": tps,
        "elapsed": elapsed,
        "comp_tokens": comp_tokens,
    }


BENCH_PROMPTS = [
    ("Explain TCP vs UDP in 3 sentences.", 100),
    ("Write a Python quicksort implementation.", 200),
    ("What causes OOM in Kubernetes pods?", 150),
    ("Write a Dockerfile for a Node.js app with multi-stage build.", 200),
    ("Explain database connection pooling briefly.", 100),
    ("Write a bash script to find and compress old log files.", 150),
    ("Compare Redis and Memcached for session storage.", 150),
    ("Explain OAuth 2.0 authorization code flow.", 200),
]


def run_benchmark(engine_name, port=8080, warmup=2, bench_runs=6):
    print(f"\n  Running warmup ({warmup} requests)...")
    for i in range(warmup):
        prompt, max_tok = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
        r = chat(prompt, max_tok, port)
        if "error" in r:
            print(f"    warmup {i+1}: ERROR {r['error']}")
        else:
            print(f"    warmup {i+1}: {r['decode_tps']:.1f} tok/s")

    print(f"  Running benchmark ({bench_runs} requests)...")
    bench_results = []
    for i in range(bench_runs):
        prompt, max_tok = BENCH_PROMPTS[(warmup + i) % len(BENCH_PROMPTS)]
        r = chat(prompt, max_tok, port)
        if "error" not in r:
            bench_results.append(r)
            print(f"    [{i+1}/{bench_runs}] {r['decode_tps']:.1f} tok/s, "
                  f"{r['comp_tokens']} tokens in {r['elapsed']:.1f}s")
        else:
            print(f"    [{i+1}/{bench_runs}] ERROR: {r['error']}")

    results = {"engine": engine_name}
    if bench_results:
        avg_tps = sum(r["decode_tps"] for r in bench_results) / len(bench_results)
        min_tps = min(r["decode_tps"] for r in bench_results)
        max_tps = max(r["decode_tps"] for r in bench_results)
        total_tokens = sum(r["comp_tokens"] for r in bench_results)
        total_time = sum(r["elapsed"] for r in bench_results)
        results.update({
            "avg_tps": round(avg_tps, 1),
            "min_tps": round(min_tps, 1),
            "max_tps": round(max_tps, 1),
            "total_tokens": total_tokens,
            "total_time": round(total_time, 1),
            "effective_tps": round(total_tokens / total_time, 1),
        })
        print(f"\n  RESULT [{engine_name}]: {avg_tps:.1f} tok/s avg "
              f"(min {min_tps:.1f}, max {max_tps:.1f})")
    else:
        results["error"] = "no successful runs"
    return results


def stop_server(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_bells_config(model_path, config_name, server_args):
    """Run a single BELLS configuration."""
    print(f"\n{'='*60}")
    print(f"ENGINE: {config_name}")
    print(f"{'='*60}")

    cmd = ["stdbuf", "-oL", "/opt/bells/build/bin/llama-server",
           "-m", model_path, "--host", "0.0.0.0", "--port", "8080"] + server_args
    print(f"CMD: {' '.join(cmd)}")

    log = open(f"/tmp/bells_{config_name}_stderr.txt", "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log)
    try:
        print("  Loading model...")
        if wait_healthy(timeout=600):
            result = run_benchmark(config_name, port=8080)
        else:
            log.close()
            with open(f"/tmp/bells_{config_name}_stderr.txt") as f:
                print(f"  FAILED. Last output:\n{f.read()[-2000:]}")
            result = {"engine": config_name, "error": "failed to start"}
    finally:
        stop_server(proc)
        log.close()

    with open(f"/tmp/bells_{config_name}_stderr.txt") as f:
        for line in f:
            low = line.lower()
            if any(kw in low for kw in ['bells', 'slots', 'auto', 'hit ', 'spec', 'mtp', 'draft', 'accept']):
                print(f"    {line.strip()}")
    return result


def run_bells(model_path, mtp_path):
    """Run BELLS benchmark configs. Returns list of results."""
    results = []

    # Config 1: BELLS auto baseline (no MTP, full VRAM for expert cache)
    result_auto = run_bells_config(model_path, "BELLS-auto", [
        "--auto", "-c", "8192", "-ctk", "q4_0", "-ctv", "q4_0",
    ])
    results.append(result_auto)
    time.sleep(5)

    # Config 2: BELLS + MTP with explicit 120 slots (leaves ~5 GiB for MTP compute)
    # --auto fills all VRAM, leaving no room for MTP compute buffers (~1 GiB)
    if mtp_path:
        result_mtp = run_bells_config(model_path, "BELLS-120+MTP", [
            "--bells-slots", "120", "--cpu-moe", "--bells-adapt", "32",
            "-c", "8192", "-ctk", "q4_0", "-ctv", "q4_0",
            "--spec-type", "draft-mtp",
            "--spec-draft-model", mtp_path,
            "--spec-draft-ngl", "0",
        ])
        results.append(result_mtp)
        time.sleep(5)

    return results


def run_strata(model_path, strata_mtp_dir, tok_dir):
    """Run Strata benchmark."""
    print("\n" + "="*60)
    print("ENGINE: Strata")
    print("="*60)

    strata_exe = find_strata_binary()
    if not strata_exe:
        print("  ERROR: Strata binary not found")
        return {"engine": "Strata", "error": "binary not found"}
    print(f"  Binary: {strata_exe}")

    strata_args = [
        "--native", model_path,
        "--max-context", "8192",
        "--kv", "q4_0",
        "--prefill", "auto",
    ]
    if strata_mtp_dir:
        rt_dir = os.path.join(strata_mtp_dir, "rt")
        strata_args += ["--mtp", rt_dir]
        print(f"  MTP: {rt_dir}")
    else:
        print("  MTP: disabled (setup failed)")

    cfg = {
        "exe": strata_exe,
        "args": strata_args,
        "cwd": "/opt/strata",
        "log": "/tmp/strata_engine.log",
    }
    config_path = "/tmp/strata-bench.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    if not tok_dir:
        print("  ERROR: Tokenizer not available")
        return {"engine": "Strata", "error": "tokenizer not available"}

    cmd = [
        "python3", "-m", "serve.server",
        "--engine", "strata",
        "--config", config_path,
        "--tokenizer", tok_dir,
        "--host", "0.0.0.0",
        "--port", "8080",
    ]
    print(f"CMD: {' '.join(cmd)}")

    log = open("/tmp/strata_stderr.txt", "w")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/strata/tools:/opt/strata"
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        cwd="/opt/strata", env=env,
    )
    try:
        print("  Loading model...")
        if wait_healthy(timeout=900):
            result = run_benchmark("Strata+MTP" if strata_mtp_dir else "Strata", port=8080)
        else:
            log.close()
            with open("/tmp/strata_stderr.txt") as f:
                tail = f.read()[-3000:]
            print(f"  FAILED. Last output:\n{tail}")
            result = {"engine": "Strata", "error": "failed to start"}
    finally:
        stop_server(proc)
        log.close()

    with open("/tmp/strata_stderr.txt") as f:
        print("  Strata log (last 30 lines):")
        lines = f.readlines()
        for line in lines[-30:]:
            print(f"    {line.rstrip()}")

    return result


@app.function(
    image=comparison_image,
    gpu="A10",
    memory=131072,
    volumes={MODEL_DIR: vol},
    timeout=10800,
)
def compare_engines():
    """Head-to-head: BELLS+MTP vs Strata+MTP on the same A10."""
    model_path, mtp_path = download_models()
    vol.commit()

    print("\n" + "="*70)
    print("BELLS vs STRATA: Head-to-Head on Same A10")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                     "--format=csv"], check=True)
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    print(f"System RAM: {ram:.1f} GiB")
    print("="*70)

    tok_dir = setup_strata_tokenizer(model_path)
    strata_mtp_dir = setup_strata_mtp()

    all_results = []

    bells_results = run_bells(model_path, mtp_path)
    all_results.extend(bells_results)
    time.sleep(5)

    strata_result = run_strata(model_path, strata_mtp_dir, tok_dir)
    all_results.append(strata_result)

    print("\n" + "="*70)
    print("HEAD-TO-HEAD COMPARISON RESULTS")
    print("="*70)
    print(f"{'Engine':<25} {'Avg tok/s':>10} {'Min':>8} {'Max':>8} {'Effective':>10}")
    print("-"*61)
    for r in all_results:
        if r.get("error"):
            print(f"{r['engine']:<25} {'FAILED':>10}   {r.get('error', '')}")
        else:
            print(f"{r['engine']:<25} {r.get('avg_tps','?'):>10} "
                  f"{r.get('min_tps','?'):>8} {r.get('max_tps','?'):>8} "
                  f"{r.get('effective_tps','?'):>10}")

    # Find best BELLS result
    best_bells = max(
        (r for r in all_results if r.get("engine", "").startswith("BELLS") and r.get("avg_tps")),
        key=lambda r: r["avg_tps"], default=None
    )
    strata = next((r for r in all_results if r.get("engine", "").startswith("Strata")), None)

    if best_bells and strata and strata.get("avg_tps"):
        ratio = best_bells["avg_tps"] / strata["avg_tps"]
        diff = best_bells["avg_tps"] - strata["avg_tps"]
        winner = "BELLS" if diff > 0 else "Strata"
        print(f"\n  Best BELLS config: {best_bells['engine']} ({best_bells['avg_tps']} tok/s)")
        print(f"  Strata: {strata['engine']} ({strata['avg_tps']} tok/s)")
        print(f"  {winner} wins by {abs(diff):.1f} tok/s ({ratio:.2f}x)")
    elif best_bells and (not strata or strata.get("error")):
        print(f"\n  Best BELLS: {best_bells['engine']} ({best_bells['avg_tps']} tok/s)")
        print(f"  Strata: FAILED to produce results")

    return json.dumps(all_results, indent=2)


@app.local_entrypoint()
def main():
    print("BELLS vs Strata: Head-to-Head on Same A10")
    print("=" * 60)
    result = compare_engines.remote()
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(result)
