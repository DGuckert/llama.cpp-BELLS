"""BELLS + MTP speculative decoding benchmark on Modal A10G.

Tests BELLS with and without MTP (Multi-Token Prediction) speculative
decoding on Flash-Next 177B. MTP uses the model's built-in draft layer
to predict 3 tokens per forward pass (~2.4-3.2x decode speedup).
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

app = modal.App("bells-mtp-benchmark")

vol = modal.Volume.from_name("bells-benchmark-models", create_if_missing=True)
MODEL_DIR = "/models"

bells_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "build-essential", "git", "wget", "curl")
    .run_commands("git clone --branch bells-next https://github.com/DGuckert/llama.cpp-BELLS.git /opt/bells")
    .run_commands("cd /opt/bells && git pull origin bells-next && git log --oneline -3")  # bust:bells-reserve
    .run_commands(
        "mkdir -p /opt/bells/build && cd /opt/bells/build && "
        "cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='86'"
    )
    .run_commands("cd /opt/bells/build && cmake --build . --target ggml-cuda -j$(nproc)")
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && "
        "cd /opt/bells/build && cmake --build . --target llama-server -j$(nproc)"
    )
    .pip_install("huggingface_hub")
)


def download_models():
    """Download Flash-Next GGUF model + MTP sidecar. Returns (model_path, mtp_path)."""
    from huggingface_hub import snapshot_download

    # Main model
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

    # MTP sidecar - use standalone (non-shared) Q4_K_M which includes token_embd.
    # The "shared" variant requires nextn_shared_target_tensors support not in our fork.
    mtp_pattern = f"{MODEL_DIR}/**/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"
    existing_mtp = sorted(globmod.glob(mtp_pattern, recursive=True))
    if existing_mtp:
        print(f"MTP sidecar already cached: {existing_mtp[0]}")
        mtp_path = existing_mtp[0]
    else:
        print("Downloading MTP sidecar (standalone Q4_K_M, ~2.8 GB)...")
        snapshot_download(
            repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
            allow_patterns=["MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"],
            local_dir=MODEL_DIR,
        )
        mtp_files = sorted(globmod.glob(mtp_pattern, recursive=True))
        if not mtp_files:
            ls = subprocess.run(["find", MODEL_DIR, "-name", "mtp*"], capture_output=True, text=True)
            print(f"MTP files found:\n{ls.stdout}")
            mtp_files = sorted(globmod.glob(f"{MODEL_DIR}/**/mtp*.gguf", recursive=True))
        if not mtp_files:
            print("WARNING: MTP sidecar not found, will skip MTP configs")
            mtp_path = None
        else:
            mtp_path = mtp_files[0]

    return model_path, mtp_path


BASE = "http://127.0.0.1:8080"


def api(method, path, body=None, timeout=120):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def wait_healthy(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = api("GET", "/health")
            if r.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def chat(prompt, max_tokens=256):
    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": f"/no_think {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.time()
    r = api("POST", "/v1/chat/completions", body, timeout=600)
    elapsed = time.time() - t0
    if "error" in r:
        return {"error": r["error"], "elapsed": elapsed}
    choice = r.get("choices", [{}])[0]
    text = choice.get("message", {}).get("content", "")
    usage = r.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tps = comp_tokens / elapsed if elapsed > 0 else 0
    return {
        "text": text[:100],
        "decode_tps": tps,
        "elapsed": elapsed,
        "comp_tokens": comp_tokens,
        "prompt_tokens": prompt_tokens,
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


def run_config(model_path, config_name, server_args, warmup=2, bench_runs=6):
    stderr_path = f"/tmp/server_{config_name}_stderr.txt"
    cmd = ["stdbuf", "-oL", "/opt/bells/build/bin/llama-server",
           "-m", model_path, "--host", "0.0.0.0", "--port", "8080"] + server_args

    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    ferr = open(stderr_path, 'w')
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ferr)
    results = {"config": config_name, "args": server_args}

    try:
        print(f"  Loading model...")
        if not wait_healthy(timeout=600):
            ferr.close()
            with open(stderr_path) as f:
                tail = f.read()[-2000:]
            print(f"  FAILED to start. Last output:\n{tail}")
            results["error"] = "server failed to start"
            return results

        print(f"  Server healthy. Running warmup ({warmup} requests)...")
        for i in range(warmup):
            prompt, max_tok = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
            r = chat(prompt, max_tok)
            if "error" in r:
                print(f"    warmup {i+1}: ERROR {r['error']}")
            else:
                print(f"    warmup {i+1}: {r['decode_tps']:.1f} tok/s")

        print(f"  Running benchmark ({bench_runs} requests)...")
        bench_results = []
        for i in range(bench_runs):
            prompt, max_tok = BENCH_PROMPTS[(warmup + i) % len(BENCH_PROMPTS)]
            r = chat(prompt, max_tok)
            if "error" not in r:
                bench_results.append(r)
                print(f"    [{i+1}/{bench_runs}] {r['decode_tps']:.1f} tok/s, "
                      f"{r['comp_tokens']} tokens in {r['elapsed']:.1f}s")
            else:
                print(f"    [{i+1}/{bench_runs}] ERROR: {r['error']}")

        if bench_results:
            avg_tps = sum(r["decode_tps"] for r in bench_results) / len(bench_results)
            min_tps = min(r["decode_tps"] for r in bench_results)
            max_tps = max(r["decode_tps"] for r in bench_results)
            total_tokens = sum(r["comp_tokens"] for r in bench_results)
            total_time = sum(r["elapsed"] for r in bench_results)
            results["avg_tps"] = round(avg_tps, 1)
            results["min_tps"] = round(min_tps, 1)
            results["max_tps"] = round(max_tps, 1)
            results["total_tokens"] = total_tokens
            results["total_time"] = round(total_time, 1)
            results["effective_tps"] = round(total_tokens / total_time, 1)
            print(f"\n  RESULT: {avg_tps:.1f} tok/s avg (min {min_tps:.1f}, max {max_tps:.1f})")
            print(f"  Total: {total_tokens} tokens in {total_time:.1f}s = {total_tokens/total_time:.1f} effective tok/s")

    finally:
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
        ferr.close()

        with open(stderr_path) as f:
            stderr = f.read()
        bells_lines = []
        for line in stderr.split('\n'):
            low = line.lower()
            if any(kw in low for kw in [
                'bells', 'slots', 'vram', 'hit ', 'expert', 'auto',
                'cache', 'cpu-moe', 'split', 'thread', 'spec', 'mtp',
                'draft', 'accept',
            ]):
                bells_lines.append(line.strip())
        if bells_lines:
            results["server_log"] = bells_lines[-30:]
            print(f"\n  Server log (key lines):")
            for l in bells_lines[-20:]:
                print(f"    {l}")

    return results


@app.function(
    image=bells_image,
    gpu="A10",
    memory=131072,
    volumes={MODEL_DIR: vol},
    timeout=7200,
)
def benchmark_mtp():
    """Test BELLS with and without MTP on A10G."""
    model_path, mtp_path = download_models()
    vol.commit()

    print("\n" + "="*60)
    print("BELLS + MTP Benchmark: Flash-Next 177B on A10G")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                     "--format=csv"], check=True)
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    print(f"System RAM: {ram:.1f} GiB")
    if mtp_path:
        print(f"MTP sidecar: {mtp_path}")
        mtp_size = os.path.getsize(mtp_path) / (1024**3)
        print(f"MTP size: {mtp_size:.2f} GiB")
    print("="*60)

    configs = [
        # Baseline: BELLS auto (now includes --bells-adapt 32 by default)
        ("bells-auto", [
            "--auto", "-c", "8192", "-ctk", "q4_0", "-ctv", "q4_0",
        ]),
    ]

    if mtp_path:
        configs.extend([
            # BELLS auto + MTP (--bells-reserve auto-detected from --spec-draft-model)
            ("bells-auto-mtp", [
                "--auto", "-c", "8192", "-ctk", "q4_0", "-ctv", "q4_0",
                "--spec-type", "draft-mtp",
                "--spec-draft-model", mtp_path,
                "--spec-draft-ngl", "0",
            ]),
            # BELLS auto + MTP + smaller context
            ("bells-auto-mtp-c4096", [
                "--auto", "-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0",
                "--spec-type", "draft-mtp",
                "--spec-draft-model", mtp_path,
                "--spec-draft-ngl", "0",
            ]),
            # BELLS auto + MTP + q8_0 cache type
            ("bells-auto-mtp-q8cache", [
                "--auto", "-c", "8192", "-ctk", "q4_0", "-ctv", "q4_0",
                "--bells-cache-type", "q8_0",
                "--spec-type", "draft-mtp",
                "--spec-draft-model", mtp_path,
                "--spec-draft-ngl", "0",
            ]),
        ])

    all_results = []
    for name, args in configs:
        result = run_config(model_path, name, args)
        all_results.append(result)
        time.sleep(5)

    print("\n" + "="*60)
    print("BELLS + MTP BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Config':<30} {'Avg tok/s':>10} {'Min':>8} {'Max':>8} {'Effective':>10}")
    print("-"*66)
    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<30} {'FAILED':>10}")
        else:
            print(f"{r['config']:<30} {r.get('avg_tps','?'):>10} "
                  f"{r.get('min_tps','?'):>8} {r.get('max_tps','?'):>8} "
                  f"{r.get('effective_tps','?'):>10}")

    return json.dumps(all_results, indent=2)


@app.local_entrypoint()
def main():
    print("BELLS + MTP Benchmark: Flash-Next 177B on A10G")
    print("=" * 60)
    result = benchmark_mtp.remote()
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(result)
