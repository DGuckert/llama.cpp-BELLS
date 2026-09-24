"""BELLS benchmark engine — explores flag combinations to find optimal config."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import system


@dataclass
class BenchConfig:
    model: str
    context: int
    ngl: int = 99
    threads: int = 0
    extra_flags: list[str] = field(default_factory=list)
    label: str = ""

    def to_args(self) -> list[str]:
        args = ["-m", self.model, "-ngl", str(self.ngl), "-c", str(self.context),
                "-n", "80", "-p", "Explain general relativity in detail",
                "--no-display-prompt", "--simple-io"]
        if self.threads > 0:
            args += ["-t", str(self.threads)]
        args += self.extra_flags
        return args

    def short_label(self) -> str:
        if self.label:
            return self.label
        flags = " ".join(self.extra_flags) if self.extra_flags else "baseline"
        return f"c{self.context} {flags}"


@dataclass
class BenchResult:
    config: BenchConfig
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    load_time_ms: float = 0.0
    oom: bool = False
    error: str = ""
    exit_code: int = 0
    duration_s: float = 0.0

    def score(self) -> float:
        if self.oom or self.error or self.gen_tps == 0:
            return 0.0
        return self.gen_tps

    def to_dict(self) -> dict:
        return {
            "label": self.config.short_label(),
            "flags": self.config.extra_flags,
            "context": self.config.context,
            "prompt_tps": round(self.prompt_tps, 2),
            "gen_tps": round(self.gen_tps, 2),
            "load_time_ms": round(self.load_time_ms, 1),
            "oom": self.oom,
            "error": self.error,
            "duration_s": round(self.duration_s, 1),
        }


def _parse_output(stderr: str) -> dict:
    """Extract timing info from llama-cli stderr."""
    result = {"prompt_tps": 0.0, "gen_tps": 0.0, "load_time_ms": 0.0}
    for line in stderr.splitlines():
        m = re.search(r"llama_perf_context_print:\s+eval time\s+=\s+[\d.]+\s+ms\s+/\s+\d+\s+tokens\s+\(\s*([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second\)", line)
        if m:
            result["gen_tps"] = float(m.group(2))
        m = re.search(r"llama_perf_context_print:\s+prompt eval time\s+=\s+[\d.]+\s+ms\s+/\s+\d+\s+tokens\s+\(\s*([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second\)", line)
        if m:
            result["prompt_tps"] = float(m.group(2))
        m = re.search(r"load time\s+=\s+([\d.]+)\s+ms", line)
        if m:
            result["load_time_ms"] = float(m.group(1))
    return result


def _is_oom(stderr: str, exit_code: int) -> bool:
    lower = stderr.lower()
    return (exit_code != 0 and
            any(x in lower for x in ["out of memory", "oom", "alloc failed",
                                      "cudamalloc failed", "vk_error_out_of_device_memory",
                                      "failed to allocate"]))


def run_single(cli_path: str, config: BenchConfig, timeout: int = 600) -> BenchResult:
    """Run one benchmark configuration."""
    args = [cli_path] + config.to_args()
    result = BenchResult(config=config)
    t0 = time.time()

    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GGML_CUDA_NO_PINNED": "1"},
        )
        result.exit_code = proc.returncode
        result.duration_s = time.time() - t0
        stderr = proc.stderr or ""

        if _is_oom(stderr, proc.returncode):
            result.oom = True
            result.error = "OOM"
            return result

        if proc.returncode != 0:
            result.error = f"exit {proc.returncode}"
            for line in stderr.splitlines()[-5:]:
                if line.strip():
                    result.error = line.strip()
                    break
            return result

        parsed = _parse_output(stderr)
        result.prompt_tps = parsed["prompt_tps"]
        result.gen_tps = parsed["gen_tps"]
        result.load_time_ms = parsed["load_time_ms"]

    except subprocess.TimeoutExpired:
        result.error = "timeout"
        result.duration_s = time.time() - t0
    except Exception as e:
        result.error = str(e)
        result.duration_s = time.time() - t0

    return result


def _detect_model_arch(model_path: str, cli_path: str) -> dict:
    """Detect n_expert and n_expert_used from the GGUF file header."""
    info = {"n_expert": 0, "n_expert_used": 0, "n_layer": 0}

    # Fast path: read GGUF metadata directly (no model load needed)
    try:
        info = _read_gguf_metadata(model_path)
        if info["n_layer"] > 0:
            return info
    except Exception:
        pass

    # Fallback: run llama-cli briefly with GPU offload to read the log
    try:
        proc = subprocess.Popen(
            [cli_path, "-m", model_path, "-n", "0", "-p", "x", "--no-display-prompt",
             "--simple-io", "-ngl", "99", "-c", "32"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={**os.environ, "GGML_CUDA_NO_PINNED": "1"},
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            stdout, stderr = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        output = (stderr or "") + (stdout or "")
        for line in output.splitlines():
            m = re.search(r"n_expert\s*=\s*(\d+)", line)
            if m:
                info["n_expert"] = int(m.group(1))
            m = re.search(r"n_expert_used\s*=\s*(\d+)", line)
            if m:
                info["n_expert_used"] = int(m.group(1))
            m = re.search(r"n_layer\s*=\s*(\d+)", line)
            if m:
                info["n_layer"] = int(m.group(1))
    except Exception:
        pass
    return info


def _read_gguf_metadata(model_path: str) -> dict:
    """Read n_expert, n_expert_used, n_layer from GGUF file header. No model load."""
    import struct
    info = {"n_expert": 0, "n_expert_used": 0, "n_layer": 0}
    key_map = {}
    for arch in ("llama", "qwen2moe", "qwen3moe", "qwen35moe", "deepseek2",
                 "mixtral", "arctic", "grok", "jamba", "dbrx"):
        key_map[f"{arch}.expert_count"] = "n_expert"
        key_map[f"{arch}.expert_used_count"] = "n_expert_used"
        key_map[f"{arch}.block_count"] = "n_layer"

    with open(model_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return info
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            return info
        _n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(min(n_kv, 200)):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8", errors="replace")
            val_type = struct.unpack("<I", f.read(4))[0]
            val = _read_gguf_value(f, val_type)

            if key in key_map and isinstance(val, int):
                info[key_map[key]] = val

            if info["n_layer"] > 0 and info["n_expert"] > 0 and info["n_expert_used"] > 0:
                break

    return info


def _read_gguf_value(f, val_type: int):
    """Read a single GGUF metadata value."""
    import struct
    if val_type == 0:    # UINT8
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == 1:  # INT8
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == 2:  # UINT16
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == 3:  # INT16
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == 4:  # UINT32
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == 5:  # INT32
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == 6:  # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == 7:  # BOOL
        return struct.unpack("<?", f.read(1))[0]
    elif val_type == 8:  # STRING
        slen = struct.unpack("<Q", f.read(8))[0]
        return f.read(slen).decode("utf-8", errors="replace")
    elif val_type == 9:  # ARRAY
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
    elif val_type == 10:  # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == 11:  # INT64
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == 12:  # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {val_type}")


@dataclass
class BenchProfile:
    name: str  # "short", "medium", "long"
    max_minutes: int
    ot_steps: list[int] = field(default_factory=list)
    bells_slots: list[int] = field(default_factory=list)
    try_refresh: bool = False
    try_split: bool = False
    try_eu_override: bool = False
    try_fa: bool = True
    try_kv_quant: bool = False


PROFILES = {
    "short": BenchProfile(
        name="short", max_minutes=60,
        ot_steps=[0, 4, 8],
        bells_slots=[0, 20, 40],
        try_fa=True,
    ),
    "medium": BenchProfile(
        name="medium", max_minutes=240,
        ot_steps=[0, 2, 4, 6, 8, 12],
        bells_slots=[0, 15, 25, 33, 40, 50, 60, 80],
        try_refresh=True,
        try_eu_override=True,
        try_fa=True,
        try_kv_quant=True,
    ),
    "long": BenchProfile(
        name="long", max_minutes=480,
        ot_steps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16],
        bells_slots=[0, 10, 15, 20, 25, 30, 33, 40, 48, 56, 64, 80, 96, 128],
        try_refresh=True,
        try_split=True,
        try_eu_override=True,
        try_fa=True,
        try_kv_quant=True,
    ),
}


def generate_configs(model: str, context: int, profile: BenchProfile,
                     arch_info: dict, sys_info: system.SystemInfo) -> list[BenchConfig]:
    """Generate benchmark configurations in priority order."""
    configs = []
    n_expert = arch_info.get("n_expert", 0)
    n_layer = arch_info.get("n_layer", 0)
    n_expert_used = arch_info.get("n_expert_used", 0)
    is_moe = n_expert > 1
    threads = min(sys_info.cpu_cores, 8)

    base = ["-fa", "on", "-t", str(threads)]

    configs.append(BenchConfig(model=model, context=context, threads=threads,
                               extra_flags=base.copy(), label="baseline"))

    if not is_moe:
        for ot in profile.ot_steps:
            if ot == 0 or ot > n_layer:
                continue
            pattern = "blk\\.(" + "|".join(str(i) for i in range(ot)) + ")\\..*=CUDA0"
            configs.append(BenchConfig(
                model=model, context=context, threads=threads,
                extra_flags=["-ot", pattern] + base,
                label=f"-ot {ot} layers",
            ))
        return configs

    configs.append(BenchConfig(
        model=model, context=context, threads=threads,
        extra_flags=["--cpu-moe-pinned"] + base,
        label="cpu-moe-pinned",
    ))
    configs.append(BenchConfig(
        model=model, context=context, threads=threads,
        extra_flags=["--cpu-moe"] + base,
        label="cpu-moe (mmap)",
    ))

    for ot in profile.ot_steps:
        if ot == 0 or ot > n_layer:
            continue
        pattern = "blk\\.(" + "|".join(str(i) for i in range(ot)) + ")\\.ffn_.*_exps=CUDA0"
        configs.append(BenchConfig(
            model=model, context=context, threads=threads,
            extra_flags=["-ot", pattern, "--cpu-moe-pinned"] + base,
            label=f"-ot {ot} experts",
        ))

    for slots in profile.bells_slots:
        if slots == 0:
            continue
        configs.append(BenchConfig(
            model=model, context=context, threads=threads,
            extra_flags=["--cpu-moe-pinned", "--bells-slots", str(slots)] + base,
            label=f"bells {slots} slots",
        ))

    if profile.try_refresh:
        for r in [2, 4, 8]:
            configs.append(BenchConfig(
                model=model, context=context, threads=threads,
                extra_flags=["--cpu-moe-pinned", "--bells-slots", "40", "--bells-refresh", str(r)] + base,
                label=f"bells 40 refresh {r}",
            ))

    if profile.try_eu_override and n_expert_used > 2:
        for eu in [max(2, n_expert_used - 2), max(2, n_expert_used - 4)]:
            if eu >= n_expert_used:
                continue
            configs.append(BenchConfig(
                model=model, context=context, threads=threads,
                extra_flags=["--cpu-moe-pinned", "--override-kv",
                             f"general.expert_used_count=int:{eu}"] + base,
                label=f"eu {eu} (was {n_expert_used})",
            ))

    if profile.try_kv_quant:
        configs.append(BenchConfig(
            model=model, context=context, threads=threads,
            extra_flags=["--cpu-moe-pinned", "-ctk", "q8_0", "-ctv", "q8_0"] + base,
            label="kv q8_0",
        ))
        configs.append(BenchConfig(
            model=model, context=context, threads=threads,
            extra_flags=["--cpu-moe-pinned", "-ctk", "q4_0", "-ctv", "q4_0"] + base,
            label="kv q4_0",
        ))

    return configs


class BenchRunner:
    def __init__(self, cli_path: str, model: str, context: int,
                 profile_name: str = "short", output_dir: Optional[str] = None,
                 on_progress=None):
        self.cli_path = cli_path
        self.model = model
        self.context = context
        self.profile = PROFILES[profile_name]
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "bench-results"
        self.on_progress = on_progress
        self.results: list[BenchResult] = []
        self.best: Optional[BenchResult] = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _emit(self, msg: str):
        if self.on_progress:
            clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', msg)
            self.on_progress(clean)

    def run(self) -> list[BenchResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sys_info = system.detect()
        self._emit(f"Detecting model architecture...")
        arch_info = _detect_model_arch(self.model, self.cli_path)
        self._emit(f"  n_expert={arch_info['n_expert']} n_expert_used={arch_info['n_expert_used']} "
                   f"n_layer={arch_info['n_layer']}")

        configs = generate_configs(self.model, self.context, self.profile, arch_info, sys_info)
        self._emit(f"Generated {len(configs)} configurations ({self.profile.name} profile, "
                   f"max {self.profile.max_minutes} min)")

        deadline = time.time() + self.profile.max_minutes * 60
        oom_flags: set[str] = set()

        for i, cfg in enumerate(configs):
            if self._cancelled:
                self._emit("Benchmark cancelled.")
                break

            if time.time() > deadline:
                self._emit(f"Time limit reached after {i} configs.")
                break

            flag_key = " ".join(sorted(cfg.extra_flags))
            skip = False
            for oom_key in oom_flags:
                if oom_key in flag_key:
                    skip = True
                    break
            if skip:
                self._emit(f"[{i+1}/{len(configs)}] SKIP {cfg.short_label()} (prior OOM)")
                continue

            remaining = int((deadline - time.time()) / 60)
            self._emit(f"[{i+1}/{len(configs)}] {cfg.short_label()} ({remaining} min remaining)")

            ctx_factor = max(1, cfg.context // 8192)
            per_run_timeout = min(300 * ctx_factor, max(120 * ctx_factor, int((deadline - time.time()) * 0.8)))
            result = run_single(self.cli_path, cfg, timeout=per_run_timeout)
            self.results.append(result)

            if result.oom:
                self._emit(f"  OOM — skipping similar configs")
                for f in cfg.extra_flags:
                    if f.startswith("--bells-slots"):
                        oom_flags.add(f)
                    elif f.startswith("-ot"):
                        oom_flags.add(f)
                continue

            if result.error:
                self._emit(f"  ERROR: {result.error}")

            if result.gen_tps == 0 and not result.oom and not result.error:
                self._emit(f"  WARNING: no throughput data captured (gen_tps=0)")
                result.error = "no perf data"

            self._emit(f"  {result.gen_tps:.1f} tok/s gen, {result.prompt_tps:.1f} tok/s prompt "
                       f"({result.duration_s:.0f}s)")

            if self.best is None or result.score() > self.best.score():
                self.best = result
                self._emit(f"  ** NEW BEST: {result.gen_tps:.1f} tok/s **")

        self._save_results()
        return self.results

    def _save_results(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_name = Path(self.model).stem
        out_file = self.output_dir / f"bench_{model_name}_{self.profile.name}_{ts}.json"

        report = {
            "model": self.model,
            "context": self.context,
            "profile": self.profile.name,
            "timestamp": ts,
            "results": [r.to_dict() for r in self.results],
            "best": self.best.to_dict() if self.best else None,
        }

        out_file.write_text(json.dumps(report, indent=2))
        self._emit(f"\nResults saved to {out_file}")

        if self.best:
            self._emit(f"\n{'='*60}")
            self._emit(f"BEST CONFIG: {self.best.config.short_label()}")
            self._emit(f"  Generation: {self.best.gen_tps:.1f} tok/s")
            self._emit(f"  Prefill:    {self.best.prompt_tps:.1f} tok/s")
            self._emit(f"  Flags:      {' '.join(self.best.config.extra_flags)}")
            self._emit(f"{'='*60}")

    def summary_text(self) -> str:
        lines = []
        lines.append(f"BELLS Benchmark — {self.profile.name} profile")
        lines.append(f"Model: {Path(self.model).name}")
        lines.append(f"Context: {self.context}")
        lines.append("")
        lines.append(f"{'Config':<40} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Status':>10}")
        lines.append("-" * 75)
        for r in sorted(self.results, key=lambda x: -x.score()):
            status = "OOM" if r.oom else r.error if r.error else "OK"
            lines.append(f"{r.config.short_label():<40} {r.gen_tps:>10.1f} {r.prompt_tps:>12.1f} {status:>10}")
        if self.best:
            lines.append("")
            lines.append(f"BEST: {self.best.config.short_label()} — {self.best.gen_tps:.1f} tok/s")
            lines.append(f"FLAGS: {' '.join(self.best.config.extra_flags)}")
        return "\n".join(lines)
