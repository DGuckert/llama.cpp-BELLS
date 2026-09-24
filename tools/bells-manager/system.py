"""Detect GPUs, RAM, CPU and available llama.cpp binaries."""

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GpuInfo:
    name: str
    vram_mb: int
    vram_free_mb: int
    vendor: str  # "nvidia", "amd", "intel"
    index: int = 0
    driver: str = ""
    compute: str = ""  # CUDA version, Vulkan version, etc.


@dataclass
class SystemInfo:
    os: str
    arch: str
    cpu: str
    cpu_cores: int
    ram_total_mb: int
    ram_free_mb: int
    gpus: list[GpuInfo] = field(default_factory=list)
    has_cuda: bool = False
    has_vulkan: bool = False


def _run(cmd: list[str], timeout: float = 10) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def _detect_nvidia() -> list[GpuInfo]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(GpuInfo(
            name=parts[1], vram_mb=int(parts[2]), vram_free_mb=int(parts[3]),
            vendor="nvidia", index=int(parts[0]), driver=parts[4],
        ))
    cuda_ver = _run(["nvcc", "--version"])
    if cuda_ver:
        for line in cuda_ver.splitlines():
            if "release" in line.lower():
                for g in gpus:
                    g.compute = line.strip().split(",")[-1].strip()
    return gpus


def _detect_amd() -> list[GpuInfo]:
    gpus = []
    out = _run(["rocm-smi", "--showid", "--showmeminfo", "vram", "--json"])
    if out:
        try:
            data = json.loads(out)
            for k, v in data.items():
                if not k.startswith("card"):
                    continue
                idx = int(k.replace("card", ""))
                total = int(v.get("VRAM Total Memory (B)", 0)) // (1024 * 1024)
                used = int(v.get("VRAM Total Used Memory (B)", 0)) // (1024 * 1024)
                gpus.append(GpuInfo(
                    name=v.get("Card Series", f"AMD GPU {idx}"),
                    vram_mb=total, vram_free_mb=total - used,
                    vendor="amd", index=idx,
                ))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    if not gpus:
        out = _run(["vulkaninfo", "--summary"])
        if out:
            for line in out.splitlines():
                low = line.lower()
                if "amd" in low and "devicename" in low:
                    name = line.split("=")[-1].strip() if "=" in line else "AMD GPU"
                    gpus.append(GpuInfo(name=name, vram_mb=0, vram_free_mb=0, vendor="amd"))
    return gpus


def _detect_intel() -> list[GpuInfo]:
    gpus = []
    out = _run(["vulkaninfo", "--summary"])
    if out:
        for line in out.splitlines():
            low = line.lower()
            if "intel" in low and "devicename" in low:
                name = line.split("=")[-1].strip() if "=" in line else "Intel GPU"
                gpus.append(GpuInfo(name=name, vram_mb=0, vram_free_mb=0, vendor="intel"))
    if not gpus and platform.system() == "Windows":
        dxdiag = _run(["powershell", "-Command",
                        "Get-CimInstance Win32_VideoController | Where-Object {$_.Name -match 'Intel'} | "
                        "Select-Object Name, AdapterRAM | ConvertTo-Json"])
        if dxdiag:
            try:
                data = json.loads(dxdiag)
                if isinstance(data, dict):
                    data = [data]
                for d in data:
                    vram = int(d.get("AdapterRAM", 0)) // (1024 * 1024)
                    gpus.append(GpuInfo(name=d.get("Name", "Intel GPU"),
                                        vram_mb=vram, vram_free_mb=vram, vendor="intel"))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return gpus


def _ram_info() -> tuple[int, int]:
    if platform.system() == "Windows":
        out = _run(["powershell", "-Command",
                     "$os = Get-CimInstance Win32_OperatingSystem; "
                     "\"$($os.TotalVisibleMemorySize),$($os.FreePhysicalMemory)\""])
        if out:
            parts = out.strip().split(",")
            return int(parts[0]) // 1024, int(parts[1]) // 1024
    else:
        try:
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    k, v = line.split(":")
                    mem[k.strip()] = int(v.strip().split()[0])
                total = mem.get("MemTotal", 0) // 1024
                free = mem.get("MemAvailable", mem.get("MemFree", 0)) // 1024
                return total, free
        except Exception:
            pass
    return 0, 0


def _cpu_info() -> tuple[str, int]:
    cores = os.cpu_count() or 1
    if platform.system() == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER", "Unknown CPU")
        out = _run(["powershell", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
        if out:
            name = out.strip()
    else:
        name = "Unknown CPU"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":")[1].strip()
                        break
        except Exception:
            pass
    return name, cores


def detect() -> SystemInfo:
    cpu_name, cpu_cores = _cpu_info()
    ram_total, ram_free = _ram_info()

    info = SystemInfo(
        os=platform.system(),
        arch=platform.machine(),
        cpu=cpu_name,
        cpu_cores=cpu_cores,
        ram_total_mb=ram_total,
        ram_free_mb=ram_free,
    )

    info.gpus.extend(_detect_nvidia())
    info.gpus.extend(_detect_amd())
    info.gpus.extend(_detect_intel())

    info.has_cuda = bool(shutil.which("nvcc")) or any(g.vendor == "nvidia" for g in info.gpus)
    info.has_vulkan = bool(shutil.which("vulkaninfo"))

    for i, g in enumerate(info.gpus):
        g.index = i

    return info


def find_binaries(search_paths: Optional[list[str]] = None) -> dict[str, Path]:
    """Find llama.cpp executables."""
    if search_paths is None:
        search_paths = []
        base = Path(__file__).resolve().parent.parent.parent
        for d in base.glob("build*/bin/Release"):
            search_paths.append(str(d))
        for d in base.glob("build*/bin"):
            search_paths.append(str(d))
        search_paths.append(str(base))

    targets = ["llama-server", "llama-cli", "llama-bench", "llama-bells-selftest",
               "llama-quantize", "llama-gguf"]
    ext = ".exe" if platform.system() == "Windows" else ""

    found = {}
    for sp in search_paths:
        p = Path(sp)
        if not p.is_dir():
            continue
        for t in targets:
            if t in found:
                continue
            candidate = p / f"{t}{ext}"
            if candidate.is_file():
                found[t] = candidate
    return found


def find_models(search_paths: Optional[list[str]] = None, max_depth: int = 3) -> list[dict]:
    """Scan for .gguf files and extract basic metadata."""
    if search_paths is None:
        search_paths = []
        for drive in ["A:", "C:", "D:", "E:"]:
            for sub in ["\\models", "\\lms", "\\bells-models"]:
                p = f"{drive}{sub}"
                if os.path.isdir(p):
                    search_paths.append(p)
        home = Path.home()
        for sub in ["models", "Downloads", ".cache/lm-studio/models",
                     ".cache/huggingface/hub"]:
            p = home / sub
            if p.is_dir():
                search_paths.append(str(p))

    raw = []
    seen = set()
    for sp in search_paths:
        p = Path(sp)
        if not p.is_dir():
            continue
        for gguf in _scan_gguf(p, max_depth):
            real = str(gguf.resolve())
            if real in seen:
                continue
            seen.add(real)
            try:
                size_gb = gguf.stat().st_size / (1024 ** 3)
            except OSError:
                continue
            name = gguf.stem
            parts = name.rsplit("-", 1)
            quant = parts[-1] if len(parts) > 1 else "unknown"
            raw.append({
                "path": str(gguf),
                "name": name,
                "quant": quant,
                "size_gb": round(size_gb, 2),
                "dir": str(gguf.parent),
            })

    import re
    groups: dict[str, list[dict]] = {}
    for m in raw:
        split_match = re.search(r"-0000\d+-of-0000\d+$", m["name"])
        if split_match:
            key = m["name"][:split_match.start()] + "|" + m["dir"]
        else:
            key = m["name"] + "|" + m["path"]
        groups.setdefault(key, []).append(m)

    models = []
    for key, parts in groups.items():
        if len(parts) == 1:
            models.append(parts[0])
        else:
            parts.sort(key=lambda x: x["path"])
            total_gb = round(sum(p["size_gb"] for p in parts), 2)
            first = parts[0]
            base_name = key.split("|")[0]
            qm = re.search(r"(Q\d+[_A-Za-z]*|IQ\d+[_A-Za-z]*|F16|F32|BF16)", base_name)
            quant = qm.group(1) if qm else first["quant"]
            models.append({
                "path": first["path"],
                "name": base_name,
                "quant": quant,
                "size_gb": total_gb,
                "dir": first["dir"],
                "parts": len(parts),
            })
    models.sort(key=lambda m: m["name"])
    return models


def _scan_gguf(root: Path, max_depth: int, depth: int = 0):
    """Yield .gguf files up to max_depth levels deep."""
    if depth > max_depth:
        return
    try:
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix == ".gguf":
                yield entry
            elif entry.is_dir() and not entry.name.startswith("."):
                yield from _scan_gguf(entry, max_depth, depth + 1)
    except PermissionError:
        pass


if __name__ == "__main__":
    info = detect()
    print(f"OS: {info.os} {info.arch}")
    print(f"CPU: {info.cpu} ({info.cpu_cores} cores)")
    print(f"RAM: {info.ram_total_mb} MB total, {info.ram_free_mb} MB free")
    for g in info.gpus:
        print(f"GPU {g.index}: {g.name} — {g.vram_mb} MB VRAM ({g.vram_free_mb} MB free) [{g.vendor}]")
    print(f"CUDA: {info.has_cuda}, Vulkan: {info.has_vulkan}")
    print()
    bins = find_binaries()
    for name, path in bins.items():
        print(f"  {name}: {path}")
    print()
    models = find_models()
    for m in models:
        print(f"  {m['name']} ({m['quant']}) — {m['size_gb']} GB — {m['dir']}")
