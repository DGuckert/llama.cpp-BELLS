#!/usr/bin/env python3
"""BELLS installer — detects GPU, builds from source or downloads precompiled binaries."""

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://github.com/DGuckert/llama.cpp-BELLS"
RELEASE_TAG = "latest"
RELEASE_BASE = f"{REPO}/releases/download"

def log(msg: str):
    print(f"[BELLS] {msg}")


def error(msg: str):
    print(f"[BELLS] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        error(f"Python 3.10+ required, got {v.major}.{v.minor}")
    log(f"Python {v.major}.{v.minor}.{v.micro}")


def detect_gpu() -> tuple[str, bool, bool]:
    """Returns (vendor, has_cuda_toolkit, has_vulkan_sdk)."""
    vendor = "none"
    has_cuda = bool(shutil.which("nvcc"))
    has_vulkan = bool(shutil.which("vulkaninfo"))

    if shutil.which("nvidia-smi"):
        vendor = "nvidia"
    elif has_vulkan:
        try:
            out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10)
            text = out.stdout.lower()
            if "amd" in text or "radeon" in text:
                vendor = "amd"
            elif "intel" in text:
                vendor = "intel"
            else:
                vendor = "vulkan"
        except Exception:
            vendor = "vulkan"
    elif platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=10,
            )
            text = out.stdout.lower()
            if "nvidia" in text:
                vendor = "nvidia"
            elif "amd" in text or "radeon" in text:
                vendor = "amd"
            elif "intel" in text:
                vendor = "intel"
        except Exception:
            pass

    return vendor, has_cuda, has_vulkan


def install_python_deps():
    log("Installing Python dependencies...")
    deps = ["textual>=0.40.0", "huggingface_hub"]
    run([sys.executable, "-m", "pip", "install", "--quiet"] + deps, check=False)


def try_precompiled(vendor: str, has_cuda: bool) -> bool:
    """Try to download precompiled binaries from GitHub releases."""
    if platform.system() != "Windows":
        return False

    if vendor == "nvidia" and has_cuda:
        asset = "bells-win-cuda-vulkan.zip"
    else:
        asset = "bells-win-vulkan.zip"

    log(f"Trying precompiled binary: {asset}")

    try:
        releases_url = f"https://api.github.com/repos/DGuckert/llama.cpp-BELLS/releases/latest"
        req = urllib.request.Request(releases_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json
            data = json.loads(resp.read())

        download_url = None
        for a in data.get("assets", []):
            if a["name"] == asset:
                download_url = a["browser_download_url"]
                break

        if not download_url:
            log(f"  Asset {asset} not found in latest release")
            return False

        dest_dir = Path(__file__).resolve().parent.parent.parent / "bin"
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / asset

        log(f"  Downloading {download_url}...")
        urllib.request.urlretrieve(download_url, zip_path)

        log(f"  Extracting to {dest_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()

        exes = list(dest_dir.glob("*.exe"))
        if exes:
            log(f"  Installed {len(exes)} executables to {dest_dir}")
            return True
        return False

    except Exception as e:
        log(f"  Download failed: {e}")
        return False


def build_from_source(vendor: str, has_cuda: bool, has_vulkan: bool) -> bool:
    """Clone and build from source."""
    repo_dir = Path(__file__).resolve().parent.parent.parent

    if not (repo_dir / "CMakeLists.txt").exists():
        log(f"Cloning repository to {repo_dir}...")
        run(["git", "clone", REPO, str(repo_dir)], check=True)

    if not shutil.which("cmake"):
        error("cmake not found. Install CMake: https://cmake.org/download/")

    build_dir = repo_dir / "build"

    cmake_args = ["-B", str(build_dir)]
    if vendor == "nvidia" and has_cuda:
        cmake_args.append("-DGGML_CUDA=ON")
        log("Building with CUDA support")
    if has_vulkan:
        cmake_args.append("-DGGML_VULKAN=ON")
        log("Building with Vulkan support")
    if not has_cuda and not has_vulkan:
        log("Building CPU-only (no GPU SDK found)")

    if platform.system() == "Windows":
        cmake_args += ["-G", "Ninja"] if shutil.which("ninja") else []

    log("Configuring...")
    r = run(["cmake"] + cmake_args, cwd=str(repo_dir))
    if r.returncode != 0:
        return False

    log("Building (this may take a while)...")
    r = run(["cmake", "--build", str(build_dir), "--config", "Release",
             "-j", str(os.cpu_count() or 4)], cwd=str(repo_dir))
    return r.returncode == 0


def create_shortcut():
    """Create a 'bells' command available system-wide."""
    bells_py = Path(__file__).resolve().parent / "bells.py"
    py_exe = sys.executable

    if platform.system() == "Windows":
        py_cmd = shutil.which("py") or py_exe
        bells_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "BELLS" / "bin"
        bells_dir.mkdir(parents=True, exist_ok=True)

        bat = bells_dir / "bells.bat"
        bat.write_text(f'@echo off\n"{py_cmd}" "{bells_py}" %*\n')
        log(f"Created {bat}")

        repo_bat = Path(__file__).resolve().parent.parent.parent / "bells.bat"
        repo_bat.write_text(f'@echo off\n"{py_cmd}" "{bells_py}" %*\n')

        _add_to_path_windows(str(bells_dir))
    else:
        local_bin = Path.home() / ".local" / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        sh = local_bin / "bells"
        sh.write_text(f'#!/bin/sh\nexec "{py_exe}" "{bells_py}" "$@"\n')
        sh.chmod(0o755)
        log(f"Created {sh}")

        if str(local_bin) not in os.environ.get("PATH", ""):
            log(f"Add to PATH: export PATH=\"{local_bin}:$PATH\"")
            for rc in [Path.home() / ".bashrc", Path.home() / ".zshrc"]:
                if rc.exists():
                    content = rc.read_text()
                    line = f'\nexport PATH="{local_bin}:$PATH"  # BELLS\n'
                    if "# BELLS" not in content:
                        rc.write_text(content + line)
                        log(f"  Added to {rc}")


def _add_to_path_windows(dir_path: str):
    """Add a directory to the user's PATH on Windows, persistently."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"[Environment]::GetEnvironmentVariable('PATH', 'User')"],
            capture_output=True, text=True, timeout=10,
        )
        user_path = result.stdout.strip()

        if dir_path.lower() in user_path.lower():
            log(f"  Already on PATH: {dir_path}")
            return

        new_path = f"{user_path};{dir_path}" if user_path else dir_path
        subprocess.run(
            ["powershell", "-Command",
             f"[Environment]::SetEnvironmentVariable('PATH', '{new_path}', 'User')"],
            capture_output=True, text=True, timeout=10,
        )
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + dir_path
        log(f"  Added to user PATH: {dir_path}")
        log(f"  Restart your terminal for 'bells' to work everywhere")
    except Exception as e:
        log(f"  Could not add to PATH automatically: {e}")
        log(f"  Manually add to PATH: {dir_path}")


def main():
    log("BELLS Installer")
    log("=" * 50)

    check_python()

    vendor, has_cuda, has_vulkan = detect_gpu()
    log(f"GPU vendor: {vendor}")
    log(f"CUDA toolkit: {'Yes' if has_cuda else 'No'}")
    log(f"Vulkan SDK: {'Yes' if has_vulkan else 'No'}")
    print()

    install_python_deps()
    print()

    repo_dir = Path(__file__).resolve().parent.parent.parent
    bins_exist = any((repo_dir / "bin").glob("*.exe")) if (repo_dir / "bin").exists() else False

    if not bins_exist:
        build_dirs = list(repo_dir.glob("build*/bin/Release/llama-server*"))
        bins_exist = bool(build_dirs)

    if not bins_exist:
        log("No binaries found. Attempting to get them...")
        if not try_precompiled(vendor, has_cuda):
            log("Precompiled not available, building from source...")
            if not build_from_source(vendor, has_cuda, has_vulkan):
                error("Build failed. Check the output above for errors.")
    else:
        log("Binaries already present, skipping build.")

    print()
    create_shortcut()
    print()

    log("Installation complete!")
    log("")
    log("Quick start:")
    log("  bells system       — Show system info")
    log("  bells models       — List local models")
    log("  bells bench        — Run benchmark")
    log("  bells estimate run -g \"3060\" -m qwen3.6-35b -q Q4_K_M — Estimate performance")


if __name__ == "__main__":
    main()
