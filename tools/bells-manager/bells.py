#!/usr/bin/env python3
"""BELLS Manager — TUI/GUI for configuring and benchmarking BELLS.

Usage:
    bells                      Launch the TUI
    bells --gui                Launch in browser (requires textual-web)
    bells bench [OPTIONS]      Run benchmark from command line
    bells system               Print system info
    bells models               List local models
    bells install              Run the installer
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_tui(args):
    from ui.app import BellsApp
    app = BellsApp()
    app.run()


def cmd_gui(args):
    try:
        from gui.host import launch
        launch()
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pywebview: pip install pywebview")
        sys.exit(1)


def cmd_bench(args):
    import system
    from bench.engine import BenchRunner, PROFILES

    if not args.model:
        models = system.find_models()
        if not models:
            print("No models found. Specify --model PATH")
            sys.exit(1)
        print("Available models:")
        for i, m in enumerate(models):
            print(f"  [{i}] {m['name']} ({m['quant']}) — {m['size_gb']} GB — {m['dir']}")
        try:
            idx = int(input("\nSelect model number: "))
            args.model = models[idx]["path"]
        except (ValueError, IndexError, KeyboardInterrupt):
            sys.exit(1)

    bins = system.find_binaries()
    cli = str(bins.get("llama-cli", ""))
    if not cli:
        print("llama-cli not found. Build BELLS first or specify --cli PATH")
        sys.exit(1)

    print(f"\nBELLS Benchmark — {args.profile} profile")
    print(f"Model: {Path(args.model).name}")
    print(f"Context: {args.context}")
    print(f"CLI: {cli}")
    print()

    def on_progress(msg):
        print(msg)

    runner = BenchRunner(
        cli_path=cli, model=args.model, context=args.context,
        profile_name=args.profile, output_dir=args.output,
        on_progress=on_progress,
    )

    try:
        results = runner.run()
    except KeyboardInterrupt:
        runner.cancel()
        print("\nBenchmark cancelled.")
        sys.exit(1)

    print()
    print(runner.summary_text())


def cmd_system(args):
    import system
    info = system.detect()
    print(f"OS:   {info.os} {info.arch}")
    print(f"CPU:  {info.cpu} ({info.cpu_cores} cores)")
    print(f"RAM:  {info.ram_total_mb // 1024} GB total, {info.ram_free_mb // 1024} GB free")
    print(f"CUDA: {'Yes' if info.has_cuda else 'No'}")
    print(f"Vulkan: {'Yes' if info.has_vulkan else 'No'}")
    print()
    if info.gpus:
        for g in info.gpus:
            vram = f"{g.vram_mb} MB ({g.vram_free_mb} MB free)" if g.vram_mb else "Unknown VRAM"
            print(f"GPU {g.index}: {g.name} — {vram} [{g.vendor}]")
    else:
        print("No GPUs detected")
    print()
    bins = system.find_binaries()
    if bins:
        print("Binaries:")
        for name, path in bins.items():
            print(f"  {name}: {path}")
    else:
        print("No llama.cpp binaries found.")


def cmd_models(args):
    import system
    models = system.find_models()
    if not models:
        print("No .gguf models found.")
        return
    print(f"Found {len(models)} models:\n")
    for m in models:
        print(f"  {m['name']:<50} {m['quant']:<12} {m['size_gb']:>7.1f} GB  {m['dir']}")


def cmd_install(args):
    installer = Path(__file__).parent / "install.py"
    if installer.exists():
        import subprocess
        subprocess.run([sys.executable, str(installer)], check=False)
    else:
        print("Installer not found.")


def main():
    parser = argparse.ArgumentParser(
        prog="bells",
        description="BELLS Manager — configure, benchmark and run BELLS",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tui", help="Launch TUI (default)")
    sub.add_parser("gui", help="Launch in browser")
    sub.add_parser("system", help="Print system info")
    sub.add_parser("models", help="List local models")
    sub.add_parser("install", help="Run installer")

    sub.add_parser("estimate", help="Estimate BELLS performance (has own args)")

    bench_p = sub.add_parser("bench", help="Run benchmark")
    bench_p.add_argument("-m", "--model", help="Path to .gguf model")
    bench_p.add_argument("-c", "--context", type=int, default=4096, help="Context window")
    bench_p.add_argument("-p", "--profile", choices=["short", "medium", "long"],
                         default="short", help="Benchmark profile")
    bench_p.add_argument("-o", "--output", default=None, help="Output directory")

    # Estimate subcommand has its own argparse, so intercept early
    if len(sys.argv) > 1 and sys.argv[1] == "estimate":
        from estimator import main as est_main
        est_main(sys.argv[2:])
        return

    args = parser.parse_args()

    if args.command == "bench":
        cmd_bench(args)
    elif args.command == "system":
        cmd_system(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "gui":
        cmd_gui(args)
    elif args.command == "install":
        cmd_install(args)
    else:
        try:
            cmd_tui(args)
        except ImportError:
            print("TUI requires textual: pip install textual")
            print("Or use: bells system / bells bench / bells models")
            sys.exit(1)


if __name__ == "__main__":
    main()
