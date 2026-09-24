"""BELLS Manager — native desktop GUI via pywebview."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webview

import system


class ServerProcess:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.log_lines: list[str] = []
        self._reader: threading.Thread | None = None

    def start(self, exe: str, args: list[str]):
        if self.proc and self.proc.poll() is None:
            return
        self.log_lines.clear()
        env = {**os.environ, "GGML_CUDA_NO_PINNED": "1"}
        self.proc = subprocess.Popen(
            [exe] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            self.log_lines.append(line.rstrip())
            if len(self.log_lines) > 500:
                self.log_lines = self.log_lines[-300:]

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        except Exception:
            pass

    def status(self) -> dict:
        if not self.proc:
            return {"running": False, "pid": None}
        alive = self.proc.poll() is None
        return {"running": alive, "pid": self.proc.pid, "exit_code": self.proc.returncode}

    def get_logs(self, since: int = 0) -> dict:
        lines = self.log_lines[since:]
        return {"lines": lines, "total": len(self.log_lines)}


class Api:
    def __init__(self):
        self._sys_info = None
        self._binaries = {}
        self._server = ServerProcess()
        self._bench_running = False
        self._bench_log: list[str] = []
        self._bench_results: list[dict] = []
        self._bench_cancel = False

    def get_system_info(self) -> dict:
        self._sys_info = system.detect()
        self._binaries = system.find_binaries()
        gpus = []
        for g in self._sys_info.gpus:
            gpus.append({
                "name": g.name, "vram_mb": g.vram_mb, "vram_free_mb": g.vram_free_mb,
                "vendor": g.vendor, "index": g.index, "driver": g.driver, "compute": g.compute,
            })
        return {
            "os": self._sys_info.os, "arch": self._sys_info.arch,
            "cpu": self._sys_info.cpu, "cpu_cores": self._sys_info.cpu_cores,
            "ram_total_mb": self._sys_info.ram_total_mb,
            "ram_free_mb": self._sys_info.ram_free_mb,
            "gpus": gpus,
            "has_cuda": self._sys_info.has_cuda,
            "has_vulkan": self._sys_info.has_vulkan,
            "binaries": {k: str(v) for k, v in self._binaries.items()},
        }

    def get_models(self) -> list:
        return system.find_models()

    def search_huggingface(self, query: str) -> list:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            results = api.list_models(
                search=query, filter="gguf", sort="downloads", direction=-1, limit=20,
            )
            out = []
            for m in results:
                out.append({
                    "id": m.id,
                    "name": m.id.split("/")[-1] if "/" in m.id else m.id,
                    "downloads": m.downloads or 0,
                })
            return out
        except Exception as e:
            return [{"error": str(e)}]

    def start_server(self, args: list[str]) -> dict:
        exe = self._binaries.get("llama-server")
        if not exe:
            return {"error": "llama-server not found"}
        self._server.start(str(exe), args)
        time.sleep(0.5)
        return self._server.status()

    def stop_server(self) -> dict:
        self._server.stop()
        return self._server.status()

    def server_status(self) -> dict:
        return self._server.status()

    def server_logs(self, since: int = 0) -> dict:
        return self._server.get_logs(since)

    def start_benchmark(self, model: str, context: int, profile: str) -> dict:
        if self._bench_running:
            return {"error": "Benchmark already running"}
        cli = self._binaries.get("llama-cli")
        if not cli:
            return {"error": "llama-cli not found"}

        self._bench_running = True
        self._bench_cancel = False
        self._bench_log.clear()
        self._bench_results.clear()

        t = threading.Thread(target=self._run_bench,
                             args=(str(cli), model, context, profile), daemon=True)
        t.start()
        return {"started": True}

    def _run_bench(self, cli: str, model: str, context: int, profile: str):
        try:
            from bench.engine import BenchRunner
            def on_progress(msg: str):
                self._bench_log.append(msg)
                if len(self._bench_log) > 500:
                    self._bench_log = self._bench_log[-300:]

            runner = BenchRunner(
                cli_path=cli, model=model, context=context,
                profile_name=profile, on_progress=on_progress,
            )
            results = runner.run()
            for r in sorted(results, key=lambda x: -x.score()):
                self._bench_results.append({
                    "config": r.config.short_label(),
                    "gen_tps": round(r.gen_tps, 1),
                    "prompt_tps": round(r.prompt_tps, 1),
                    "oom": r.oom,
                    "error": r.error or "",
                    "flags": " ".join(r.config.extra_flags),
                })
        except Exception as e:
            self._bench_log.append(f"ERROR: {e}")
        finally:
            self._bench_running = False

    def cancel_benchmark(self):
        self._bench_cancel = True

    def bench_status(self) -> dict:
        return {
            "running": self._bench_running,
            "log": self._bench_log[-50:],
            "log_total": len(self._bench_log),
            "results": self._bench_results,
        }

    def open_folder(self, path: str):
        os.startfile(path)


def launch():
    api = Api()
    html_path = Path(__file__).parent / "index.html"
    window = webview.create_window(
        "BELLS Manager",
        str(html_path),
        js_api=api,
        width=1100,
        height=750,
        min_size=(800, 500),
        background_color="#0d1117",
    )
    webview.start(debug=False)


if __name__ == "__main__":
    launch()
