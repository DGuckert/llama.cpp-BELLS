"""Server management screen — start, stop, monitor llama-server."""

import os
import signal
import subprocess
import threading
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, RichLog, Static


class StatusBadge(Static):
    """Color-coded status indicator."""

    DEFAULT_CSS = """
    StatusBadge {
        width: auto;
        padding: 0 2;
        text-style: bold;
    }

    StatusBadge.stopped {
        color: $error;
    }

    StatusBadge.running {
        color: $success;
    }

    StatusBadge.starting {
        color: $warning;
    }
    """

    def set_state(self, text: str, state: str = "stopped") -> None:
        self.update(text)
        self.remove_class("stopped", "running", "starting")
        self.add_class(state)


class ServerScreen(Screen):
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+c", "stop_server", "Stop"),
    ]

    CSS = """
    ServerScreen {
        background: $background;
    }

    #server-header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    #server-title {
        text-style: bold;
        color: $accent;
    }

    #server-status-row {
        height: 1;
        layout: horizontal;
    }

    .status-label {
        width: 10;
        color: $text-muted;
    }

    #server-cmd {
        color: $text-muted;
        height: 1;
    }

    #server-log {
        height: 1fr;
        margin: 1 2;
        border: solid $primary;
        background: $background;
    }

    #server-actions {
        height: 3;
        dock: bottom;
        padding: 0 2;
        background: $panel;
        border-top: solid $primary;
    }

    #server-actions Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(self, server_path: str, args: list[str]):
        super().__init__()
        self.server_path = server_path
        self.server_args = args
        self.process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="server-header"):
            yield Static("Server", id="server-title")
            with Horizontal(id="server-status-row"):
                yield Static("Status", classes="status-label")
                yield StatusBadge("Stopped", id="status-badge", classes="stopped")
            cmd = f"{Path(self.server_path).name} {' '.join(self.server_args)}"
            yield Static(cmd, id="server-cmd")
        with Horizontal(id="server-actions"):
            yield Button("Start", id="srv-start-btn", variant="success")
            yield Button("Stop", id="srv-stop-btn", variant="error", disabled=True)
            yield Button("Restart", id="srv-restart-btn", variant="default", disabled=True)
            yield Button("Open Browser", id="srv-browser-btn", variant="primary", disabled=True)
        yield RichLog(id="server-log", highlight=True, markup=True)
        yield Footer()

    def _log(self, msg: str) -> None:
        try:
            self.query_one("#server-log", RichLog).write(msg)
        except Exception:
            pass

    def _set_status(self, text: str, state: str = "stopped") -> None:
        try:
            self.query_one("#status-badge", StatusBadge).set_state(text, state)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "srv-start-btn":
            self._start_server()
        elif event.button.id == "srv-stop-btn":
            self._stop_server()
        elif event.button.id == "srv-restart-btn":
            self._stop_server()
            self._start_server()
        elif event.button.id == "srv-browser-btn":
            self._open_browser()

    @work(thread=True)
    def _start_server(self) -> None:
        if self.process and self.process.poll() is None:
            self._log("[yellow]Server already running[/]")
            return

        self.app.call_from_thread(self._set_status, "Starting...", "starting")
        self._log(f"[bold]Starting server...[/]")

        try:
            self.process = subprocess.Popen(
                [self.server_path] + self.server_args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "GGML_CUDA_NO_PINNED": "1"},
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        except Exception as e:
            self._log(f"[red]Failed to start: {e}[/]")
            self.app.call_from_thread(self._set_status, f"Error: {e}", "stopped")
            return

        self.app.call_from_thread(self._set_status, f"Running (PID {self.process.pid})", "running")
        self.app.call_from_thread(
            self.query_one("#srv-start-btn", Button).__setattr__, "disabled", True)
        self.app.call_from_thread(
            self.query_one("#srv-stop-btn", Button).__setattr__, "disabled", False)
        self.app.call_from_thread(
            self.query_one("#srv-restart-btn", Button).__setattr__, "disabled", False)
        self.app.call_from_thread(
            self.query_one("#srv-browser-btn", Button).__setattr__, "disabled", False)

        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        if not self.process or not self.process.stdout:
            return
        try:
            for line in self.process.stdout:
                self.app.call_from_thread(self._log, line.rstrip())
        except Exception:
            pass

        rc = self.process.wait() if self.process else -1
        self.app.call_from_thread(self._log, f"[bold]Server exited (code {rc})[/]")
        self.app.call_from_thread(self._set_status, f"Stopped (exit {rc})", "stopped")
        self.app.call_from_thread(
            self.query_one("#srv-start-btn", Button).__setattr__, "disabled", False)
        self.app.call_from_thread(
            self.query_one("#srv-stop-btn", Button).__setattr__, "disabled", True)
        self.app.call_from_thread(
            self.query_one("#srv-restart-btn", Button).__setattr__, "disabled", True)

    def _stop_server(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self._log("[yellow]Stopping server...[/]")
        self.app.call_from_thread(self._set_status, "Stopping...", "starting")
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._log("[red]Force killing...[/]")
            self.process.kill()
        except Exception as e:
            self._log(f"[red]Error stopping: {e}[/]")

    def _open_browser(self) -> None:
        port = "8080"
        host = "127.0.0.1"
        for i, arg in enumerate(self.server_args):
            if arg == "--port" and i + 1 < len(self.server_args):
                port = self.server_args[i + 1]
            elif arg == "--host" and i + 1 < len(self.server_args):
                host = self.server_args[i + 1]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        import webbrowser
        webbrowser.open(f"http://{host}:{port}")

    def action_go_back(self) -> None:
        self._stop_server()
        self.app.pop_screen()

    def action_stop_server(self) -> None:
        self._stop_server()
