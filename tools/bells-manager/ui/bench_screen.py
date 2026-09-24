"""Benchmark screen — pick model, context, profile and watch it run."""

import time
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label,
                             ProgressBar, RichLog, Select, Static)


class BenchScreen(Screen):
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+c", "cancel_bench", "Cancel"),
    ]

    CSS = """
    BenchScreen {
        background: $background;
    }

    #bench-header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    #bench-title {
        text-style: bold;
        color: $accent;
    }

    #bench-subtitle {
        color: $text-muted;
    }

    #bench-setup {
        height: auto;
        padding: 1 2;
        border-bottom: solid $primary;
        background: $surface;
    }

    .setup-row {
        height: 3;
    }

    .setup-label {
        width: 12;
        padding: 1 0 0 0;
        color: $text-muted;
        text-style: bold;
    }

    .setup-input {
        width: 1fr;
    }

    #bench-browse-btn {
        width: 12;
        margin: 0 0 0 1;
    }

    #bench-body {
        height: 1fr;
    }

    #bench-log {
        height: 1fr;
        margin: 1 2;
        border: solid $primary;
        background: $background;
    }

    #bench-results {
        height: 40%;
        margin: 0 2 1 2;
    }

    #bench-results > .datatable--header {
        background: $surface;
        color: $accent;
        text-style: bold;
    }

    #bench-actions {
        height: 3;
        dock: bottom;
        padding: 0 2;
        background: $panel;
        border-top: solid $primary;
    }

    #bench-actions Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(self, cli_path: str = "", model_path: str = ""):
        super().__init__()
        self.cli_path = cli_path
        self.model_path = model_path
        self.runner = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="bench-header"):
            yield Static("Benchmark", id="bench-title")
            yield Static("Finds optimal flags for your model and hardware", id="bench-subtitle")
        with Vertical(id="bench-setup"):
            with Horizontal(classes="setup-row"):
                yield Static("Model", classes="setup-label")
                yield Input(value=self.model_path, id="bench-model", classes="setup-input")
                yield Button("Browse", id="bench-browse-btn")
            with Horizontal(classes="setup-row"):
                yield Static("Context", classes="setup-label")
                yield Input(value="4096", id="bench-context", classes="setup-input")
            with Horizontal(classes="setup-row"):
                yield Static("Profile", classes="setup-label")
                yield Select(
                    [("Short  ~1hr  (7 configs)", "short"),
                     ("Medium ~4hr  (22 configs)", "medium"),
                     ("Long   ~8hr  (34 configs)", "long")],
                    value="short", id="bench-profile",
                )
        with Horizontal(id="bench-actions"):
            yield Button("Start", id="bench-start-btn", variant="success")
            yield Button("Cancel", id="bench-cancel-btn", variant="error", disabled=True)
        with Vertical(id="bench-body"):
            yield RichLog(id="bench-log", highlight=True, markup=True)
            yield DataTable(id="bench-results")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#bench-results", DataTable)
        table.add_columns("Config", "Gen tok/s", "Prompt tok/s", "Status")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "bench-start-btn":
            self._start_bench()
        elif event.button.id == "bench-cancel-btn":
            if self.runner:
                self.runner.cancel()
                self._log("[bold red]Cancelling...[/]")
        elif event.button.id == "bench-browse-btn":
            from .models_screen import ModelsScreen
            def on_model(path: str | None) -> None:
                if path:
                    self.query_one("#bench-model", Input).value = path
            self.app.push_screen(ModelsScreen(), on_model)

    def _log(self, msg: str) -> None:
        try:
            self.query_one("#bench-log", RichLog).write(msg)
        except Exception:
            pass

    @work(thread=True)
    def _start_bench(self) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from bench.engine import BenchRunner

        model = self.query_one("#bench-model", Input).value.strip()
        if not model:
            self._log("[red]No model selected[/]")
            return

        context = int(self.query_one("#bench-context", Input).value.strip() or "4096")
        profile = self.query_one("#bench-profile", Select).value

        cli = self.cli_path
        if not cli:
            import system
            bins = system.find_binaries()
            cli = str(bins.get("llama-cli", ""))
        if not cli:
            self._log("[red]llama-cli not found. Build or install first.[/]")
            return

        self.app.call_from_thread(
            self.query_one("#bench-start-btn", Button).__setattr__, "disabled", True)
        self.app.call_from_thread(
            self.query_one("#bench-cancel-btn", Button).__setattr__, "disabled", False)

        self._log(f"[bold]BELLS Benchmark[/]")
        self._log(f"  Model:   {Path(model).name}")
        self._log(f"  Context: {context}")
        self._log(f"  Profile: {profile}")
        self._log("")

        table = self.query_one("#bench-results", DataTable)
        self.app.call_from_thread(table.clear)

        def on_progress(msg: str):
            self.app.call_from_thread(self._log, msg)

        self.runner = BenchRunner(
            cli_path=cli, model=model, context=context,
            profile_name=profile, on_progress=on_progress,
        )

        results = self.runner.run()

        for r in sorted(results, key=lambda x: -x.score()):
            status = "OOM" if r.oom else r.error if r.error else "OK"
            self.app.call_from_thread(
                table.add_row,
                r.config.short_label(),
                f"{r.gen_tps:.1f}",
                f"{r.prompt_tps:.1f}",
                status,
            )

        if self.runner.best:
            self._log("")
            self._log(f"[bold green]BEST: {self.runner.best.config.short_label()} "
                       f"-- {self.runner.best.gen_tps:.1f} tok/s[/]")
            self._log(f"  FLAGS: {' '.join(self.runner.best.config.extra_flags)}")

        self.app.call_from_thread(
            self.query_one("#bench-start-btn", Button).__setattr__, "disabled", False)
        self.app.call_from_thread(
            self.query_one("#bench-cancel-btn", Button).__setattr__, "disabled", True)

    def action_go_back(self) -> None:
        if self.runner:
            self.runner.cancel()
        self.app.pop_screen()

    def action_cancel_bench(self) -> None:
        if self.runner:
            self.runner.cancel()
            self._log("[bold red]Cancelling...[/]")
