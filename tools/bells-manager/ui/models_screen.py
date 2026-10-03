"""Model browser and HuggingFace search screen."""

import os
import subprocess
import threading
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label,
                             Static)


class ModelsScreen(Screen):
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("s", "search_hf", "HuggingFace"),
        ("escape", "app.pop_screen", "Back"),
    ]

    CSS = """
    ModelsScreen {
        background: $background;
    }

    #models-header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    #models-title {
        text-style: bold;
        color: $accent;
    }

    #models-subtitle {
        color: $text-muted;
    }

    #model-table {
        height: 1fr;
        margin: 1 2;
    }

    #model-table > .datatable--header {
        background: $surface;
        color: $accent;
        text-style: bold;
    }

    #hf-panel {
        height: auto;
        max-height: 45%;
        display: none;
        border-top: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #hf-panel.visible {
        display: block;
    }

    #hf-title {
        text-style: bold;
        color: $accent;
        padding: 0 0 1 0;
    }

    #search-row {
        height: 3;
    }

    #search-row Input {
        width: 1fr;
        margin: 0 1 0 0;
    }

    #search-row Button {
        width: 14;
    }

    #hf-table {
        height: 1fr;
        max-height: 15;
        margin: 1 0;
    }

    #hf-status {
        color: $text-muted;
        height: 1;
    }

    #status-bar {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self, model_paths: list[str] | None = None):
        super().__init__()
        self.model_paths = model_paths
        self.models: list[dict] = []
        self.hf_results: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="models-header"):
            yield Static("Local Models", id="models-title")
            yield Static("Select a model to use it in Configure or Benchmark", id="models-subtitle")
        yield DataTable(id="model-table")
        with Vertical(id="hf-panel"):
            yield Static("HuggingFace Search", id="hf-title")
            with Horizontal(id="search-row"):
                yield Input(placeholder="Search (e.g. Qwen3 GGUF Q4)...", id="hf-input")
                yield Button("Search", id="hf-search-btn", variant="primary")
            yield DataTable(id="hf-table")
            yield Label("Press S to toggle search panel", id="hf-status")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#model-table", DataTable)
        table.add_columns("Name", "Quant", "Size", "Location")
        table.cursor_type = "row"
        table.zebra_stripes = True

        hf_table = self.query_one("#hf-table", DataTable)
        hf_table.add_columns("Model", "Quant", "Size", "Downloads", "Repo ID")
        hf_table.cursor_type = "row"
        hf_table.zebra_stripes = True

        self.action_refresh()

    def action_refresh(self) -> None:
        import system
        self.models = system.find_models(self.model_paths)
        table = self.query_one("#model-table", DataTable)
        table.clear()
        for m in self.models:
            size = f"{m['size_gb']:.1f} GB"
            parts_info = f" ({m['parts']} parts)" if m.get("parts", 1) > 1 else ""
            table.add_row(m["name"], m["quant"], size + parts_info, m["dir"])
        self.query_one("#status-bar", Static).update(
            f" {len(self.models)} models found  |  R=Refresh  S=HuggingFace")

    def action_search_hf(self) -> None:
        panel = self.query_one("#hf-panel")
        panel.toggle_class("visible")
        if panel.has_class("visible"):
            self.query_one("#hf-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hf-search-btn":
            query = self.query_one("#hf-input", Input).value.strip()
            if query:
                self._search_hf(query)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "hf-input":
            query = event.value.strip()
            if query:
                self._search_hf(query)

    @work(thread=True)
    def _search_hf(self, query: str) -> None:
        status = self.query_one("#hf-status", Label)
        self.app.call_from_thread(status.update, "Searching...")

        try:
            from huggingface_hub import HfApi
            api = HfApi()
            results = api.list_models(
                search=query, filter="gguf", sort="downloads", direction=-1, limit=20,
            )
            self.hf_results = []
            table = self.query_one("#hf-table", DataTable)
            self.app.call_from_thread(table.clear)

            for model in results:
                name = model.id.split("/")[-1] if "/" in model.id else model.id
                downloads = f"{model.downloads:,}" if model.downloads else "?"
                entry = {"id": model.id, "name": name, "downloads": downloads}
                self.hf_results.append(entry)
                self.app.call_from_thread(table.add_row, name, "", "", downloads, model.id)

            self.app.call_from_thread(status.update, f"{len(self.hf_results)} results found")
        except ImportError:
            self.app.call_from_thread(status.update,
                                  "pip install huggingface_hub to enable search")
        except Exception as e:
            self.app.call_from_thread(status.update, f"Error: {e}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "model-table":
            row_idx = event.cursor_row
            if 0 <= row_idx < len(self.models):
                model = self.models[row_idx]
                self.dismiss(model["path"])

        elif event.data_table.id == "hf-table":
            row_idx = event.cursor_row
            if 0 <= row_idx < len(self.hf_results):
                hf = self.hf_results[row_idx]
                self.app.push_screen(HFDownloadScreen(hf["id"]))


class HFDownloadScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    CSS = """
    HFDownloadScreen {
        align: center middle;
        background: $background 70%;
    }

    #dl-container {
        width: 80;
        height: auto;
        max-height: 35;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }

    #dl-title {
        text-style: bold;
        color: $accent;
        padding: 0 0 1 0;
    }

    #dl-dir {
        margin: 0 0 1 0;
    }

    #dl-log {
        height: 15;
        overflow-y: auto;
        border: solid $primary;
        padding: 0 1;
        margin: 1 0;
    }

    #dl-actions {
        height: 3;
    }

    #dl-actions Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(self, repo_id: str):
        super().__init__()
        self.repo_id = repo_id

    def compose(self) -> ComposeResult:
        with Vertical(id="dl-container"):
            yield Static(f"Download: {self.repo_id}", id="dl-title")
            yield Input(placeholder="Save directory (leave empty for ~/models)", id="dl-dir")
            with Horizontal(id="dl-actions"):
                yield Button("List Files", id="dl-list-btn", variant="primary")
                yield Button("Download All GGUF", id="dl-go-btn", variant="success", disabled=True)
            yield VerticalScroll(Static("", id="dl-log-text"), id="dl-log")

    @work(thread=True)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dl-list-btn":
            self._list_files()
        elif event.button.id == "dl-go-btn":
            self._download()

    def _list_files(self) -> None:
        log = self.query_one("#dl-log-text", Static)
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            files = api.list_repo_files(self.repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if gguf_files:
                text = "\n".join(f"  {f}" for f in gguf_files)
                self.app.call_from_thread(log.update, text)
                self.app.call_from_thread(
                    self.query_one("#dl-go-btn", Button).__setattr__, "disabled", False)
            else:
                self.app.call_from_thread(log.update, "No .gguf files found in this repo.")
        except Exception as e:
            self.app.call_from_thread(log.update, f"Error: {e}")

    def _download(self) -> None:
        log = self.query_one("#dl-log-text", Static)
        save_dir = self.query_one("#dl-dir", Input).value.strip()
        if not save_dir:
            save_dir = str(Path.home() / "models")

        try:
            from huggingface_hub import hf_hub_download, HfApi
            api = HfApi()
            files = api.list_repo_files(self.repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            os.makedirs(save_dir, exist_ok=True)
            for i, f in enumerate(gguf_files):
                self.app.call_from_thread(log.update,
                                      f"Downloading {i+1}/{len(gguf_files)}: {f}...")
                hf_hub_download(self.repo_id, f, local_dir=save_dir)

            self.app.call_from_thread(log.update,
                                  f"Done! {len(gguf_files)} files saved to {save_dir}")
        except Exception as e:
            self.app.call_from_thread(log.update, f"Error: {e}")
