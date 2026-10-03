"""BELLS Manager — main TUI application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, Static, TabbedContent, TabPane


LOGO = """\
╔╗ ╔═╗╦  ╦  ╔═╗
╠╩╗║╣ ║  ║  ╚═╗
╚═╝╚═╝╩═╝╩═╝╚═╝"""


class MetricCard(Static):
    """Compact metric display card."""

    DEFAULT_CSS = """
    MetricCard {
        height: 5;
        border: solid $primary;
        padding: 0 2;
        background: $surface;
    }

    MetricCard .metric-label {
        color: $text-muted;
        text-style: bold;
    }

    MetricCard .metric-value {
        color: $success;
        text-style: bold;
        margin-top: 1;
    }
    """

    value = reactive("")

    def __init__(self, label: str, initial_value: str = "—") -> None:
        super().__init__()
        self._label_text = label
        self.value = initial_value

    def compose(self) -> ComposeResult:
        yield Static(self._label_text, classes="metric-label")
        yield Static(self.value, classes="metric-value", id=f"mv-{id(self)}")

    def watch_value(self, new_value: str) -> None:
        try:
            self.query_one(f"#mv-{id(self)}", Static).update(new_value)
        except Exception:
            pass


class MenuCard(Button):
    """Main menu navigation card."""

    DEFAULT_CSS = """
    MenuCard {
        width: 100%;
        height: 7;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        content-align: left middle;
        transition: background 200ms;
    }

    MenuCard:hover {
        background: $primary;
        border: solid $accent;
    }

    MenuCard:focus {
        border: solid $accent;
    }
    """


class BellsApp(App):
    TITLE = "BELLS Manager"
    SUB_TITLE = "Per-layer VRAM expert cache for MoE models"

    CSS = """
    Screen {
        background: $background;
    }

    #main-container {
        width: 100%;
        height: 100%;
        align: center middle;
    }

    #hero {
        width: 76;
        height: auto;
        padding: 1 2;
    }

    #logo {
        text-align: center;
        color: $accent;
        text-style: bold;
        padding: 1 0;
    }

    #tagline {
        text-align: center;
        color: $text-muted;
        padding: 0 0 1 0;
    }

    #metrics-grid {
        grid-size: 3;
        grid-gutter: 1 2;
        height: auto;
        margin: 0 0 1 0;
    }

    #menu-grid {
        grid-size: 2;
        grid-gutter: 1 2;
        height: auto;
    }

    #quit-row {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    #btn-quit {
        width: 20;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("m", "models", "Models"),
        ("c", "configure", "Configure"),
        ("b", "benchmark", "Benchmark"),
        ("s", "sysinfo", "System"),
    ]

    def __init__(self):
        super().__init__()
        self.selected_model: str = ""
        self.sys_info = None
        self.binaries: dict = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-container"):
            with Vertical(id="hero"):
                yield Static(LOGO, id="logo")
                yield Static(self.SUB_TITLE, id="tagline")

                with Grid(id="metrics-grid"):
                    yield MetricCard("GPU", "detecting...")
                    yield MetricCard("RAM", "detecting...")
                    yield MetricCard("Binaries", "detecting...")

                with Grid(id="menu-grid"):
                    yield MenuCard("Models\nBrowse & download weights",
                                   id="btn-models", variant="primary")
                    yield MenuCard("Configure\nSet flags & launch server",
                                   id="btn-config", variant="primary")
                    yield MenuCard("Benchmark\nFind optimal config",
                                   id="btn-bench", variant="primary")
                    yield MenuCard("System Info\nGPU, RAM & binaries",
                                   id="btn-sysinfo", variant="primary")

                with Container(id="quit-row"):
                    yield Button("Quit", id="btn-quit", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self._detect_system()

    def _detect_system(self) -> None:
        import system
        self.sys_info = system.detect()
        self.binaries = system.find_binaries()

        parts = []
        for g in self.sys_info.gpus:
            vram = f"{g.vram_mb // 1024}GB" if g.vram_mb else "?"
            parts.append(f"{g.name} ({vram})")
        gpu_str = " + ".join(parts) if parts else "None detected"

        ram_str = f"{self.sys_info.ram_total_mb // 1024}GB / {self.sys_info.ram_free_mb // 1024}GB free"

        if self.binaries:
            bin_str = f"{len(self.binaries)} found"
        else:
            bin_str = "Not found"

        cards = self.query("MetricCard").nodes
        if len(cards) >= 3:
            cards[0].value = gpu_str
            cards[1].value = ram_str
            cards[2].value = bin_str

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-models":
            self.action_models()
        elif event.button.id == "btn-config":
            self.action_configure()
        elif event.button.id == "btn-bench":
            self.action_benchmark()
        elif event.button.id == "btn-sysinfo":
            self.action_sysinfo()
        elif event.button.id == "btn-quit":
            self.exit()

    def action_models(self) -> None:
        from .models_screen import ModelsScreen
        def on_model_selected(path: str | None) -> None:
            if path:
                self.selected_model = path
                self.notify(f"Selected: {Path(path).name}")
        self.push_screen(ModelsScreen(), on_model_selected)

    def action_configure(self) -> None:
        from .config_screen import ConfigScreen
        def on_config_result(result: dict | None) -> None:
            if not result:
                return
            if result["action"] == "launch":
                server_path = self.binaries.get("llama-server")
                if not server_path:
                    self.notify("llama-server not found!", severity="error")
                    return
                from .server_screen import ServerScreen
                self.push_screen(ServerScreen(str(server_path), result["args"]))
            elif result["action"] == "cli":
                cli_path = self.binaries.get("llama-cli")
                if not cli_path:
                    self.notify("llama-cli not found!", severity="error")
                    return
                import subprocess, os
                subprocess.Popen([str(cli_path)] + result["args"],
                                 env={**os.environ, "GGML_CUDA_NO_PINNED": "1"})

        self.push_screen(ConfigScreen(model_path=self.selected_model), on_config_result)

    def action_benchmark(self) -> None:
        from .bench_screen import BenchScreen
        cli = str(self.binaries.get("llama-cli", ""))
        self.push_screen(BenchScreen(cli_path=cli, model_path=self.selected_model))

    def action_sysinfo(self) -> None:
        from .sysinfo_screen import SysInfoScreen
        self.push_screen(SysInfoScreen(self.sys_info, self.binaries))


def main():
    app = BellsApp()
    app.run()


if __name__ == "__main__":
    main()
