"""System info display screen."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, Label, ProgressBar, Static


class InfoRow(Widget):
    """Key-value info row."""

    DEFAULT_CSS = """
    InfoRow {
        layout: horizontal;
        height: 1;
    }

    InfoRow .key {
        width: 14;
        color: $text-muted;
    }

    InfoRow .val {
        width: 1fr;
        color: $text;
    }
    """

    def __init__(self, key: str, value: str) -> None:
        super().__init__()
        self._key = key
        self._val = value

    def compose(self) -> ComposeResult:
        yield Static(self._key, classes="key")
        yield Static(self._val, classes="val")


class InfoCard(Widget):
    """Bordered card with a title."""

    DEFAULT_CSS = """
    InfoCard {
        width: 100%;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $surface;
    }

    InfoCard > .card-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self._children = []

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="card-title")


class GpuCard(Widget):
    """GPU info card with VRAM progress bar."""

    DEFAULT_CSS = """
    GpuCard {
        width: 100%;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $surface;
    }

    GpuCard > .gpu-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    GpuCard ProgressBar {
        margin: 1 0 0 0;
    }
    """

    def __init__(self, gpu) -> None:
        super().__init__()
        self.gpu = gpu

    def compose(self) -> ComposeResult:
        g = self.gpu
        yield Static(f"GPU {g.index} — {g.name}", classes="gpu-title")
        yield InfoRow("Vendor", g.vendor)
        if g.vram_mb:
            used = g.vram_mb - g.vram_free_mb
            pct = (used / g.vram_mb * 100) if g.vram_mb else 0
            yield InfoRow("VRAM", f"{g.vram_mb} MB total, {g.vram_free_mb} MB free ({pct:.0f}% used)")
            yield ProgressBar(total=100, show_percentage=True, id=f"vram-bar-{g.index}")
        else:
            yield InfoRow("VRAM", "Unknown")
        if g.driver:
            yield InfoRow("Driver", g.driver)
        if g.compute:
            yield InfoRow("Compute", g.compute)

    def on_mount(self) -> None:
        g = self.gpu
        if g.vram_mb:
            used = g.vram_mb - g.vram_free_mb
            pct = (used / g.vram_mb * 100) if g.vram_mb else 0
            try:
                self.query_one(f"#vram-bar-{g.index}", ProgressBar).update(progress=pct)
            except Exception:
                pass


class SysInfoScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    CSS = """
    SysInfoScreen {
        background: $background;
    }

    #sysinfo-header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    #sysinfo-title {
        text-style: bold;
        color: $accent;
    }

    #sysinfo-scroll {
        height: 1fr;
        padding: 1 2;
    }

    .bin-row {
        height: 1;
        layout: horizontal;
    }

    .bin-name {
        width: 20;
        color: $accent;
        text-style: bold;
    }

    .bin-path {
        width: 1fr;
        color: $text-muted;
    }
    """

    def __init__(self, sys_info, binaries):
        super().__init__()
        self.sys_info = sys_info
        self.binaries = binaries

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="sysinfo-header"):
            yield Static("System Information", id="sysinfo-title")
        with VerticalScroll(id="sysinfo-scroll"):
            with InfoCard("Platform"):
                if self.sys_info:
                    yield InfoRow("OS", f"{self.sys_info.os} {self.sys_info.arch}")
                    yield InfoRow("CPU", f"{self.sys_info.cpu}")
                    yield InfoRow("Cores", f"{self.sys_info.cpu_cores}")
                    yield InfoRow("RAM",
                                  f"{self.sys_info.ram_total_mb // 1024} GB total, "
                                  f"{self.sys_info.ram_free_mb // 1024} GB free")
                    yield InfoRow("CUDA", "Yes" if self.sys_info.has_cuda else "No")
                    yield InfoRow("Vulkan", "Yes" if self.sys_info.has_vulkan else "No")

            if self.sys_info and self.sys_info.gpus:
                for g in self.sys_info.gpus:
                    yield GpuCard(g)
            else:
                with InfoCard("GPUs"):
                    yield Static("No GPUs detected", classes="val")

            with InfoCard("Binaries"):
                if self.binaries:
                    for name, path in self.binaries.items():
                        with Horizontal(classes="bin-row"):
                            yield Static(name, classes="bin-name")
                            yield Static(str(path), classes="bin-path")
                else:
                    yield Static("No llama.cpp binaries found. Run the installer.")
        yield Footer()
