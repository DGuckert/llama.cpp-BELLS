"""Server configuration screen — flag picker with descriptions."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (Button, Checkbox, Footer, Header, Input, Label,
                             Rule, Select, Static, Switch)

FLAG_DEFS = [
    {
        "flag": "--cpu-moe",
        "label": "CPU MoE (mmap)",
        "desc": "Expert layers on CPU via mmap. Best when model exceeds RAM.",
        "type": "bool",
        "group": "offload",
        "exclusive": ["--cpu-moe-pinned"],
    },
    {
        "flag": "--cpu-moe-pinned",
        "label": "CPU MoE (pinned)",
        "desc": "Expert layers on CPU, pinned memory. Faster when model fits RAM.",
        "type": "bool",
        "group": "offload",
        "exclusive": ["--cpu-moe"],
    },
    {
        "flag": "-ngl",
        "label": "GPU layers",
        "desc": "Layers to offload to GPU. 99 = all non-expert.",
        "type": "int",
        "default": "99",
        "group": "offload",
    },
    {
        "flag": "-ot",
        "label": "Override tensor",
        "desc": "Partial offload: blk\\.(0|1)\\.ffn_.*_exps=CUDA0",
        "type": "str",
        "group": "offload",
    },
    {
        "flag": "--n-cpu-moe",
        "label": "N CPU MoE layers",
        "desc": "First N expert layers on CPU, rest on GPU.",
        "type": "int",
        "group": "offload",
    },
    {
        "flag": "--bells-slots",
        "label": "Cache slots",
        "desc": "Experts resident per layer in VRAM. More = higher hit rate.",
        "type": "int",
        "group": "bells",
    },
    {
        "flag": "--bells-l2-slots",
        "label": "L2 slots",
        "desc": "Victim cache slots on second GPU. Needs multi-GPU.",
        "type": "int",
        "group": "bells",
    },
    {
        "flag": "--bells-refresh",
        "label": "Refresh rate",
        "desc": "Observe 1-in-N MoE layers per token. Needs --pin-experts.",
        "type": "int",
        "group": "bells",
    },
    {
        "flag": "--bells-split",
        "label": "Split",
        "desc": "Experts per token on GPU. Rest run on CPU concurrently.",
        "type": "int",
        "group": "bells",
    },
    {
        "flag": "--bells-passive",
        "label": "Passive mode",
        "desc": "Allocate cache but skip routing. Measures fixed overhead.",
        "type": "bool",
        "group": "bells",
    },
    {
        "flag": "-c",
        "label": "Context window",
        "desc": "Maximum context length in tokens.",
        "type": "int",
        "default": "4096",
        "group": "general",
    },
    {
        "flag": "-fa",
        "label": "Flash attention",
        "desc": "Enable flash attention for faster inference.",
        "type": "select",
        "options": ["on", "off", "auto"],
        "default": "on",
        "group": "general",
    },
    {
        "flag": "-t",
        "label": "Threads",
        "desc": "CPU threads for computation.",
        "type": "int",
        "default": "8",
        "group": "general",
    },
    {
        "flag": "--port",
        "label": "Port",
        "desc": "HTTP port for llama-server.",
        "type": "int",
        "default": "8080",
        "group": "server",
    },
    {
        "flag": "--host",
        "label": "Host",
        "desc": "Bind address for llama-server.",
        "type": "str",
        "default": "127.0.0.1",
        "group": "server",
    },
    {
        "flag": "-ctk",
        "label": "KV cache K",
        "desc": "K quantization. q8_0 = lossless, q4_0 saves VRAM.",
        "type": "select",
        "options": ["f16", "q8_0", "q4_0"],
        "default": "f16",
        "group": "general",
    },
    {
        "flag": "-ctv",
        "label": "KV cache V",
        "desc": "V quantization. Same trade-off as K.",
        "type": "select",
        "options": ["f16", "q8_0", "q4_0"],
        "default": "f16",
        "group": "general",
    },
    {
        "flag": "--override-kv",
        "label": "KV override",
        "desc": "e.g. general.expert_used_count=int:6",
        "type": "str",
        "group": "advanced",
    },
    {
        "flag": "--temp",
        "label": "Temperature",
        "desc": "Sampling temperature. Lower = more deterministic.",
        "type": "str",
        "default": "0.6",
        "group": "sampling",
    },
]

GROUPS = [
    ("offload", "Offloading"),
    ("bells", "BELLS Cache"),
    ("general", "General"),
    ("server", "Server"),
    ("sampling", "Sampling"),
    ("advanced", "Advanced"),
]


class FlagRow(Widget):
    """A single flag configuration row."""

    DEFAULT_CSS = """
    FlagRow {
        layout: horizontal;
        height: 3;
        padding: 0 1;
    }

    FlagRow:hover {
        background: $surface;
    }

    FlagRow .flag-label {
        width: 20;
        padding: 1 0 0 0;
        color: $text;
    }

    FlagRow .flag-input {
        width: 24;
    }

    FlagRow .flag-desc {
        width: 1fr;
        color: $text-muted;
        padding: 1 0 0 1;
    }
    """


class GroupHeader(Static):
    """Section header for flag groups."""

    DEFAULT_CSS = """
    GroupHeader {
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 2;
        width: 100%;
        margin: 1 0 0 0;
    }
    """


class ConfigScreen(Screen):
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+s", "save_config", "Save"),
        ("ctrl+r", "reset", "Reset"),
    ]

    CSS = """
    ConfigScreen {
        background: $background;
    }

    #config-header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    #config-title {
        text-style: bold;
        color: $accent;
    }

    #config-model {
        color: $text-muted;
    }

    #config-scroll {
        height: 1fr;
        padding: 0 1;
    }

    #cmd-bar {
        height: auto;
        dock: bottom;
        background: $surface;
        padding: 1 2;
        border-top: solid $primary;
    }

    #cmd-label {
        color: $text-muted;
        height: 1;
    }

    #cmd-text {
        color: $accent;
        height: auto;
    }

    #action-row {
        height: 3;
        dock: bottom;
        padding: 0 2;
        background: $panel;
        border-top: solid $primary;
    }

    #action-row Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(self, model_path: str = "", initial_flags: dict | None = None):
        super().__init__()
        self.model_path = model_path
        self.initial_flags = initial_flags or {}
        self.flag_values: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="config-header"):
            yield Static("Configure", id="config-title")
            model_display = Path(self.model_path).name if self.model_path else "No model selected"
            yield Static(f"Model: {model_display}", id="config-model")
        with VerticalScroll(id="config-scroll"):
            for group_id, group_title in GROUPS:
                group_flags = [f for f in FLAG_DEFS if f.get("group") == group_id]
                if not group_flags:
                    continue
                yield GroupHeader(f" {group_title} ")
                for fdef in group_flags:
                    with FlagRow():
                        yield Label(fdef["label"], classes="flag-label")
                        if fdef["type"] == "bool":
                            default = self.initial_flags.get(fdef["flag"], False)
                            yield Switch(value=bool(default), id=f"flag-{fdef['flag']}")
                        elif fdef["type"] == "select":
                            default = self.initial_flags.get(fdef["flag"],
                                                             fdef.get("default", ""))
                            options = [(o, o) for o in fdef["options"]]
                            yield Select(options, value=default,
                                         id=f"flag-{fdef['flag']}")
                        else:
                            default = str(self.initial_flags.get(fdef["flag"],
                                                                 fdef.get("default", "")))
                            yield Input(value=default, placeholder=fdef.get("default", ""),
                                        id=f"flag-{fdef['flag']}", classes="flag-input")
                        yield Label(fdef["desc"], classes="flag-desc")
        with Vertical(id="cmd-bar"):
            yield Static("COMMAND PREVIEW", id="cmd-label")
            yield Static("llama-server", id="cmd-text")
        with Horizontal(id="action-row"):
            yield Button("Launch Server", id="launch-btn", variant="success")
            yield Button("Copy Command", id="copy-btn", variant="primary")
            yield Button("Run CLI", id="cli-btn", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        self._update_preview()

    def on_switch_changed(self, event) -> None:
        self._update_preview()

    def on_input_changed(self, event) -> None:
        self._update_preview()

    def on_select_changed(self, event) -> None:
        self._update_preview()

    def build_args(self, mode: str = "server") -> list[str]:
        args = []
        if self.model_path:
            args += ["-m", self.model_path]

        for fdef in FLAG_DEFS:
            widget_id = f"flag-{fdef['flag']}"
            try:
                widget = self.query_one(f"#{widget_id}")
            except Exception:
                continue

            if fdef["type"] == "bool":
                if hasattr(widget, "value") and widget.value:
                    args.append(fdef["flag"])
            elif fdef["type"] == "select":
                val = widget.value if hasattr(widget, "value") else ""
                if val and val != fdef.get("default", ""):
                    args += [fdef["flag"], str(val)]
                elif val and fdef["flag"] == "-fa":
                    args += [fdef["flag"], str(val)]
            else:
                val = widget.value.strip() if hasattr(widget, "value") else ""
                if val and val != fdef.get("default", ""):
                    args += [fdef["flag"], val]
                elif val and fdef["flag"] in ("-c", "-ngl", "-t"):
                    args += [fdef["flag"], val]

        return args

    def _update_preview(self) -> None:
        args = self.build_args()
        cmd = "llama-server " + " ".join(args)
        try:
            self.query_one("#cmd-text", Static).update(cmd)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch-btn":
            args = self.build_args("server")
            self.dismiss({"action": "launch", "args": args})
        elif event.button.id == "copy-btn":
            args = self.build_args("server")
            cmd = "llama-server " + " ".join(args)
            try:
                import pyperclip
                pyperclip.copy(cmd)
            except ImportError:
                pass
            self.notify(f"Copied: {cmd}")
        elif event.button.id == "cli-btn":
            args = self.build_args("cli")
            self.dismiss({"action": "cli", "args": args})

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_reset(self) -> None:
        for fdef in FLAG_DEFS:
            widget_id = f"flag-{fdef['flag']}"
            try:
                widget = self.query_one(f"#{widget_id}")
                if fdef["type"] == "bool":
                    widget.value = False
                elif fdef["type"] == "select":
                    widget.value = fdef.get("default", fdef["options"][0])
                else:
                    widget.value = fdef.get("default", "")
            except Exception:
                pass
        self._update_preview()
