"""The shared CPU sampler's summary and process labels (perf numbers carry their load)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cpu_log", Path(__file__).resolve().parents[2] / "scripts" / "agents" / "cpu_log.py"
)
assert _SPEC and _SPEC.loader
cpu_log = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpu_log)


def test_summary_reports_window_load_ram_and_busiest_commands() -> None:
    lines = [
        {"t": "10:00:00", "load1": 2.0, "free_gb": 9.0, "top": [{"cmd": "py -m vibe_quant"}]},
        {"t": "10:01:00", "load1": 6.0, "free_gb": 5.5, "top": [{"cmd": "py -m vibe_quant"},
                                                                {"cmd": "py layer2.py"}]},
        {"t": "10:05:00", "load1": 9.0, "free_gb": 1.0, "top": [{"cmd": "py layer2.py"}]},
    ]
    got = cpu_log.summarize(lines, "10:00", "10:02")
    assert got == {"samples": 2, "load_mean": 4.0, "load_max": 6.0, "free_gb_min": 5.5,
                   "top_cmds": [("py -m vibe_quant", 2), ("py layer2.py", 1)]}
    assert cpu_log.summarize(lines, "11:00", None) == {"samples": 0}
