"""The shared CPU sampler's summary and process labels (perf numbers carry their load)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

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


def test_top_parses_comma_decimal_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    class R:
        stdout = (
            "  12  4,5  2048 /x/Python -m vibe_quant discovery\n"
            "   7  0,1  1024 /usr/bin/caffeinate -ims\n"
        )

    monkeypatch.setattr(cpu_log.subprocess, "run", lambda *a, **k: R())
    assert cpu_log._top(1) == [{"pid": 12, "cpu": 4.5, "rss_mb": 2, "cmd": "py -m vibe_quant"}]


def test_label_names_the_python_module_or_script() -> None:
    assert cpu_log.label("/x/Python -m vibe_quant discovery --run-id 3") == "py -m vibe_quant"
    assert cpu_log.label("/x/python scripts/research/power/budget.py discover") == "py budget.py"
    assert cpu_log.label("/usr/bin/caffeinate -ims") == "caffeinate"
    spawn = "/x/Python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=14)"
    assert cpu_log.label(spawn) == "py mp-worker"
    assert cpu_log.label("/x/Python -u -c import sys;exec(eval(sys.stdin.readline()))") == (
        "py xdist-worker"
    )
