"""swarm_check's per-process cache of the tree-wide scans (graph, pytest_plugins, loaders)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "swarm_check.py"
if "swarm_check" in sys.modules:
    swarm_check = sys.modules["swarm_check"]
else:
    _spec = importlib.util.spec_from_file_location("swarm_check", _SCRIPT)
    assert _spec is not None and _spec.loader is not None
    swarm_check = importlib.util.module_from_spec(_spec)
    sys.modules["swarm_check"] = swarm_check
    _spec.loader.exec_module(swarm_check)


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "vibe_quant").mkdir()
    (tmp_path / "vibe_quant" / "a.py").write_text("import vibe_quant.b\n")
    (tmp_path / "vibe_quant" / "b.py").write_text("B = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("import vibe_quant.a\n")
    return tmp_path


def _count_calls(monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    calls: list[int] = []
    real: Callable[..., Any] = getattr(swarm_check, name)

    def counted(*args: Any, **kw: Any) -> Any:
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(swarm_check, name, counted)
    return calls


def test_import_graph_is_parsed_once_until_a_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _tree(tmp_path)
    calls = _count_calls(monkeypatch, "_build_import_graph")
    first = swarm_check.build_import_graph(wt)
    first["vibe_quant/a.py"].add("mutated")  # callers get a copy
    again = swarm_check.build_import_graph(wt)
    assert len(calls) == 1
    assert again["vibe_quant/a.py"] == {"vibe_quant.b"}
    (wt / "vibe_quant" / "a.py").write_text("import vibe_quant.b\nimport vibe_quant.c\nx = 1\n")
    (wt / "vibe_quant" / "c.py").write_text("C = 1\n")
    changed = swarm_check.build_import_graph(wt)
    assert len(calls) == 2
    assert changed["vibe_quant/a.py"] == {"vibe_quant.b", "vibe_quant.c"}


def test_plugins_and_loader_scans_are_cached_and_see_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _tree(tmp_path)
    loader_calls = _count_calls(monkeypatch, "_loader_call_sites")
    plugin_calls = _count_calls(monkeypatch, "_check_pytest_plugins")
    assert swarm_check.loader_call_sites(wt) == []
    assert swarm_check.loader_call_sites(wt) == []
    swarm_check.check_pytest_plugins(wt)
    swarm_check.check_pytest_plugins(wt)
    assert (len(loader_calls), len(plugin_calls)) == (1, 1)
    (wt / "vibe_quant" / "b.py").write_text("import importlib\nimportlib.import_module('x')\n")
    assert swarm_check.loader_call_sites(wt) == ["vibe_quant/b.py"]
    (wt / "tests" / "test_a.py").write_text("pytest_plugins = ['x']\n")
    for _ in range(2):  # a FullSuite is cached and re-raised
        with pytest.raises(swarm_check.FullSuite):
            swarm_check.check_pytest_plugins(wt)
    assert len(plugin_calls) == 2
