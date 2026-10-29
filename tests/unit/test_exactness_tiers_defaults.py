"""Defaults and cell-pool plumbing of ``scripts/agents/exactness_tiers.py``.

The tiers golden lives OUTSIDE git (user decision): ``<main checkout>/data/swarm/_goldens/
tiers.json``. A linked worktree must resolve it through git-common-dir, never from a
hard-coded checkout path, and a bare ``--check``/``--capture`` must use it.

The cells are independent (each validation works in its own TemporaryDirectory copy of the
state DB), so ``--jobs N`` runs them in a spawn process pool while the parent prints and
compares in CELLS order: the output stays bit-identical to the serial loop.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _AGENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tiers = _load("exactness_tiers")

_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "tiers",
    "GIT_AUTHOR_EMAIL": "tiers@test",
    "GIT_COMMITTER_NAME": "tiers",
    "GIT_COMMITTER_EMAIL": "tiers@test",
}


def _git_env() -> dict[str, str]:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV_OVERRIDES)
    return env


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
        env=_git_env(),
    )
    return out.stdout


def _git_pair(tmp_path: Path) -> tuple[Path, Path]:
    """(main checkout, linked worktree of it): git-common-dir of the worktree is main's .git."""
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(main)], check=True, env=_git_env())
    (main / "keep.txt").write_text("x\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "feature", str(wt))
    return main, wt


def test_default_golden_is_under_the_main_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main, wt = _git_pair(tmp_path)
    monkeypatch.setattr(tiers, "REPO_ROOT", wt)
    expected = main / "data" / "swarm" / "_goldens" / "tiers.json"
    assert tiers.default_golden() == expected  # from the linked worktree
    assert tiers.default_golden(main) == expected


def test_bare_check_uses_the_main_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main, wt = _git_pair(tmp_path)
    monkeypatch.setattr(tiers, "REPO_ROOT", wt)
    assert tiers.main(["--check"]) == 2
    out = capsys.readouterr().out
    assert out.strip() == f"no capture file at {main / 'data/swarm/_goldens/tiers.json'}"


def test_explicit_check_path_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _main, wt = _git_pair(tmp_path)
    monkeypatch.setattr(tiers, "REPO_ROOT", wt)
    explicit = tmp_path / "elsewhere.json"
    explicit.write_text(json.dumps({"cells": []}))
    assert tiers.main(["--check", str(explicit)]) == 0  # no cells: nothing runs
    assert "no capture file" not in capsys.readouterr().out


def test_jobs_default_is_capped_at_four() -> None:
    assert min(4, os.cpu_count() or 1) == tiers.JOBS_DEFAULT


def test_jobs_flag_reaches_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Path, int]] = []
    monkeypatch.setattr(tiers, "check", lambda path, jobs: seen.append((path, jobs)) or 0)
    golden = tmp_path / "g.json"
    assert tiers.main(["--check", str(golden)]) == 0
    assert tiers.main(["--check", str(golden), "--jobs", "1"]) == 0
    assert seen == [(golden, tiers.JOBS_DEFAULT), (golden, 1)]


def test_jobs_zero_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        tiers.main(["--check", str(tmp_path / "g.json"), "--jobs", "0"])
    assert exc.value.code == 2


class _FakeExecutor:
    """Stands in for ProcessPoolExecutor: records how it was built and what it was given."""

    seen: dict[str, Any] = {}

    def __init__(self, max_workers: int, mp_context: Any) -> None:
        _FakeExecutor.seen["init"] = (max_workers, mp_context)

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def map(self, fn: Any, tasks: Any) -> Any:
        tasks = list(tasks)
        _FakeExecutor.seen["map"] = (fn, tasks)
        return iter([{"i": i} for i in range(len(tasks))])


def test_pool_runs_cells_in_order_with_the_spawn_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiers, "ProcessPoolExecutor", _FakeExecutor)
    specs = [{"name": "C1"}, {"name": "C2"}, {"name": "C3"}]
    got = list(tiers.execute_cells(specs, tmp_path / "db.sqlite", jobs=5))
    assert got == [{"i": 0}, {"i": 1}, {"i": 2}]  # pool.map order == CELLS order
    max_workers, context = cast("tuple[int, Any]", _FakeExecutor.seen["init"])
    assert max_workers == 3  # min(jobs, cells)
    assert context.get_start_method() == "spawn"
    fn, tasks = cast("tuple[Any, Any]", _FakeExecutor.seen["map"])
    assert fn is tiers._execute_task
    assert tasks == [(spec, str(tmp_path / "db.sqlite")) for spec in specs]


def test_jobs_one_keeps_the_serial_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tiers, "_execute", lambda spec, db: {"cell": spec["name"], "db": str(db)}
    )

    def no_pool(**kwargs: object) -> None:
        raise AssertionError("a pool started for --jobs 1")

    monkeypatch.setattr(tiers, "ProcessPoolExecutor", no_pool)
    got = list(tiers.execute_cells([{"name": "A"}, {"name": "B"}], tmp_path / "db.sqlite", 1))
    assert got == [
        {"cell": "A", "db": str(tmp_path / "db.sqlite")},
        {"cell": "B", "db": str(tmp_path / "db.sqlite")},
    ]


def _result(trades: int) -> dict[str, Any]:
    return {
        "screening": {
            "metrics": {"total_trades": trades, "liquidation_count": 0},
            "margin_refusals": 0,
        },
        "validation": {
            "result": {"total_trades": trades, "notes": {}, "liquidation_events": []}
        },
    }


def test_capture_uses_the_pool_and_stores_cells_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        tiers,
        "CELLS",
        (
            tiers.Cell("Z-first", ("BTCUSDT",), "2024-01-01", "2024-02-01"),
            tiers.Cell("A-second", ("BTCUSDT",), "2024-01-01", "2024-02-01"),
        ),
    )
    monkeypatch.setattr(tiers, "_base_dsl", lambda: {"timeframe": "4h"})
    monkeypatch.setattr(tiers, "validate_strategy_dict", lambda dsl: dsl)  # the DSL is not the test
    monkeypatch.setattr(tiers, "_find_state_db", lambda: tmp_path / "db.sqlite")
    seen: dict[str, Any] = {}

    def fake(specs: Any, db: Path, jobs: int) -> Any:
        seen["specs"] = [spec["name"] for spec in specs]
        seen["jobs"] = jobs
        return iter([_result(1), _result(2)])

    monkeypatch.setattr(tiers, "execute_cells", fake)
    out = tmp_path / "tiers.json"
    assert tiers.capture(out, jobs=3) == 0
    assert seen == {"specs": ["Z-first", "A-second"], "jobs": 3}
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Z-first: counts=")
    assert lines[1].startswith("A-second: counts=")
    cells = json.loads(out.read_text())["cells"]
    assert [cell["name"] for cell in cells] == ["Z-first", "A-second"]
    assert [cell["golden"]["screening"]["metrics"]["total_trades"] for cell in cells] == [1, 2]
