"""Concurrent swarm-check gates split the cores (gate_slots + swarm_check wiring)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"slots_{name}", _AGENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


swarm_check = _load("swarm_check")


@pytest.fixture
def sleepers() -> Iterator[list[subprocess.Popen[bytes]]]:
    procs: list[subprocess.Popen[bytes]] = []
    yield procs
    for p in procs:
        p.kill()
        p.wait()


def _live_pid(procs: list[subprocess.Popen[bytes]]) -> int:
    p = subprocess.Popen(["sleep", "30"])
    procs.append(p)
    return p.pid


def _fake_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = swarm_check.TestPlan(
        commands=[(["tests/unit/test_x.py"], False, "")],
        explicit=frozenset(),
        full=False,
        notes=[],
        files=[],
    )
    monkeypatch.setattr(swarm_check, "plan_tests", lambda cfg, state: plan)
    monkeypatch.setattr(
        swarm_check, "slow_tests_check", lambda *a, **k: swarm_check.Check("slow-tests", "SKIP", "")
    )
    monkeypatch.setattr(swarm_check.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(swarm_check.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.delenv("SWARM_CHECK_WORKERS", raising=False)


def _run_step(tmp_path: Path, slots: Path | None, run: Any) -> Any:
    cfg = swarm_check.Config(worktree=tmp_path, main_root=tmp_path, slots_dir=slots)
    state = swarm_check.GitState(changed=["scripts/x.py"], untracked=[])
    return swarm_check._run_tests(cfg, state, {"pytest": ["pytest"]}, run, {})


def test_the_test_step_decides_workers_from_the_gates_running_then(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleepers: list[subprocess.Popen[bytes]]
) -> None:
    _fake_plan(monkeypatch)
    slots = tmp_path / "slots"
    slots.mkdir()
    seen: list[tuple[str | None, int]] = []

    def run(cmd: Any, cwd: Path, env: Any = None, timeout: int = 0) -> tuple[int, str]:
        seen.append(((env or {}).get("PYTEST_XDIST_AUTO_NUM_WORKERS"), len(list(slots.iterdir()))))
        return 0, "1 passed in 0.1s"

    assert _run_step(tmp_path, slots, run).check.status == "PASS"
    assert seen == [(None, 1)]  # lone gate: xdist auto, my slot held during pytest
    assert list(slots.iterdir()) == []

    for _ in range(3):
        (slots / str(_live_pid(sleepers))).write_text("")
    seen.clear()
    _run_step(tmp_path, slots, run)
    assert seen == [("2", 4)]  # 10 cores // 4 gates
