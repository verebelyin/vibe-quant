"""Concurrent swarm-check gates split the cores (gate_slots + swarm_check wiring)."""

from __future__ import annotations

import importlib.util
import os
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


gate_slots = _load("gate_slots")
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


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_lone_gate_on_a_quiet_host_keeps_xdist_auto(tmp_path: Path) -> None:
    with gate_slots.gate_slot(tmp_path) as running:
        assert running == 1
        assert swarm_check.xdist_auto_workers({}, 1.0, 10, running) is None


def test_three_live_gates_plus_me_split_ten_cores_two_each(
    tmp_path: Path, sleepers: list[subprocess.Popen[bytes]]
) -> None:
    for _ in range(3):
        (tmp_path / str(_live_pid(sleepers))).write_text("")
    with gate_slots.gate_slot(tmp_path) as running:
        assert running == 4
        assert swarm_check.xdist_auto_workers({}, 1.0, 10, running) == "2"  # 10 // 4 = 2
    assert swarm_check.xdist_auto_workers({}, 1.0, 10, 2) == "5"
    assert swarm_check.xdist_auto_workers({}, 1.0, 10, 3) == "3"
    assert swarm_check.xdist_auto_workers({}, 1.0, 10, 8) == "2"  # floor of 2


def test_load_rule_and_share_take_the_smaller() -> None:
    # 2 gates on 10 cores at load 12: load rule = max(5, 2, -2) = 5, share = 5
    assert swarm_check.xdist_auto_workers({}, 12.0, 10, 2) == "5"
    # 3 gates: share 3 beats the load rule's 5
    assert swarm_check.xdist_auto_workers({}, 12.0, 10, 3) == "3"
    # a lone gate on a loaded host keeps the old load rule
    assert swarm_check.xdist_auto_workers({}, 6.4, 10, 1) == "5"


def test_swarm_check_workers_env_wins_over_the_share() -> None:
    assert swarm_check.xdist_auto_workers({"SWARM_CHECK_WORKERS": "7"}, 1.0, 10, 4) == "7"


def test_a_dead_pid_slot_is_not_counted_and_is_cleaned(tmp_path: Path) -> None:
    dead = tmp_path / str(_dead_pid())
    dead.write_text("")
    (tmp_path / "notes.txt").write_text("not a slot")
    with gate_slots.gate_slot(tmp_path) as running:
        assert running == 1
    assert not dead.exists()
    assert (tmp_path / "notes.txt").exists()


def test_slot_is_removed_when_the_body_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), gate_slots.gate_slot(tmp_path):
        assert (tmp_path / str(os.getpid())).exists()
        raise RuntimeError("pytest blew up")
    assert list(tmp_path.iterdir()) == []


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


def test_the_slot_is_released_when_pytest_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_plan(monkeypatch)
    slots = tmp_path / "slots"

    def run(cmd: Any, cwd: Path, env: Any = None, timeout: int = 0) -> tuple[int, str]:
        raise OSError("pytest launch failed")

    with pytest.raises(OSError):
        _run_step(tmp_path, slots, run)
    assert list(slots.iterdir()) == []


def test_sigterm_removes_the_slot_of_a_running_gate(tmp_path: Path) -> None:
    code = (
        "import sys, time\nfrom pathlib import Path\nimport gate_slots\n"
        "with gate_slots.gate_slot(Path(sys.argv[1])):\n    print('ready', flush=True)\n    time.sleep(30)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_AGENTS))
    p = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, env=env, text=True
    )
    try:
        assert p.stdout is not None
        assert p.stdout.readline().strip() == "ready"
        assert len(list(tmp_path.iterdir())) == 1
        p.terminate()
        p.wait(timeout=5)
    finally:
        p.kill()
        p.wait()
    assert list(tmp_path.iterdir()) == []


def test_the_sigterm_handler_is_restored_after_the_slot(tmp_path: Path) -> None:
    import signal

    before = signal.getsignal(signal.SIGTERM)
    with gate_slots.gate_slot(tmp_path):
        assert signal.getsignal(signal.SIGTERM) != before
    assert signal.getsignal(signal.SIGTERM) == before


def test_an_old_slot_of_a_live_pid_is_stale_and_removed(
    tmp_path: Path, sleepers: list[subprocess.Popen[bytes]]
) -> None:
    reused = tmp_path / str(_live_pid(sleepers))  # a pid an unrelated process owns now
    reused.write_text("")
    old = reused.stat().st_mtime - swarm_check.TEST_TIMEOUT - 1
    os.utime(reused, (old, old))
    with gate_slots.gate_slot(tmp_path, swarm_check.TEST_TIMEOUT) as running:
        assert running == 1
    assert not reused.exists()


def test_a_fresh_slot_of_a_live_pid_still_counts(
    tmp_path: Path, sleepers: list[subprocess.Popen[bytes]]
) -> None:
    (tmp_path / str(_live_pid(sleepers))).write_text("")
    with gate_slots.gate_slot(tmp_path, swarm_check.TEST_TIMEOUT) as running:
        assert running == 2


def test_stale_age_is_the_test_step_timeout() -> None:
    assert gate_slots.STALE_AFTER == swarm_check.TEST_TIMEOUT


def test_an_explicit_worker_count_keeps_its_value_but_peers_count_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_plan(monkeypatch)
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "7")
    slots = tmp_path / "slots"
    seen: list[tuple[str | None, int]] = []

    def run(cmd: Any, cwd: Path, env: Any = None, timeout: int = 0) -> tuple[int, str]:
        seen.append(((env or {}).get("PYTEST_XDIST_AUTO_NUM_WORKERS"), len(list(slots.iterdir()))))
        return 0, "1 passed in 0.1s"

    _run_step(tmp_path, slots, run)
    assert seen == [(None, 1)]  # slot held; env not overridden (the real env's 7 passes through)
