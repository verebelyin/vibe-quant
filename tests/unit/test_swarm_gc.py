"""scripts/agents/swarm_gc.py deletes only idle jobs' scratch dirs and old gate logs."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "swarm_gc.py"
_spec = importlib.util.spec_from_file_location("swarm_gc", _SCRIPT)
assert _spec is not None and _spec.loader is not None
swarm_gc = importlib.util.module_from_spec(_spec)
sys.modules["swarm_gc"] = swarm_gc
_spec.loader.exec_module(swarm_gc)

_OLD = time.time() - 30 * 86400


def _file(path: Path, size: int = 10, old: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if old:
        os.utime(path, (_OLD, _OLD))
    return path


def test_gc_deletes_idle_scratch_and_old_gate_logs_and_keeps_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idle_scratch = _file(tmp_path / "20261001-old" / "power" / "scratch" / "run1" / "state.db")
    _file(tmp_path / "20261001-old" / "power" / "scratch" / "inner" / "scratch" / "x.parquet")
    raw = _file(tmp_path / "20261001-old" / "raw" / "klines.csv", size=swarm_gc.REPORT_MIN_BYTES)
    ledger = _file(tmp_path / "20261001-old" / "ledger.md")
    active_scratch = _file(tmp_path / "20261104-live" / "scratch" / "state.db")
    _file(tmp_path / "20261104-live" / "ledger.md", old=False)  # touched today: job is active
    old_log = _file(tmp_path / "_gate-logs" / "wt-abc-20261001T000000Z.log")
    new_log = _file(tmp_path / "_gate-logs" / "wt-abc-20261105T000000Z.log", old=False)
    golden = _file(tmp_path / "_goldens" / "gate_golden.json")

    assert swarm_gc.main(["--root", str(tmp_path)]) == 0  # dry run deletes nothing
    out = capsys.readouterr().out
    assert "would delete" in out and "1 scratch dirs, 1 gate logs" in out
    assert "20261001-old/raw/klines.csv" in out  # big evidence is reported, not deleted
    assert idle_scratch.exists() and old_log.exists()

    assert swarm_gc.main(["--root", str(tmp_path), "--apply"]) == 0
    assert not (tmp_path / "20261001-old" / "power" / "scratch").exists()
    assert not old_log.exists()
    for kept in (raw, ledger, active_scratch, new_log, golden):
        assert kept.exists(), kept
