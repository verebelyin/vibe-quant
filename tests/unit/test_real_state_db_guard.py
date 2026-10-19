"""Tests can never open the real state DB, even through a symlinked data/state."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vibe_quant.db import StateManager
from vibe_quant.db import connection as conn_mod


def test_state_manager_refuses_the_real_db_path() -> None:
    sm = StateManager(conn_mod._REAL_DB)  # lazy: the first query opens the connection
    with pytest.raises(conn_mod.RealStateDBUnderTestError):
        sm.conn.execute("SELECT 1")


def test_symlinked_data_state_dir_is_refused_before_any_write(tmp_path: Path) -> None:
    # The 2026-10-19 escape: a scratch copy whose data/state links to main's.
    scratch_state = tmp_path / "scratch" / "data" / "state"
    scratch_state.parent.mkdir(parents=True)
    os.symlink(conn_mod._REAL_DB.parent, scratch_state)
    before = conn_mod._REAL_DB.stat().st_mtime_ns if conn_mod._REAL_DB.exists() else None
    with pytest.raises(conn_mod.RealStateDBUnderTestError):
        conn_mod.get_connection(scratch_state / "vibe_quant.db")
    after = conn_mod._REAL_DB.stat().st_mtime_ns if conn_mod._REAL_DB.exists() else None
    assert before == after


def test_tmp_db_still_opens(tmp_path: Path) -> None:
    sm = StateManager(tmp_path / "vibe_quant.db")
    assert sm.conn.execute("SELECT 1").fetchone()[0] == 1


def test_real_db_is_this_checkouts_state_db() -> None:
    repo = Path(__file__).resolve().parents[2]
    assert repo / "data" / "state" / "vibe_quant.db" == conn_mod._REAL_DB


def test_under_test_needs_only_one_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """pytest imported OR the conftest sentinel: a job subprocess a test launched has only the env."""
    import sys

    monkeypatch.delenv("VIBE_QUANT_PYTEST", raising=False)
    assert conn_mod._under_test() is True  # pytest is imported here
    monkeypatch.delitem(sys.modules, "pytest")
    assert conn_mod._under_test() is False
    monkeypatch.setenv("VIBE_QUANT_PYTEST", "1")
    assert conn_mod._under_test() is True
