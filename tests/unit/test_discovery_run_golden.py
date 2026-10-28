"""Regression test for the brand-new-DB open race in ``discovery.__main__.main()``.

The rabmo.7 mock-run golden (which pinned every logged line and the notes bytes of
four success / four failure shapes) is gone with the closed epic; the one outcome a
reader depends on here is vibe-quant-mt0qw: on a brand-new DB file, main()'s first
open and the heartbeat thread's first open must not race their WAL switch into
"database is locked".
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_BASE = [
    "--mock",
    "--seed",
    "7",
    "--population-size",
    "6",
    "--max-generations",
    "2",
    "--max-workers",
    "-1",
    "--start-date",
    "2024-01-01",
    "--end-date",
    "2025-01-01",
    "--timeframe",
    "4h",
    "--symbols",
    "BTCUSDT",
]


def test_main_on_a_brand_new_db_never_hits_database_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main()'s first open and the heartbeat thread's first open of a brand-new DB file
    must not race their WAL switch into "database is locked" (vibe-quant-mt0qw)."""
    import vibe_quant.discovery.__main__ as dm

    codes = []
    for i in range(10):
        db = tmp_path / f"fresh{i}.db"
        monkeypatch.setattr(sys, "argv", ["prog", *_BASE, "--run-id", "1", "--db", str(db)])
        codes.append(dm.main())

    out, err = capsys.readouterr()
    assert "database is locked" not in out + err
    assert out.count("Run 1 not found") == 10  # every run reached its real outcome
    assert codes == [1] * 10
