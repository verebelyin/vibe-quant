"""The funding caches are keyed on the archive's ABSOLUTE path.

``FundingCalculator()`` defaults to the relative ``data/archive/raw_data.db``; a cache
keyed on that string served the first root's funding rates to a later run whose cwd
holds a different archive, shifting a backtest's net return and Sharpe.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vibe_quant.validation import funding
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from collections.abc import Iterator

_REL = Path("data/archive/raw_data.db")
_T0_MS = 1_704_067_200_000  # 2024-01-01 00:00 UTC


def _root(base: Path, name: str, rate: float) -> Path:
    root = base / name
    (root / "data" / "archive").mkdir(parents=True)
    conn = sqlite3.connect(str(root / _REL))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    conn.execute(
        "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
        ("BTCUSDT", _T0_MS + 3, rate),
    )
    conn.commit()
    conn.close()
    return root


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    funding.clear_rate_cache()
    yield
    funding.clear_rate_cache()


def test_relative_archive_from_two_cwds_gets_each_roots_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _root(tmp_path, "a", 0.0004)
    b = _root(tmp_path, "b", 0.0009)
    monkeypatch.chdir(a)
    assert FundingCalculator(_REL).settlements("BTCUSDT")[1] == (0.0004,)
    monkeypatch.chdir(b)
    assert FundingCalculator(_REL).settlements("BTCUSDT")[1] == (0.0009,)
    assert FundingCalculator().settlements("BTCUSDT")[1] == (0.0009,)  # default path, same cwd
    monkeypatch.chdir(a)
    assert FundingCalculator().settlements("BTCUSDT")[1] == (0.0004,)


def test_calculator_stays_bound_to_the_root_it_was_built_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _root(tmp_path, "a", 0.0004)
    b = _root(tmp_path, "b", 0.0009)
    monkeypatch.chdir(a)
    calc = FundingCalculator(_REL)
    monkeypatch.chdir(b)
    assert calc.settlements("BTCUSDT")[1] == (0.0004,)


def test_absolute_and_relative_spellings_share_one_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _root(tmp_path, "a", 0.0004)
    monkeypatch.chdir(a)
    FundingCalculator(_REL).settlements("BTCUSDT")
    FundingCalculator(a.resolve() / _REL).settlements("BTCUSDT")
    assert list(funding._RATE_CACHE) == [(str(a.resolve() / _REL), "BTCUSDT")]


def test_real_catalog_backtest_does_not_shift_the_scratch_root_holdout_sharpe() -> None:
    """The leaker->victim pair that exposed the stale key, in one fresh process.

    ``test_runner_unavailable_mtm...`` backtests on the real archive's funding; the W3 power
    test then chdirs to a scratch root whose archive has the same relative path and pins its
    holdout Sharpe (3.469234306697298). Before the fix the pair gave 3.4707467952952955.
    """
    import subprocess
    import sys

    here = Path(__file__).parent
    pair = [
        f"{here}/test_mtm_daily_returns.py::test_runner_unavailable_mtm_after_success_returns_none",
        f"{here}/test_power_layer2.py::test_worker_chain_scores_through_the_shared_w3_cell_fns",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *pair, "-q", "-p", "no:randomly", "-p", "no:xdist"],
        capture_output=True, text=True, check=False, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-1500:]
