"""Tests for NTPurgedKFoldRunner adapter (bd-xrli).

Covers DSL+symbols+timeframe resolution, bar-timestamp loading from a
synthetic parquet catalog, and the index→date mapping that lets the
PurgedKFold splitter feed NT screening backtests. The real NT backtest
itself is monkeypatched — its behavior is covered by screening tests.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _seed_db(tmp_path: Path, *, timeframe: str = "4h") -> Path:
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dsl_config TEXT NOT NULL
        );
        CREATE TABLE backtest_runs (
            id INTEGER PRIMARY KEY,
            strategy_id INTEGER,
            symbols TEXT NOT NULL,
            timeframe TEXT
        );
        """
    )
    dsl = {"name": "stub", "timeframe": timeframe, "indicators": {}}
    conn.execute(
        "INSERT INTO strategies (id, name, dsl_config) VALUES (?,?,?)",
        (1, "stub", json.dumps(dsl)),
    )
    conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, symbols, timeframe) VALUES (?,?,?,?)",
        (42, 1, json.dumps(["BTCUSDT"]), timeframe),
    )
    conn.commit()
    conn.close()
    return db


def _seed_catalog(tmp_path: Path, *, n_bars: int = 100, timeframe_dir: str = "4-HOUR") -> Path:
    """Write a tiny parquet bar file with monotonic ts_event values."""
    import pandas as pd

    catalog = tmp_path / "catalog"
    bar_dir = catalog / "data" / "bar" / f"BTCUSDT-PERP.BINANCE-{timeframe_dir}-LAST-EXTERNAL"
    bar_dir.mkdir(parents=True)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    # Use 4h spacing so the date math is easy to reason about.
    ts_event = [
        int((base + timedelta(hours=4 * i)).timestamp() * 1e9) for i in range(n_bars)
    ]
    df = pd.DataFrame({"ts_event": ts_event, "ts_init": ts_event})
    df.to_parquet(bar_dir / "bars.parquet", index=False)
    return catalog


@pytest.fixture
def stub_env(tmp_path: Path) -> tuple[Path, Path]:
    return _seed_db(tmp_path), _seed_catalog(tmp_path)


def test_resolves_run_and_loads_bars(stub_env: tuple[Path, Path]) -> None:
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db, catalog = stub_env
    runner = NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=catalog)
    assert runner.n_samples == 100
    assert runner._symbols == ["BTCUSDT"]  # noqa: SLF001
    assert runner._timeframe == "4h"  # noqa: SLF001
    assert runner._dsl_dict["name"] == "stub"  # noqa: SLF001


def test_missing_run_raises(stub_env: tuple[Path, Path]) -> None:
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db, catalog = stub_env
    with pytest.raises(ValueError, match="backtest_runs.id=999"):
        NTPurgedKFoldRunner(run_id=999, db_path=db, catalog_path=catalog)


def test_unsupported_timeframe_raises(tmp_path: Path) -> None:
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db = _seed_db(tmp_path, timeframe="2h")
    catalog = _seed_catalog(tmp_path)  # parquet under 4-HOUR; lookup uses 2h → mismatch
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=catalog)


def test_missing_catalog_raises(tmp_path: Path) -> None:
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db = _seed_db(tmp_path)
    empty_catalog = tmp_path / "empty"
    (empty_catalog / "data" / "bar").mkdir(parents=True)
    with pytest.raises(ValueError, match="No parquet files"):
        NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=empty_catalog)


def test_index_range_to_dates_uses_min_max(stub_env: tuple[Path, Path]) -> None:
    """train indices may be non-contiguous (purged k-fold). We must
    collapse to (min, max) so NT can backtest a single date span."""
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db, catalog = stub_env
    runner = NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=catalog)

    # Simulated purged k-fold: train = [0..19] ∪ [40..99], test = [20..39].
    train = list(range(0, 20)) + list(range(40, 100))
    start, end = runner._index_range_to_dates(train)  # noqa: SLF001
    assert start == "2024-01-01"  # bar 0 is 2024-01-01 00:00 UTC
    # bar 99 is 2024-01-01 + 99*4h = 396h ≈ 16.5 days → 2024-01-17
    assert end == "2024-01-17"


def test_index_range_to_dates_empty_raises(stub_env: tuple[Path, Path]) -> None:
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db, catalog = stub_env
    runner = NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=catalog)
    with pytest.raises(ValueError, match="empty index list"):
        runner._index_range_to_dates([])  # noqa: SLF001


def test_run_calls_backtest_twice_with_correct_dates(
    stub_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify train + test backtests both fire and FoldResult fields wire."""
    from vibe_quant.overfitting import nt_cv_runner

    db, catalog = stub_env
    runner = nt_cv_runner.NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=catalog)

    calls: list[tuple[str, str]] = []

    def fake_backtest(self: object, start: str, end: str) -> tuple[float, float]:  # noqa: ARG001
        calls.append((start, end))
        # Train returns higher (overfit), test returns lower — sanity for diagnostics.
        if len(calls) == 1:
            return 2.0, 0.5
        return 1.0, 0.2

    monkeypatch.setattr(nt_cv_runner.NTPurgedKFoldRunner, "_backtest", fake_backtest)

    train = list(range(0, 20)) + list(range(40, 100))
    test = list(range(20, 40))
    fold = runner.run(train, test)

    assert len(calls) == 2
    assert fold.train_size == len(train)
    assert fold.test_size == len(test)
    assert fold.train_sharpe == 2.0
    assert fold.test_sharpe == 1.0
    assert fold.train_return == 0.5
    assert fold.test_return == 0.2


def test_runs_through_purgedkfoldcv(
    stub_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: PurgedKFoldCV.run() iterates folds via the runner."""
    from vibe_quant.overfitting import nt_cv_runner
    from vibe_quant.overfitting.purged_kfold import CVConfig, PurgedKFoldCV

    db, catalog = stub_env
    runner = nt_cv_runner.NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=catalog)

    # Stub the NT call so the test stays fast and offline.
    monkeypatch.setattr(
        nt_cv_runner.NTPurgedKFoldRunner,
        "_backtest",
        lambda _self, _s, _e: (1.5, 0.1),
    )

    cv = PurgedKFoldCV(
        config=CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0),
        min_oos_sharpe=0.5,
        max_oos_sharpe_std=2.0,
    )
    result = cv.run(n_samples=runner.n_samples, runner=runner)
    assert len(result.fold_results) == 3
    assert result.mean_oos_sharpe == 1.5
    assert result.is_robust is True
