"""OI_CHANGE end to end: a strategy reading archived open interest fills in both NT tiers.

The bars come from the ZEC 1m catalog (skipped without it); the positioning data is a tmp
archive whose ``sum_open_interest`` oscillates with a 2-hour cycle, so the 1-hour
``OI_CHANGE`` swings about +/-2% and ``x < -0.002`` fires several times a day. Bit parity of
the generated stream with the as-of lookup is proven on synthetic bars in
``test_fast_tier_asof.py`` (no catalog needed); this file proves the engines reach a fill.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.archive import MetricsFileMeta, MetricsRow, RawDataArchive
from vibe_quant.dsl import aux_data

if TYPE_CHECKING:
    from collections.abc import Iterator

START, END = "2025-06-02", "2025-06-03"
pytestmark = pytest.mark.skipif(
    not Path("data/catalog/data/bar/ZECUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").is_dir(),
    reason="ZEC 1m catalog data not available",
)


@pytest.fixture
def clean() -> Iterator[None]:
    yield
    aux_data.configure(None, None)


def _oscillating_archive(path: Path) -> Path:
    """ZECUSDT metrics 2025-06-01..03: ``sum_open_interest = 1e6 * (1 + 0.01 sin(2 pi k / 24))``
    at 5-minute row k (a 24-row cycle, so 12 rows apart the ratio swings about +/-2%)."""
    archive = RawDataArchive(path)
    try:
        for offset in range(3):
            day = date(2025, 6, 1) + timedelta(days=offset)
            t0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
            rows: list[MetricsRow] = []
            for k in range(288):
                oi = 1e6 * (1 + 0.01 * math.sin(2 * math.pi * k / 24))
                ms = int((t0 + k * timedelta(minutes=5)).timestamp()) * 1000
                rows.append((ms, oi, None, None, None, None, None))
            meta = MetricsFileMeta(
                filename=f"ZECUSDT-metrics-{day}.zip",
                md5=f"md5-{day}",
                sha256="0" * 64,
                size=1,
                vision_last_modified=None,
                csv_rows=len(rows),
                status="ok",
                detail=None,
            )
            assert archive.insert_metrics_file("ZECUSDT", day, meta, rows) == 288
    finally:
        archive.close()
    return path


def _oi_dsl() -> dict[str, Any]:
    return {
        "name": "oi_change_smoke",
        "timeframe": "1m",
        "indicators": {"oi": {"type": "OI_CHANGE", "period": 12}},
        "entry_conditions": {"long": ["oi < -0.002"]},
        "exit_conditions": {"long": ["oi > 0.002"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def test_oi_change_trades_screening_and_validation(tmp_path: Path, clean: None) -> None:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import _dsl_to_dict
    from vibe_quant.validation.runner import ValidationRunner

    archive = _oscillating_archive(tmp_path / "metrics.db")
    dsl = _oi_dsl()

    aux_data.configure(archive, None)
    screening = NTScreeningRunner(
        _dsl_to_dict(validate_strategy_dict(dsl)),
        ["ZECUSDT"],
        START,
        END,
        funding_archive_path=str(archive),
    )({})
    assert screening.total_trades > 0

    db = tmp_path / "state.db"
    state = StateManager(db)
    try:
        run_id = state.create_backtest_run(
            strategy_id=state.create_strategy(name="oi_change_smoke", dsl_config=dsl),
            run_mode="validation",
            symbols=["ZECUSDT"],
            timeframe="1m",
            start_date=START,
            end_date=END,
            parameters={},
        )
    finally:
        state.close()
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", funding_archive_path=archive)
    try:
        validation = runner.run(run_id=run_id)
    finally:
        runner.close()
    assert validation.total_trades > 0
