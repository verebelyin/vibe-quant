"""Validation installs screening's compute_fn precompute, and it changes no number.

Without it a recursive compute_fn indicator (PMAX, full history) is recomputed over the
whole buffer on every bar: O(n^2), a 15m PMAX run over 2023..2026 took ~13 h. The
precompute replays the indicator once; the generated feed writes the stored value only
on a bar whose ts_init and close match, so every result field must equal the live per-bar
path (``VIBE_QUANT_PTA_PRECOMPUTE=0``). Real BTCUSDT catalog bars: skipped when absent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.runner import ValidationRunner

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
_BAR_DIR = CATALOG / "data" / "bar"
needs_catalog = pytest.mark.skipif(
    not (_BAR_DIR / "BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL").is_dir()
    or not (_BAR_DIR / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").is_dir(),
    reason="needs the BTCUSDT 15m and 1m catalog",
)

# The Pine script's defaults (period 10, multiplier 3, EMA 10), always in the market.
PMAX_DSL: dict[str, Any] = {
    "name": "val_pta_pre_pmax",
    "timeframe": "15m",
    "indicators": {"pm": {"type": "PMAX", "period": 10, "ma_period": 10, "multiplier": 3.0,
                          "kind": "ema"}},
    "entry_conditions": {"long": ["pm.mavg crosses_above pm"],
                         "short": ["pm.mavg crosses_below pm"]},
    "stop_loss": {"type": "fixed_pct", "percent": 50.0},
    "take_profit": {"type": "fixed_pct", "percent": 100.0},
    "position_mode": "reverse",
}


def _validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precompute: bool) -> tuple[Any, list[dict[str, object]]]:
    monkeypatch.setenv("VIBE_QUANT_PTA_PRECOMPUTE", "1" if precompute else "0")
    db = tmp_path / f"state-{precompute}.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        strategy_id=state.create_strategy(name="val_pta_pre_pmax", dsl_config=PMAX_DSL),
        run_mode="validation", symbols=["BTCUSDT"], timeframe="15m",
        start_date="2026-01-01", end_date="2026-01-15", parameters={},
    )
    state.close()
    runner = ValidationRunner(
        db_path=db, logs_path=tmp_path / "logs", catalog_path=CATALOG, funding_archive_path=ARCHIVE
    )
    try:
        return runner.run(run_id=run_id), runner.last_pta_precompute
    finally:
        runner.close()


@pytest.mark.slow(reason="two real NT validation runs (precompute on, off), ~2.5 s")
@needs_catalog
def test_validation_precompute_feeds_every_bar_and_changes_no_result_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on, report = _validate(tmp_path, monkeypatch, precompute=True)
    off, off_report = _validate(tmp_path, monkeypatch, precompute=False)

    assert off_report == []
    (entry,) = report
    assert entry["instrument"] == "BTCUSDT-PERP.BINANCE"
    assert entry["tf"] == "15m"
    assert entry["failed_at"] is None
    assert entry["fed"] == entry["n"]
    assert int(str(entry["n"])) > 1_300  # 14 days of 15m bars, every one fed

    assert on.total_trades >= 5
    skip = {"run_id", "execution_time_seconds"}
    for f in dataclasses.fields(on):
        if f.name not in skip:
            assert getattr(on, f.name) == getattr(off, f.name), f.name
