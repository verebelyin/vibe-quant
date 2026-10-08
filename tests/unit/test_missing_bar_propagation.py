"""Missing bar types must surface through every runner path (vibe-quant-c8va4 fix round 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import (
    CatalogManager,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening.nt_runner import (
    MissingBarDataError,
    NTScreeningRunner,
    require_bars_in_window,
)

if TYPE_CHECKING:
    from pathlib import Path

_DAY = 86_400_000
_DAY0 = 1_704_067_200_000  # 2024-01-01T00:00Z


def _dsl(tf: str = "1h", extra: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": f"missing_bar_{tf}_{extra or 'x'}",
        "timeframe": tf,
        "indicators": {"ema": {"type": "EMA", "period": 20}},
        "entry_conditions": {"long": ["ema > 1000000"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    if extra:
        d["indicators"]["ema_d"] = {"type": "EMA", "period": 200, "timeframe": extra}
        d["additional_timeframes"] = [extra]
    return d


def test_runner_raises_on_empty_catalog(tmp_path: Path) -> None:
    runner = NTScreeningRunner(_dsl("1h"), ["BTCUSDT"], "2024-01-01", "2024-02-01", catalog_path=str(tmp_path))
    with pytest.raises(MissingBarDataError, match=r"BTCUSDT-PERP\.BINANCE-1-HOUR-LAST-EXTERNAL"):
        runner({})


def test_dsl_with_1d_additional_timeframe_validates_and_compiles() -> None:
    dsl = validate_strategy_dict(_dsl("4h", "1d"))
    assert "1d" in dsl.additional_timeframes
    StrategyCompiler().compile_to_module(dsl)


def test_1d_smoke_on_catalog_without_1d(tmp_path: Path) -> None:
    """A catalog holding only 4h bars: the 1d requirement fails loudly."""
    inst = create_instrument("BTCUSDT")
    cm = CatalogManager(tmp_path)
    cm.write_instrument(inst)
    cm.write_bars(_minute_bars(inst, "4h", 1440))
    runner = NTScreeningRunner(_dsl("4h", "1d"), ["BTCUSDT"], "2024-01-01", "2024-01-02", catalog_path=str(tmp_path))
    with pytest.raises(MissingBarDataError, match="1-DAY"):
        runner({})


def _minute_bars(inst: Any, interval: str, n: int) -> list[Any]:
    klines = [
        {"open_time": _DAY0 + i * 60_000, "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.5, "volume": 1.0, "close_time": _DAY0 + i * 60_000 + 59_999}
        for i in range(n)
    ]
    return klines_to_bars(klines, inst.id, get_bar_type("BTCUSDT", interval), inst.size_precision, inst.price_precision)


def test_partial_overlap_does_not_raise(tmp_path: Path) -> None:
    inst = create_instrument("BTCUSDT")
    cm = CatalogManager(tmp_path)
    cm.write_instrument(inst)
    cm.write_bars(_minute_bars(inst, "1m", 1440))  # 2024-01-01 only
    bt = "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
    require_bars_in_window(str(tmp_path), bt, "2023-12-01", "2024-03-01")  # window wider both sides
    with pytest.raises(MissingBarDataError):
        require_bars_in_window(str(tmp_path), bt, "2025-01-01", "2025-02-01")


def test_discovery_all_missing_aborts_with_bar_type(tmp_path: Path) -> None:
    from tests.unit.test_discovery_pipeline import _make_config

    def bt(chrom: Any) -> dict[str, Any]:
        from vibe_quant.discovery.genome import chromosome_to_dsl

        dsl = chromosome_to_dsl(chrom)
        dsl["timeframe"] = "4h"
        r = NTScreeningRunner(dsl, ["BTCUSDT"], "2024-01-01", "2024-02-01", catalog_path=str(tmp_path))
        r({})
        return {}

    pipe = DiscoveryPipeline(_make_config(population_size=6, max_generations=2, max_workers=1), bt)
    # DataUnavailableError (MissingBarDataError) now propagates raw out of run()
    # instead of being wrapped as "all N evaluations failed" (vibe-quant-5x7r8).
    with pytest.raises(MissingBarDataError, match=r"BTCUSDT-PERP\.BINANCE-4-HOUR-LAST-EXTERNAL"):
        pipe.run()
