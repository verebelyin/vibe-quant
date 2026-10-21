"""Fast tier: the replay equals NT screening fill-for-fill on real 1m data.

The parity tests run the same chromosome through ``NTBacktestFn`` and ``FastBacktestFn``
on two real BTCUSDT days and require identical fills and identical metrics (no
tolerance). ``scripts/agents/fast_tier_parity.py`` runs the same comparison over seeded
random genomes and longer windows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.discovery.operators import ConditionType as C
from vibe_quant.discovery.operators import Direction, StrategyChromosome, StrategyGene
from vibe_quant.fast_tier import FastTierUnsupported
from vibe_quant.fast_tier.fills import bar_quarter_sizes, bar_ticks

REPO = Path(__file__).resolve().parents[2]
CATALOG_BARS = REPO / "data" / "catalog" / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
METRICS = ("total_trades", "total_return", "sharpe_ratio", "profit_factor", "max_drawdown")
START, END = "2026-03-02", "2026-03-04"

needs_catalog = pytest.mark.skipif(not CATALOG_BARS.is_dir(), reason="needs the BTCUSDT 1m catalog")


def test_bar_quarter_sizes_matches_nt_rounding() -> None:
    # volume 10.0 at size increment 0.001 (raw scale 1e16 cancels out)
    assert bar_quarter_sizes(10_000, 1) == (2_500, 2_500)
    assert bar_quarter_sizes(10_003, 2) == (2_500, 2_502)  # quarter rounded down to the increment
    assert bar_quarter_sizes(1, 5) == (5, 5)  # never below the increment
    assert bar_quarter_sizes(12, 5) == (5, 5)  # 3 quarters exceed the volume -> close = increment


def test_bar_ticks_order_and_skips() -> None:
    q, cs = 7, 9
    # |H-O| = 5 < |L-O| = 10: high first
    assert bar_ticks(100, 105, 90, 95, 99, q, cs) == [
        (100, "O", q), (105, "H", q), (90, "L", q), (95, "C", cs),
    ]
    # tie -> low first; an open equal to the last price is not a tick
    assert bar_ticks(100, 110, 90, 90, 100, q, cs) == [(90, "L", q), (110, "H", q), (90, "C", cs)]
    # a close equal to the running last price is not a tick
    assert bar_ticks(100, 104, 100, 104, 100, q, cs) == [(104, "H", q)]
    # a flat bar at the last price processes nothing
    assert bar_ticks(100, 100, 100, 100, 100, q, cs) == []
    # first bar of the run: the open is always processed
    assert bar_ticks(100, 100, 100, 100, None, q, cs) == [(100, "O", q)]


def _genomes() -> list[StrategyChromosome]:
    stoch = {"period_k": 14.0, "period_d": 3.0}
    return [
        # many short trades: tight SL/TP so stops, at-limit TP draws and exits all occur
        StrategyChromosome(
            entry_genes=[StrategyGene("CCI", {"period": 20.0}, C.CROSSES_ABOVE, 100.0)],
            exit_genes=[StrategyGene("STOCH", stoch, C.GT, 95.0)],
            stop_loss_pct=0.5, take_profit_pct=0.6, direction=Direction.LONG,
        ),
        # both directions with a compute_fn (pandas) indicator on the exit side
        StrategyChromosome(
            entry_genes=[StrategyGene("RSI", {"period": 14.0}, C.CROSSES_BELOW, 30.0)],
            exit_genes=[StrategyGene("WILLR", {"period": 14.0}, C.GT, -20.0)],
            stop_loss_pct=0.8, take_profit_pct=1.0, direction=Direction.BOTH,
            short_entry_genes=[StrategyGene("RSI", {"period": 14.0}, C.CROSSES_ABOVE, 70.0)],
            short_exit_genes=[StrategyGene("WILLR", {"period": 14.0}, C.LT, -80.0)],
        ),
    ]


def _fills(positions: list[Any]) -> list[tuple[Any, ...]]:
    return [
        (int(f.ts_event), f.order_side.name, f.order_type.name, str(f.last_qty), str(f.last_px),
         str(f.commission))
        for pos in sorted(positions, key=lambda p: int(p.ts_opened))
        for f in pos.events
    ]


@pytest.fixture
def scratch_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(REPO)
    assert os.environ["VIBE_QUANT_DB"] != str(REPO / "data" / "state" / "vibe_quant.db")


@needs_catalog
@pytest.mark.parametrize("liquidation", ["off", "isolated"])
@pytest.mark.parametrize("genome_index", [0, 1])
def test_fast_tier_matches_nt_fill_for_fill(
    scratch_env: None, monkeypatch: pytest.MonkeyPatch, genome_index: int, liquidation: str
) -> None:
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.screening import nt_runner
    from vibe_quant.validation.extraction import all_positions

    captured: dict[str, list[Any]] = {}
    original = nt_runner.NTScreeningRunner._extract_metrics

    def grab(self: Any, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        captured["positions"] = all_positions(engine)
        return original(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "_extract_metrics", grab)
    kw: dict[str, Any] = {}
    if liquidation != "off":
        kw["liquidation"] = LiquidationSettings(mode="isolated", leverage=10)
    chromosome = _genomes()[genome_index]
    fast_fn = FastBacktestFn(["BTCUSDT"], "1m", START, END, **kw)
    fast = fast_fn._run_single(chromosome, START, END)
    nt = NTBacktestFn(["BTCUSDT"], "1m", START, END, **kw)._run_single(chromosome, START, END)

    assert nt["total_trades"] > 5  # the window exercises fills, not an empty run
    assert _fills(fast_fn.last_positions) == _fills(captured["positions"])
    assert {m: fast[m] for m in METRICS} == {m: nt[m] for m in METRICS}


def test_unsupported_shapes_raise_before_any_engine() -> None:
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

    chromosome = _genomes()[0]
    with pytest.raises(FastTierUnsupported, match="1m only"):
        FastBacktestFn(["BTCUSDT"], "4h", START, END)._run_single(chromosome, START, END)
    with pytest.raises(FastTierUnsupported, match="one symbol"):
        FastBacktestFn(["BTCUSDT", "ETHUSDT"], "1m", START, END)._run_single(chromosome, START, END)


def test_unsupported_becomes_an_error_row_through_call() -> None:
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

    row = FastBacktestFn(["BTCUSDT"], "4h", START, END)(_genomes()[0])
    assert "error" in row
    assert row["total_trades"] == 0
