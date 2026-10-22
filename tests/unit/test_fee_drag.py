"""Fee-drag reporting on the discovery money path (bd vibe-quant-2gyk1.4, D3-4a).

Covers the three shared helpers (``gross_pnl``, ``fee_drag``,
``fees_pct_equity_per_year``), the shared JSON vector D3-4b's vitest asserts
against (``tests/fixtures/fee_drag_vectors.json``), the raw fee/gross/day
fields surviving window + symbol aggregation and the early-exit path, and the
two places a user sees the number: discovery champion entries and ``_reject``
rows. Fitness/scoring must be untouched -- the same ``adjusted_score`` with and
without the new bt-dict keys.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_screening_fill_mode import _END, _START, _dsl, _write_catalog
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult, _evaluate_single
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.metrics import fee_drag, fees_pct_equity_per_year, gross_pnl
from vibe_quant.screening.nt_runner import NTScreeningRunner

if TYPE_CHECKING:
    from collections.abc import Mapping

VECTORS = Path(__file__).resolve().parents[1] / "fixtures" / "fee_drag_vectors.json"


def _close(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)


def _gene(indicator: str = "RSI") -> StrategyGene:
    return StrategyGene(
        indicator_type=indicator,
        parameters={"period": 14.0},
        condition=ConditionType.CROSSES_ABOVE,
        threshold=30.0,
    )


def _chromosome() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_gene()],
        exit_genes=[_gene("EMA")],
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        direction=Direction.LONG,
    )


def _m(
    ret: float,
    *,
    trades: int = 50,
    fees: float = 0.0,
    gross: float = 0.0,
    balance: float = 1000.0,
    days: float = 30.0,
) -> dict[str, Any]:
    """Per-window/per-symbol bt dict with the new raw fee fields."""
    return {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": trades,
        "total_return": ret,
        "skewness": 0.0,
        "kurtosis": 3.0,
        "trade_returns": (ret,),
        "total_fees": fees,
        "gross_pnl": gross,
        "starting_balance": balance,
        "days": days,
    }


def _fit(fee_drag: float | None = None, fees_pct: float | None = None) -> FitnessResult:
    return FitnessResult(
        sharpe_ratio=2.0, max_drawdown=0.1, profit_factor=1.8, total_trades=150,
        total_return=0.4, complexity_penalty=0.0, overtrade_penalty=0.0, sl_tp_penalty=0.0,
        raw_score=0.6, adjusted_score=0.55, passed_filters=True, filter_results={},
        fee_drag=fee_drag, fees_pct_equity_yr=fees_pct,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_fee_drag_none_when_gross_nonpositive() -> None:
    """Never a division error: gross <= 0 or non-finite -> None."""
    assert fee_drag(250.0, 0.0) is None
    assert fee_drag(250.0, -1.0) is None
    assert fee_drag(250.0, float("nan")) is None
    assert fee_drag(250.0, float("inf")) is None
    assert fee_drag(250.0, float("-inf")) is None
    assert fee_drag(250.0, 1000.0) == 0.25


def test_fees_pct_equity_per_year() -> None:
    """Exact annualized fees; None when days/balance are non-positive."""
    assert fees_pct_equity_per_year(25.0, 10000.0, 365.0) == 0.0025
    assert fees_pct_equity_per_year(1825.0, 10000.0, 365.0) == pytest.approx(0.1825)
    assert fees_pct_equity_per_year(91250.0, 10000.0, 180.0) == pytest.approx(
        91250.0 / 10000.0 * 365.0 / 180.0
    )
    assert fees_pct_equity_per_year(25.0, 10000.0, 0.0) is None
    assert fees_pct_equity_per_year(25.0, 10000.0, -5.0) is None
    assert fees_pct_equity_per_year(25.0, 0.0, 180.0) is None
    assert fees_pct_equity_per_year(25.0, -1.0, 180.0) is None


def test_fee_drag_shared_vector() -> None:
    """The JSON vector D3-4b's vitest also reads: same inputs -> same outputs."""
    vectors: list[Mapping[str, float | None]] = json.loads(VECTORS.read_text())
    assert vectors, "shared vector must not be empty"
    for case in vectors:
        assert case["net"] is not None
        gross = gross_pnl(
            float(case["net"]),  # type: ignore[arg-type]
            float(case["fees"]),  # type: ignore[arg-type]
            float(case["slippage"]),  # type: ignore[arg-type]
            float(case["funding"]),  # type: ignore[arg-type]
        )
        assert _close(gross, case["expected_gross"])
        assert _close(
            fee_drag(float(case["fees"]), gross),  # type: ignore[arg-type]
            case["expected_fee_drag"],
        )
        assert _close(
            fees_pct_equity_per_year(
                float(case["fees"]),  # type: ignore[arg-type]
                float(case["starting_balance"]),  # type: ignore[arg-type]
                float(case["days"]),  # type: ignore[arg-type]
            ),
            case["expected_fees_pct_equity_yr"],
        )


# ---------------------------------------------------------------------------
# Fitness: score unchanged, fields present
# ---------------------------------------------------------------------------


def test_overtrade_penalty_unchanged_with_fee_fields() -> None:
    """The new bt-dict keys must not move the score."""
    chrom = _chromosome()
    base = {
        "sharpe_ratio": 1.5, "max_drawdown": 0.2, "profit_factor": 1.8,
        "total_trades": 100, "total_return": 0.3, "trade_returns": (0.003,) * 100,
    }  # fmt: skip
    with_fees = {
        **base,
        "total_fees": 50.0,
        "gross_pnl": 300.0,
        "starting_balance": 1000.0,
        "days": 180.0,
    }
    a = _evaluate_single(chrom, lambda c: dict(base), min_trades=50, timeframe="1h")
    b = _evaluate_single(chrom, lambda c: dict(with_fees), min_trades=50, timeframe="1h")
    assert a.adjusted_score == b.adjusted_score
    assert b.fee_drag == pytest.approx(50.0 / 300.0)
    assert b.fees_pct_equity_yr == pytest.approx(50.0 / 1000.0 * 365.0 / 180.0)
    assert a.fee_drag is None and a.fees_pct_equity_yr is None


def test_fee_fields_survive_eval_windows_3() -> None:
    """Sums over 3 windows; the drag is from the sums, not averaged ratios."""
    results = [
        _m(0.1, fees=10.0, gross=100.0, days=30.0),
        _m(0.2, fees=20.0, gross=200.0, days=31.0),
        _m(0.3, fees=30.0, gross=300.0, days=30.0),
    ]
    agg = NTBacktestFn._aggregate_multi_window(results, 0)
    assert agg["total_fees"] == 60.0
    assert agg["gross_pnl"] == 600.0
    assert agg["days"] == 91
    fit = _evaluate_single(_chromosome(), lambda c: dict(agg), min_trades=0, timeframe="1h")
    # Sums, not mean of (10/100, 20/200, 30/300) == 0.1.
    assert fit.fee_drag == pytest.approx(60.0 / 600.0)


def test_fee_fields_survive_early_exit() -> None:
    """A stopped run still reports the fees it actually paid."""
    results = [_m(0.1, fees=10.0, gross=100.0, days=30.0), _m(-0.05, fees=5.0, gross=80.0, days=31.0)]
    out = NTBacktestFn._early_exit_result(results, 1, low_trades=False)
    assert out["early_exit"] == 1
    assert out["total_fees"] == 15.0
    assert out["gross_pnl"] == 180.0
    assert out["days"] == 61


def test_worst_mode_sums_fees_over_symbols() -> None:
    """Worst-of-symbols sums the fee/gross flows; days are the shared window span."""
    results = [
        _m(0.3, fees=100.0, gross=400.0, days=90.0),
        _m(0.1, fees=150.0, gross=500.0, days=90.0),
    ]
    out = NTBacktestFn._aggregate_symbols(results, ["A", "B"])
    assert out["total_fees"] == 250.0
    assert out["gross_pnl"] == 900.0
    assert out["days"] == 90
    fit = _evaluate_single(_chromosome(), lambda c: dict(out), min_trades=0, timeframe="1h")
    assert fit.fee_drag == pytest.approx(250.0 / 900.0)
    assert fit.fees_pct_equity_yr == pytest.approx(250.0 / 2000.0 * 365.0 / 90.0)


def test_worst_mode_annualizes_fees_over_summed_symbol_equity() -> None:
    """Each worst-mode symbol runs on its OWN account, so the annual fee rate
    divides the summed fees by the SUMMED starting balances (N x balance)."""
    results = [
        _m(0.3, fees=10.0, gross=400.0, balance=1000.0, days=365.0),
        _m(0.1, fees=10.0, gross=500.0, balance=1000.0, days=365.0),
    ]
    out = NTBacktestFn._aggregate_symbols(results, ["A", "B"])
    assert out["total_fees"] == 20.0
    assert out["starting_balance"] == 2000.0  # 2 accounts x 1000
    assert out["days"] == 365
    fit = _evaluate_single(_chromosome(), lambda c: dict(out), min_trades=0, timeframe="1h")
    # 20 / 2000 * 365/365 == 0.01; dividing by ONE balance gives 0.02 (the old bug).
    assert fit.fees_pct_equity_yr == 0.01
    assert fit.fee_drag == pytest.approx(20.0 / 900.0)  # summed fees / summed gross


def test_worst_mode_annualizes_fees_over_unequal_symbol_balances() -> None:
    """Each symbol runs on its OWN account, so the annual fee rate divides the
    summed fees by the SUM of the symbols' starting balances -- here 1000 + 3000,
    not a single balance scaled by the symbol count."""
    results = [
        _m(0.3, fees=10.0, gross=400.0, balance=1000.0, days=365.0),
        _m(0.1, fees=10.0, gross=500.0, balance=3000.0, days=365.0),
    ]
    out = NTBacktestFn._aggregate_symbols(results, ["A", "B"])
    assert out["total_fees"] == 20.0
    assert out["starting_balance"] == 4000.0  # 1000 + 3000, NOT 2 x 1000
    assert out["days"] == 365
    fit = _evaluate_single(_chromosome(), lambda c: dict(out), min_trades=0, timeframe="1h")
    # 20 / 4000 * 365/365 == 0.005; a single-balance approximation (2 x 1000)
    # would divide by 2000 and report 0.01.
    assert fit.fees_pct_equity_yr == 0.005
    assert fit.fee_drag == pytest.approx(20.0 / 900.0)  # summed fees / summed gross


def test_portfolio_mode_annualizes_fees_over_shared_account() -> None:
    """Portfolio mode is one shared account: the same totals -> one balance."""
    bt = {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 50, "total_return": 0.3, "trade_returns": (0.3,),
        "total_fees": 20.0, "gross_pnl": 900.0, "starting_balance": 1000.0, "days": 365,
    }  # fmt: skip
    fit = _evaluate_single(_chromosome(), lambda c: dict(bt), min_trades=0, timeframe="1h")
    assert fit.fees_pct_equity_yr == 0.02  # 20 / 1000 * 365/365
    assert fit.fee_drag == pytest.approx(20.0 / 900.0)


# ---------------------------------------------------------------------------
# Rejection rows and champion entries
# ---------------------------------------------------------------------------


def test_rejection_rows_carry_fee_drag() -> None:
    cfg = DiscoveryConfig(
        population_size=4, max_generations=2, elite_count=1, top_k=2,
        min_trades=1, symbols=["BTC/USDT"], timeframe="1h",
        start_date="2024-01-01", end_date="2024-02-01", max_workers=1,
    )  # fmt: skip
    pipe = DiscoveryPipeline(cfg, lambda chrom: {})
    pipe._reject(_chromosome(), _fit(fee_drag=0.2, fees_pct=0.05), "guardrails", ["nope"])
    entry = pipe._guardrail_rejections[0]
    assert entry["stage"] == "guardrails"
    assert entry["fee_drag"] == 0.2
    assert entry["fees_pct_equity_yr"] == 0.05


def test_champion_entry_carries_fee_drag() -> None:
    from vibe_quant.discovery.__main__ import _champion_entry

    fitness = _fit(fee_drag=0.2, fees_pct=0.05)
    entry = _champion_entry(
        _chromosome(), fitness, timeframe="1h", dsr=None, buy_hold=None
    )
    assert entry["fee_drag"] == 0.2
    assert entry["fees_pct_equity_yr"] == 0.05
    assert entry["return_pct"] == fitness.total_return


# ---------------------------------------------------------------------------
# Independent ratio validity (non-finite inputs void only what they feed)
# ---------------------------------------------------------------------------

_NAN = float("nan")
_INF = float("inf")


def _fields(**kw: float) -> tuple[float | None, float | None]:
    from vibe_quant.discovery.fitness import _fee_drag_fields

    bt: dict[str, float | int] = {
        "total_fees": 50.0, "gross_pnl": 300.0, "starting_balance": 1000.0, "days": 180.0,
    }
    bt.update(kw)
    return _fee_drag_fields(bt)


def test_fee_ratios_are_judged_independently() -> None:
    annual = 50.0 / 1000.0 * 365.0 / 180.0
    assert _fields() == (pytest.approx(50.0 / 300.0), pytest.approx(annual))
    # Bad gross voids only fee_drag: annual stays valid.
    assert _fields(gross_pnl=_NAN) == (None, pytest.approx(annual))
    assert _fields(gross_pnl=_INF) == (None, pytest.approx(annual))
    assert _fields(gross_pnl=-5.0) == (None, pytest.approx(annual))
    # Bad balance/days void only the annualized rate: drag stays valid.
    assert _fields(starting_balance=_NAN) == (pytest.approx(50.0 / 300.0), None)
    assert _fields(starting_balance=_INF) == (pytest.approx(50.0 / 300.0), None)
    assert _fields(days=_NAN) == (pytest.approx(50.0 / 300.0), None)
    assert _fields(days=0) == (pytest.approx(50.0 / 300.0), None)
    # Non-finite fees void both (a NaN numerator would otherwise leak as NaN).
    assert _fields(total_fees=_NAN) == (None, None)
    assert _fields(total_fees=_INF) == (None, None)
    # No fee fields at all (mock / old backtest): nothing to report.
    from vibe_quant.discovery.fitness import _fee_drag_fields

    assert _fee_drag_fields({"sharpe_ratio": 1.0}) == (None, None)


def test_fee_ratios_independent_through_evaluate_single() -> None:
    base = {
        "sharpe_ratio": 1.5, "max_drawdown": 0.2, "profit_factor": 1.8,
        "total_trades": 100, "total_return": 0.3, "trade_returns": (0.003,) * 100,
        "total_fees": 50.0, "gross_pnl": _NAN, "starting_balance": 1000.0, "days": 180.0,
    }  # fmt: skip
    fit = _evaluate_single(_chromosome(), lambda c: dict(base), min_trades=50, timeframe="1h")
    assert fit.fee_drag is None
    assert fit.fees_pct_equity_yr == 0.10138888888888889


# ---------------------------------------------------------------------------
# Score bit-identity: fee fields never reach fitness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sharpe", "dd", "pf", "trades", "ret"),
    [
        (1.5, 0.2, 1.8, 100, 0.3),
        (0.4, 0.05, 1.1, 60, 0.02),
        (2.2, 0.35, 3.0, 400, 1.4),
        (1.0, 0.1, 1.5, 10, 0.5),  # below min_trades -> hard-gated to 0
        (1.0, 0.1, 1.5, 100, -0.1),  # negative return -> hard-gated to 0
    ],
)
def test_score_bit_identical_with_and_without_fee_fields(
    sharpe: float, dd: float, pf: float, trades: int, ret: float
) -> None:
    base = {
        "sharpe_ratio": sharpe, "max_drawdown": dd, "profit_factor": pf,
        "total_trades": trades, "total_return": ret, "trade_returns": (ret / trades,) * trades,
    }  # fmt: skip
    loud = {
        **base, "total_fees": 9e9, "gross_pnl": -1e9, "starting_balance": 1.0, "days": 1,
    }
    a = _evaluate_single(_chromosome(), lambda c: dict(base), min_trades=50, timeframe="1h")
    b = _evaluate_single(_chromosome(), lambda c: dict(loud), min_trades=50, timeframe="1h")
    for field in (
        "sharpe_ratio", "max_drawdown", "profit_factor", "total_trades", "total_return",
        "raw_score", "adjusted_score", "complexity_penalty", "overtrade_penalty",
        "sl_tp_penalty", "passed_filters",
    ):
        assert getattr(a, field) == getattr(b, field), field


# ---------------------------------------------------------------------------
# Hard-gated candidates still report the fees they paid
# ---------------------------------------------------------------------------


def test_trade_gated_fitness_still_reports_fee_drag() -> None:
    """Zero score (too few trades / negative return) must not hide the cost."""
    for trades, ret in ((10, 0.5), (100, -0.1)):
        bt = {
            "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": trades, "total_return": ret, "trade_returns": (0.0,) * trades,
            "total_fees": 40.0, "gross_pnl": 80.0, "starting_balance": 1000.0, "days": 100,
        }  # fmt: skip
        fit = _evaluate_single(_chromosome(), lambda c, b=bt: dict(b), min_trades=50, timeframe="1h")
        assert fit.adjusted_score == 0.0
        assert fit.fee_drag == 0.5
        assert fit.fees_pct_equity_yr == 40.0 / 1000.0 * 365.0 / 100.0


def test_window_trade_gate_aggregate_keeps_fee_fields() -> None:
    """The per-window trade-gate failure aggregate sums the fees actually paid."""
    results = [
        _m(0.1, trades=1, fees=10.0, gross=100.0, days=30.0),
        _m(0.2, trades=1, fees=20.0, gross=200.0, days=31.0),
    ]
    agg = NTBacktestFn._aggregate_multi_window(results, min_trades=100)
    assert agg["sharpe_ratio"] == -1.0  # the gate branch, not the normal aggregate
    assert agg["total_fees"] == 30.0
    assert agg["gross_pnl"] == 300.0
    assert agg["starting_balance"] == 1000.0
    assert agg["days"] == 61
    fit = _evaluate_single(_chromosome(), lambda c: dict(agg), min_trades=100, timeframe="1h")
    assert fit.adjusted_score == 0.0
    assert fit.fee_drag == pytest.approx(30.0 / 300.0)


# ---------------------------------------------------------------------------
# Real screening run: runner -> _run_single -> NTBacktestFn -> fitness
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_catalog(tmp_path_factory.mktemp("fee_drag_catalog"))


class _CatalogFn(NTBacktestFn):
    """NTBacktestFn on the synthetic catalog; everything else is the real code."""

    catalog_root = ""

    def _make_runner(
        self, dsl_dict: dict[str, Any], start_date: str, end_date: str,
        symbols: list[str] | None = None,
    ) -> NTScreeningRunner:
        return NTScreeningRunner(
            dsl_dict=dsl_dict,
            symbols=self.symbols if symbols is None else symbols,
            start_date=start_date,
            end_date=end_date,
            fill_ticks=self.fill_ticks,
            launch=LaunchSettings(fee_schedule=self.fee_schedule),
            catalog_path=self.catalog_root,
        )


@pytest.fixture
def always_long(monkeypatch: pytest.MonkeyPatch) -> None:
    """chromosome_to_dsl -> the always-long synthetic strategy (rising market, 1 trade)."""
    monkeypatch.setattr(
        "vibe_quant.discovery.genome.chromosome_to_dsl", lambda chrom: _dsl("4h")
    )


def test_screening_runner_reports_gross_and_inputs(catalog: Path) -> None:
    runner = NTScreeningRunner(_dsl("4h"), ["BTCUSDT"], _START, _END, catalog_path=str(catalog))
    m = runner({})
    assert m.total_trades == 1 and m.total_fees > 0.0
    assert runner.last_starting_balance == 1000.0
    assert runner.last_days == 4  # [2024-01-01, 2024-01-05)
    # Reported-cost convention: net + fees + (no separate slippage) + funding.
    assert m.gross_pnl == pytest.approx(
        m.total_return * 1000.0 + m.total_fees + m.total_funding, rel=1e-12
    )
    assert m.gross_pnl > m.total_return * 1000.0  # costs were added back


def test_run_single_carries_fee_inputs(catalog: Path, always_long: None) -> None:
    fn = _CatalogFn(["BTCUSDT"], "4h", _START, _END)
    fn.catalog_root = str(catalog)
    truth = NTScreeningRunner(_dsl("4h"), ["BTCUSDT"], _START, _END, catalog_path=str(catalog))({})
    out = fn._run_single(_chromosome(), _START, _END)
    assert out["total_fees"] == truth.total_fees > 0.0
    assert out["gross_pnl"] == truth.gross_pnl
    assert out["starting_balance"] == 1000.0
    assert out["days"] == 4


def test_discovery_fitness_from_real_screening_run_and_trade_gate(
    catalog: Path, always_long: None
) -> None:
    fn = _CatalogFn(["BTCUSDT"], "4h", _START, _END)
    fn.catalog_root = str(catalog)
    ok = _evaluate_single(_chromosome(), fn, min_trades=1, timeframe="4h")
    assert ok.adjusted_score > 0.0
    assert ok.fee_drag is not None and 0.0 < ok.fee_drag < 1.0
    assert ok.fees_pct_equity_yr is not None and ok.fees_pct_equity_yr > 0.0
    # Same run behind the hard trade gate: score 0, fee drag still reported.
    gated = _evaluate_single(_chromosome(), fn, min_trades=5, timeframe="4h")
    assert gated.adjusted_score == 0.0
    assert gated.fee_drag == ok.fee_drag
    assert gated.fees_pct_equity_yr == ok.fees_pct_equity_yr


# ---------------------------------------------------------------------------
# Persisted rows: legacy fields untouched, new keys always present (null)
# ---------------------------------------------------------------------------

_LEGACY_CHAMPION_KEYS = [
    "dsl", "chromosome", "score", "sharpe", "max_dd", "pf", "trades", "return_pct",
]


def test_champion_entry_legacy_fields_and_key_order_preserved() -> None:
    from vibe_quant.discovery.__main__ import _champion_entry
    from vibe_quant.discovery.genome import chromosome_to_dsl, chromosome_to_serializable

    chrom = _chromosome()
    fitness = _fit(fee_drag=0.2, fees_pct=0.05)
    fitness = replace(fitness, symbol_scores={"BTCUSDT": 0.5123456789})
    dsr: dict[str, object] = {"dsr": 0.97}
    bh: dict[str, object] = {"beat": True}
    entry = _champion_entry(chrom, fitness, timeframe="1h", dsr=dsr, buy_hold=bh)
    assert list(entry) == [
        *_LEGACY_CHAMPION_KEYS, "dsr", "buy_hold", "symbol_scores",
        "fee_drag", "fees_pct_equity_yr",
    ]
    expected_dsl = {**chromosome_to_dsl(chrom), "timeframe": "1h"}
    assert entry["dsl"] == expected_dsl
    assert entry["chromosome"] == chromosome_to_serializable(chrom)
    # "score" is the ADJUSTED score, not the raw one (raw 0.6 != adjusted 0.55).
    assert entry["score"] == fitness.adjusted_score == 0.55
    assert entry["score"] != fitness.raw_score
    assert entry["sharpe"] == 2.0 and entry["max_dd"] == 0.1
    assert entry["pf"] == 1.8 and entry["trades"] == 150 and entry["return_pct"] == 0.4
    assert entry["dsr"] is dsr and entry["buy_hold"] is bh
    assert entry["symbol_scores"] == {"BTCUSDT": 0.512346}
    # Without the optional blocks the legacy order is still dsl..return_pct, then drag.
    bare = _champion_entry(chrom, _fit(), timeframe="1h", dsr=None, buy_hold=None)
    assert list(bare) == [*_LEGACY_CHAMPION_KEYS, "fee_drag", "fees_pct_equity_yr"]


def test_champion_entry_persists_unavailable_drag_as_json_null() -> None:
    from vibe_quant.discovery.__main__ import _champion_entry

    entry = _champion_entry(_chromosome(), _fit(), timeframe="1h", dsr=None, buy_hold=None)
    loaded = json.loads(json.dumps(entry))
    assert "fee_drag" in loaded and loaded["fee_drag"] is None
    assert "fees_pct_equity_yr" in loaded and loaded["fees_pct_equity_yr"] is None


def test_rejection_row_persists_unavailable_drag_as_json_null() -> None:
    cfg = DiscoveryConfig(
        population_size=4, max_generations=2, elite_count=1, top_k=2,
        min_trades=1, symbols=["BTC/USDT"], timeframe="1h",
        start_date="2024-01-01", end_date="2024-02-01", max_workers=1,
    )  # fmt: skip
    pipe = DiscoveryPipeline(cfg, lambda chrom: {})
    pipe._reject(_chromosome(), _fit(), "guardrails", ["nope"])
    loaded = json.loads(json.dumps(pipe._guardrail_rejections[0]))
    assert "fee_drag" in loaded and loaded["fee_drag"] is None
    assert "fees_pct_equity_yr" in loaded and loaded["fees_pct_equity_yr"] is None
    assert loaded["reasons"] == ["nope"] and loaded["stage"] == "guardrails"
