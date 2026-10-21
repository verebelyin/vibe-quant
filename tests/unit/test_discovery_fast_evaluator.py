"""Discovery ``--evaluator fast``: the fast tier scores the GA, NT confirms the top-K.

The fast tier must leave a discovery's outcome unchanged: the same genomes, the same
fitness and the same gate verdicts as an NT-evaluated run with the same seed, with
every gate reading NT numbers (vibe-quant-z8vqv.5).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult
from vibe_quant.fast_tier import FastTierUnsupported
from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

REPO = Path(__file__).resolve().parents[2]
CATALOG_BARS = REPO / "data" / "catalog" / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
needs_catalog = pytest.mark.skipif(not CATALOG_BARS.is_dir(), reason="needs the BTCUSDT 1m catalog")
START, END = "2026-03-02", "2026-03-04"


@pytest.fixture(autouse=True)
def _scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from vibe_quant.discovery.genome import MA_POOL
    from vibe_quant.discovery.operators import _INDICATOR_NAMES, _MA_NAMES, INDICATOR_POOL

    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(REPO)
    saved = dict(INDICATOR_POOL), list(_INDICATOR_NAMES), dict(MA_POOL), list(_MA_NAMES)
    yield
    INDICATOR_POOL.clear()
    INDICATOR_POOL.update(saved[0])
    _INDICATOR_NAMES[:] = saved[1]
    MA_POOL.clear()
    MA_POOL.update(saved[2])
    _MA_NAMES[:] = saved[3]


def _config(max_workers: int | None = None) -> DiscoveryConfig:
    # top_k 1 < genomes keeping an MTM series: the confirm must cover more than the top-K
    return DiscoveryConfig(
        population_size=4, max_generations=2, convergence_generations=2, top_k=1,
        min_trades=5, max_workers=max_workers, symbols=["BTCUSDT"], timeframe="1m",
        start_date=START, end_date=END, indicator_pool=["CCI", "WILLR", "RSI"],
        direction="long", eval_windows=1,
    )


def _run(fn: NTBacktestFn, max_workers: int | None = None) -> tuple[DiscoveryResult, DiscoveryPipeline]:
    random.seed(5)
    pipe = DiscoveryPipeline(_config(max_workers), fn)
    return pipe.run(), pipe


def _fit(fr: Any) -> tuple[object, ...]:
    from vibe_quant.discovery.pipeline import _confirm_key

    return _confirm_key(fr)


def _outcome(result: DiscoveryResult) -> dict[str, Any]:
    return {
        "evaluated": result.total_candidates_evaluated,
        "generations": [(g.best_fitness, g.mean_fitness) for g in result.generations],
        "champions": [(repr(c), _fit(fr)) for c, fr in result.top_strategies],
        "rejections": [(r.get("uid"), r.get("reasons")) for r in result.guardrail_rejections],
        "dsr_trials": result.dsr_trials,
        "dsr_records": result.dsr_records,
    }


@needs_catalog
def test_fast_evaluator_ends_with_the_production_nt_outcome() -> None:
    # Production NT runs its backtests in worker processes, so they never touch the
    # GA's global random (in-process they do: vibe-quant-0bwe2); the fast tier, sequential
    # here, draws nothing from it either.
    nt, nt_pipe = _run(NTBacktestFn(["BTCUSDT"], "1m", START, END), max_workers=2)
    fast, fast_pipe = _run(FastBacktestFn(["BTCUSDT"], "1m", START, END, nt_fallback=True))

    assert _outcome(fast) == _outcome(nt)
    assert {k: _fit(v) for k, v in fast_pipe._fitness_cache.items()} == {
        k: _fit(v) for k, v in nt_pipe._fitness_cache.items()
    }
    assert nt.fast_tier_confirm is None
    series_keys = {k for k, fr in fast_pipe._fitness_cache.items() if fr.mtm_daily_returns is not None}
    top_keys = {fast_pipe._dsl_key(c) for c, _ in fast.top_strategies}
    assert len(series_keys) > _config().top_k  # DSR effective-N rows beyond the top-K...
    assert fast.fast_tier_confirm is not None
    assert fast.fast_tier_confirm.confirmed == len(series_keys | top_keys)  # ...and were confirmed
    assert _outcome(nt)["rejections"] or _outcome(nt)["champions"]  # gates actually ran


class _WrongFast(FastBacktestFn):
    """A broken fast tier: one result field is off by a little."""

    field = "sharpe_ratio"

    def _run_fast(self, *a: Any, **k: Any) -> dict[str, float | int]:
        out = super()._run_fast(*a, **k)
        if self.field == "sharpe_ratio":
            out["sharpe_ratio"] = float(out["sharpe_ratio"]) + 1e-9
        else:  # a bootstrap-only input: one trade return nudged
            rets = list(out["trade_returns"])  # type: ignore[call-overload]
            if rets:
                rets[0] += 1e-12
            out["trade_returns"] = tuple(rets)  # type: ignore[assignment]
        return out


@needs_catalog
@pytest.mark.parametrize("field", ["sharpe_ratio", "trade_returns"])
def test_a_wrong_fast_tier_fails_the_run_before_any_gate(field: str) -> None:
    from vibe_quant.discovery.pipeline import FastTierParityError

    fn = _WrongFast(["BTCUSDT"], "1m", START, END, nt_fallback=True)
    fn.field = field
    with pytest.raises(FastTierParityError, match="fast tier != NT"):
        _run(fn)


@needs_catalog
def test_unsupported_genomes_fall_back_to_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_fast_tier import _genomes

    def unsupported(self: Any, *a: Any, **k: Any) -> Any:
        raise FastTierUnsupported("forced")

    monkeypatch.setattr(FastBacktestFn, "_run_fast", unsupported)
    chromosome = _genomes()[0]
    nt = NTBacktestFn(["BTCUSDT"], "1m", START, END)._run_single(chromosome, START, END)
    fallback = FastBacktestFn(["BTCUSDT"], "1m", START, END, nt_fallback=True)
    random.seed(9)
    state = random.getstate()
    assert fallback._run_single(chromosome, START, END) == nt
    assert random.getstate() == state  # NT's fill model reseeded and drew; restored
    with pytest.raises(FastTierUnsupported, match="forced"):
        FastBacktestFn(["BTCUSDT"], "1m", START, END)._run_single(chromosome, START, END)


def test_nt_twin_copies_every_setting_and_is_a_plain_nt_fn() -> None:
    from vibe_quant.liquidation_settings import LiquidationSettings

    fast = FastBacktestFn(
        ["BTCUSDT"], "1m", START, END, windows=[(START, "2026-03-03"), ("2026-03-03", END)],
        min_trades=7, liquidation=LiquidationSettings(mode="isolated", leverage=10), nt_fallback=True,
    )
    fast.collect_daily_returns = True  # what the pipeline sets after construction
    twin = fast.nt_twin()
    assert type(twin) is NTBacktestFn
    assert {k: v for k, v in vars(fast).items() if k != "nt_fallback"} == vars(twin)


def test_cell_fns_put_the_fast_tier_on_the_train_fn_only() -> None:
    from vibe_quant.discovery.__main__ import _make_cell_fns
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    kw: dict[str, Any] = {
        "use_mock": False, "symbols": ["BTCUSDT"], "timeframe": "1m", "symbol_agg": "worst",
        "fee_schedule": FeeSchedule(), "entry_order": "market",
        "liquidation": LiquidationSettings(mode="isolated", leverage=10), "sizing": SizingSettings(),
        "train_start": START, "train_end": "2026-03-03", "eval_windows": None, "min_trades": 5,
        "holdout": ("2026-03-03", END), "needs_factory": True,
    }
    fast = _make_cell_fns(**kw, evaluator="fast")
    nt = _make_cell_fns(**kw)
    assert isinstance(fast.backtest_fn, FastBacktestFn) and fast.backtest_fn.nt_fallback
    assert type(nt.backtest_fn) is NTBacktestFn
    assert vars(fast.backtest_fn.nt_twin()) == vars(nt.backtest_fn)
    for fns in (fast, nt):
        assert type(fns.holdout_backtest_fn) is NTBacktestFn
        assert fns.backtest_fn_factory is not None
        assert type(fns.backtest_fn_factory(START, END)) is NTBacktestFn
