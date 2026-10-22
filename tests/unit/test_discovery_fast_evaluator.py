"""Discovery ``--evaluator fast``: the fast tier scores the GA, NT confirms the top-K.

The fast tier must leave a discovery's outcome unchanged: the same genomes, the same
fitness and the same gate verdicts as an NT-evaluated run with the same seed, with
every gate reading NT numbers (vibe-quant-z8vqv.5).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult
from vibe_quant.fast_tier import FastTierUnsupported, fallbacks
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
    # the confirm keeps the per-bar indicator path: the one setting that differs (Q6)
    assert fast.pta_precompute is True and twin.pta_precompute is False
    assert {k: v for k, v in vars(fast).items() if k != "nt_fallback"} | {
        "pta_precompute": False
    } == vars(twin)


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
    twin_vars = vars(fast.backtest_fn.nt_twin())
    assert twin_vars["pta_precompute"] is False and vars(nt.backtest_fn)["pta_precompute"] is True
    assert twin_vars == {**vars(nt.backtest_fn), "pta_precompute": False}
    for fns in (fast, nt):
        assert type(fns.holdout_backtest_fn) is NTBacktestFn
        assert fns.backtest_fn_factory is not None
        assert type(fns.backtest_fn_factory(START, END)) is NTBacktestFn


def test_nt_twin_runs_live_indicator_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The confirm twin builds runners with pta_precompute False; the fast fn's own NT fallback ON."""
    from vibe_quant.screening import nt_runner

    seen: list[bool] = []
    original = nt_runner.NTScreeningRunner.__init__

    def spy(self: Any, *args: Any, **kwargs: Any) -> None:
        seen.append(kwargs["pta_precompute"])
        original(self, *args, **kwargs)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "__init__", spy)
    fast = FastBacktestFn(["BTCUSDT"], "1m", START, END, nt_fallback=True)
    fast._make_runner({"name": "x", "timeframe": "1m", "indicators": {}}, START, END)
    fast.nt_twin()._make_runner({"name": "x", "timeframe": "1m", "indicators": {}}, START, END)
    assert seen == [True, False]
    # a callable built without __init__ (older pickle) defaults to ON
    bare = NTBacktestFn.__new__(NTBacktestFn)
    bare.__dict__.update({k: v for k, v in vars(fast).items() if k != "pta_precompute"})
    bare._make_runner({"name": "x", "timeframe": "1m", "indicators": {}}, START, END)
    assert seen == [True, False, True]


# ---------------------------------------------------------------------------
# NT fallback accounting (vibe-quant-z8vqv.7)
# ---------------------------------------------------------------------------

FALLBACK_LOGGER = "vibe_quant.fast_tier.backtest_fn"
_METRICS: dict[str, float | int] = {
    "sharpe_ratio": 1.0,
    "max_drawdown": 0.1,
    "profit_factor": 1.5,
    "total_trades": 30,
    "total_return": 0.05,
}


@pytest.fixture(autouse=True)
def _clear_fallbacks() -> Any:
    fallbacks.clear()
    yield
    fallbacks.clear()


def _unsupported(reason: str) -> Any:
    def _run_fast(self: Any, *a: Any, **k: Any) -> Any:
        raise FastTierUnsupported(reason)

    return _run_fast


def _nt_metrics(self: Any, *a: Any, **k: Any) -> dict[str, float | int]:
    return dict(_METRICS)


def _fallback_fn(monkeypatch: pytest.MonkeyPatch, reason: str = "time filters") -> None:
    monkeypatch.setattr(FastBacktestFn, "_run_fast", _unsupported(reason))
    monkeypatch.setattr(NTBacktestFn, "_run_single", _nt_metrics)


def _lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == FALLBACK_LOGGER]


def _result(**over: Any) -> DiscoveryResult:
    base: dict[str, Any] = {
        "generations": [],
        "top_strategies": [],
        "total_candidates_evaluated": 10,
        "converged": True,
        "convergence_generation": 1,
    }
    base.update(over)
    return DiscoveryResult(**base)


def _notes(result: DiscoveryResult) -> dict[str, object]:
    from vibe_quant.discovery import record

    return record.provenance_notes(
        result,
        use_mock=True,
        eval_windows_count=1,
        eval_windows=None,
        symbol_agg="portfolio",
        split_ratio=0.0,
        direction=None,
        cross_window_months=[],
        cross_window_min_sharpe=0.5,
        wfa_oos_step_days=0,
        wfa_min_consistency=0.75,
        num_seeds=1,
        seed=42,
        bootstrap_min_sharpe=0.0,
        holdout_min_sharpe=0.0,
        symbols=["BTCUSDT"],
    )


def test_nt_fallback_logs_uid_and_reason_and_counts_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.unit.test_fast_tier import _genomes

    chromosome = _genomes()[0]
    chromosome.time_filters = {"block_hours": [0, 1]}
    _fallback_fn(monkeypatch)
    fn = FastBacktestFn(["BTCUSDT"], "1m", START, END, nt_fallback=True)

    with caplog.at_level(logging.INFO, logger=FALLBACK_LOGGER):
        assert fn._run_single(chromosome, START, END) == _METRICS

    lines = _lines(caplog)
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert chromosome.uid in lines[0].getMessage()
    assert "time filters" in lines[0].getMessage()
    assert lines[0].exc_info is None
    assert fallbacks.summary() == {"total": 1, "by_reason": {"time filters": 1}}


def test_fallback_reason_digits_are_normalised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.unit.test_fast_tier import _genomes

    raw = "open tick gaps through the take-profit at bar 1234"
    _fallback_fn(monkeypatch, raw)
    fn = FastBacktestFn(["BTCUSDT"], "1m", START, END, nt_fallback=True)

    with caplog.at_level(logging.INFO, logger=FALLBACK_LOGGER):
        fn._run_single(_genomes()[0], START, END)

    message = _lines(caplog)[0].getMessage()
    assert raw in message  # the log carries the raw reason (with the bar id)
    assert "BTCUSDT" in message
    assert f"{START}..{END}" in message
    assert fallbacks.summary() == {
        "total": 1,
        "by_reason": {"open tick gaps through the take-profit at bar N": 1},
    }


def test_notes_carry_nt_fallbacks_for_fast_only() -> None:
    from vibe_quant.discovery.pipeline import FastTierConfirm

    fast = _notes(_result(fast_tier_confirm=FastTierConfirm(3)))
    assert fast["evaluator"] == {
        "name": "fast",
        "nt_confirmed": 3,
        "nt_fallbacks": {"total": 0, "by_reason": {}},
    }
    fallbacks.record("time filters")
    counted = _notes(_result(fast_tier_confirm=FastTierConfirm(1)))
    assert counted["evaluator"]["nt_fallbacks"] == {  # type: ignore[index]
        "total": 1,
        "by_reason": {"time filters": 1},
    }
    assert "evaluator" not in _notes(_result())  # evaluator=nt keeps no key


def test_no_fallback_raises_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.unit.test_fast_tier import _genomes

    monkeypatch.setattr(FastBacktestFn, "_run_fast", _unsupported("time filters"))
    fn = FastBacktestFn(["BTCUSDT"], "1m", START, END)

    with (
        caplog.at_level(logging.INFO, logger=FALLBACK_LOGGER),
        pytest.raises(FastTierUnsupported, match="time filters"),
    ):
        fn._run_single(_genomes()[0], START, END)

    assert _lines(caplog) == []
    assert fallbacks.summary() == {"total": 0, "by_reason": {}}


class _StubNT(NTBacktestFn):
    """NT leg without an engine: picklable, so a spawned worker has it too."""

    def _run_single(self, *a: Any, **k: Any) -> dict[str, float | int]:
        return dict(_METRICS)


class _UnsupportedFast(FastBacktestFn, _StubNT):
    """The real ``FastBacktestFn._run_single`` fallback path with an unsupported replay."""

    def _run_fast(self, *a: Any, **k: Any) -> Any:
        raise FastTierUnsupported("time filters")


@pytest.mark.parametrize("start_method", ["spawn", "fork"])
@pytest.mark.parametrize(
    ("symbols", "agg"),
    [(["BTCUSDT"], "portfolio"), (["BTCUSDT", "ETHUSDT"], "worst")],
    ids=["generic", "split_by_symbol"],
)
def test_real_process_pool_counts_every_worker_fallback(
    monkeypatch: pytest.MonkeyPatch, start_method: str, symbols: list[str], agg: str
) -> None:
    """Real worker processes (not threads): spawn re-imports, fork inherits the parent's store."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from tests.unit.test_fast_tier import _genomes
    from vibe_quant.discovery import fitness
    from vibe_quant.discovery.pipeline import FastTierConfirm

    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} start method unavailable")
    monkeypatch.setattr(fitness, "_parallel_broken", False)
    chroms = _genomes()
    fn = _UnsupportedFast(symbols, "1m", START, END, symbol_agg=agg, nt_fallback=True)
    expected = len(chroms) * len(symbols)
    fallbacks.record("parent only")  # parent history a forked worker must not inherit or resend

    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context(start_method)
    ) as pool:
        results = fitness.evaluate_population(
            chroms, fn, None, max_workers=2, executor=pool, min_trades=10, timeframe="1m"
        )

    assert [r.error for r in results] == [None] * len(chroms)
    want = {"total": expected + 1, "by_reason": {"parent only": 1, "time filters": expected}}
    assert fallbacks.summary() == want
    evaluator = _notes(_result(fast_tier_confirm=FastTierConfirm(0)))["evaluator"]
    assert evaluator["nt_fallbacks"] == want  # type: ignore[index]


def test_pool_paths_record_worker_fallback_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tests.unit.test_fast_tier import _genomes
    from vibe_quant.discovery import fitness

    monkeypatch.setattr(fitness, "_parallel_broken", False)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(fallbacks, "record_worker", recorded.append)
    chroms = _genomes()

    def generic(chrom: Any) -> dict[str, float | int]:
        fallbacks.record("time filters")
        return dict(_METRICS)

    def fake_symbols(chrom: Any, symbols: list[str]) -> dict[str, float | int]:
        fallbacks.record("time filters")
        return dict(_METRICS)

    _, _, snapshot = fitness._evaluate_single_task(chroms[0], generic, None, 10, "1m")
    assert snapshot == fallbacks.snapshot()

    split = NTBacktestFn(["BTCUSDT", "ETHUSDT"], "1m", START, END, symbol_agg="worst")
    monkeypatch.setattr(split, "_eval_symbols", fake_symbols)
    out, _, snapshot = fitness._eval_symbol_task(split, chroms[0], "BTCUSDT")
    assert out == _METRICS
    assert snapshot == fallbacks.snapshot()

    # Generic pool path: one task per chromosome, snapshot shipped to the parent.
    fallbacks.clear()
    recorded.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fitness.evaluate_population(
            chroms, generic, None, max_workers=2, executor=pool, min_trades=10, timeframe="1m"
        )
    assert len(recorded) == len(chroms)
    assert all(snap["proc_id"] == fallbacks.snapshot()["proc_id"] for snap in recorded)
    assert fallbacks.summary() == {"total": len(chroms), "by_reason": {"time filters": len(chroms)}}

    # Sequential path: the same total without any pool transport.
    fallbacks.clear()
    recorded.clear()
    fitness.evaluate_population(chroms, generic, None, max_workers=None, min_trades=10, timeframe="1m")
    assert fallbacks.summary() == {"total": len(chroms), "by_reason": {"time filters": len(chroms)}}

    # Split-by-symbol path: one task per chromosome x symbol.
    fallbacks.clear()
    recorded.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fitness.evaluate_population(
            chroms, split, None, max_workers=2, executor=pool, min_trades=10, timeframe="1m"
        )
    assert len(recorded) == len(chroms) * len(split.symbols)
    assert fallbacks.summary() == {
        "total": len(chroms) * len(split.symbols),
        "by_reason": {"time filters": len(chroms) * len(split.symbols)},
    }
