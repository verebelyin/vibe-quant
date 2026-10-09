"""Bar-coverage preflight: missing catalog bars must fail the run BEFORE evaluation.

vibe-quant-11zoq: ``DiscoveryPipeline._preflight_bars`` checks every window a
backtest worker will actually run -- the train fn's eval sub-windows, the
holdout fn's window, and the cross-window / WFA gate windows -- against the
default GA catalog, letting ``MissingBarDataError`` propagate out of ``run()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import _random_chromosome
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.dsl.identity import dsl_body_key
from vibe_quant.screening import nt_runner
from vibe_quant.screening.nt_runner import MissingBarDataError
from vibe_quant.utils import split_into_windows

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome

_TRAIN = ("2024-01-01", "2024-03-16")
_HOLDOUT = ("2024-03-16", "2024-06-01")
_BAR_TYPE = "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL"


def _metrics() -> dict[str, float | int]:
    """Fixed passing metrics (mirrors tests/unit/test_discovery_pipeline.py)."""
    import random as _rng

    r = _rng.Random(7)
    trade_returns = tuple(r.gauss(0.0033, 0.001) for _ in range(150))
    return {
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.08,
        "profit_factor": 2.0,
        "total_trades": 150,
        "total_return": 0.5,
        "trade_returns": trade_returns,  # type: ignore[dict-item]
    }


class _RecordingBacktestFn(NTBacktestFn):
    """NTBacktestFn-shaped injected fn: counts calls, never runs NT."""

    def __init__(
        self,
        start: str = _TRAIN[0],
        end: str = _TRAIN[1],
        *,
        windows: list[tuple[str, str]] | None = None,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
    ) -> None:
        super().__init__(symbols or ["BTCUSDT"], timeframe, start, end, windows=windows)
        self.calls = 0

    def __call__(self, chromosome: StrategyChromosome) -> dict[str, float | int]:
        self.calls += 1
        return _metrics()


def _holdout_fn(**kwargs: Any) -> _RecordingBacktestFn:
    return _RecordingBacktestFn(_HOLDOUT[0], _HOLDOUT[1], **kwargs)


def _cfg(**overrides: Any) -> DiscoveryConfig:
    d: dict[str, Any] = {
        "population_size": 4,
        "max_generations": 2,
        "elite_count": 1,
        "tournament_size": 2,
        "convergence_generations": 1,
        "top_k": 1,
        "min_trades": 1,
        "max_workers": 1,  # sequential so the call counter stays in-process
        "symbols": ["BTCUSDT"],
        "timeframe": "1h",
        "start_date": _TRAIN[0],
        "end_date": _TRAIN[1],
        "train_test_split": 0.5,
        "holdout_start_date": _HOLDOUT[0],
        "holdout_end_date": _HOLDOUT[1],
        "require_dsr": False,
    }
    d.update(overrides)
    return DiscoveryConfig(**d)


@pytest.fixture(autouse=True)
def _no_aux_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aux coverage is tested elsewhere; keep these tests to bar coverage."""
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_aux_data", lambda self: None)


def test_holdout_window_missing_bars_fails_before_any_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()

    def _raise_for_holdout(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        if (start, end) == _HOLDOUT:
            raise MissingBarDataError(f"No catalog bars for {bar_type} in window {start}..{end}")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _raise_for_holdout)
    pipe = DiscoveryPipeline(_cfg(), fn, holdout_backtest_fn=holdout_fn)
    with pytest.raises(MissingBarDataError):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_empty_catalog_fails_before_any_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fn = _RecordingBacktestFn()
    monkeypatch.setattr("vibe_quant.data.catalog.DEFAULT_CATALOG_PATH", tmp_path)
    pipe = DiscoveryPipeline(_cfg(), fn, holdout_backtest_fn=_holdout_fn())
    with pytest.raises(MissingBarDataError, match="1-HOUR"):
        pipe.run()
    assert fn.calls == 0


def test_train_and_holdout_covered_run_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    checked: list[tuple[str, str, str]] = []

    def _ok(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        checked.append((bar_type, start, end))

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _ok)
    pipe = DiscoveryPipeline(_cfg(), fn, holdout_backtest_fn=holdout_fn)
    result = pipe.run()
    assert result.top_strategies is not None
    assert fn.calls > 0
    assert checked == [
        (_BAR_TYPE, _TRAIN[0], _TRAIN[1]),
        (_BAR_TYPE, _HOLDOUT[0], _HOLDOUT[1]),
    ]


def test_bar_preflight_uses_resolved_default_catalog_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers read the default catalog; preflight must check that exact path."""
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    catalog_paths: list[str] = []

    def _record(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        catalog_paths.append(catalog_path)

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _record)
    pipe = DiscoveryPipeline(_cfg(), fn, holdout_backtest_fn=holdout_fn)
    pipe.run()
    assert catalog_paths
    assert set(catalog_paths) == {str(Path(DEFAULT_CATALOG_PATH).resolve())}


def test_second_symbol_missing_bars_fails_before_any_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = _RecordingBacktestFn(symbols=["BTCUSDT", "ETHUSDT"])
    holdout_fn = _holdout_fn(symbols=["BTCUSDT", "ETHUSDT"])

    def _second_symbol_missing(
        catalog_path: str, bar_type: str, start: str, end: str
    ) -> None:
        if bar_type.startswith("ETHUSDT"):
            raise MissingBarDataError(f"No catalog bars for {bar_type} in window {start}..{end}")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _second_symbol_missing)
    pipe = DiscoveryPipeline(
        _cfg(symbols=["BTCUSDT", "ETHUSDT"]), fn, holdout_backtest_fn=holdout_fn
    )
    with pytest.raises(MissingBarDataError, match="ETHUSDT"):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_eval_subwindow_missing_bars_fails_before_any_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bars covering only the tail of train: overlap OK on the full span, but
    eval sub-window 1 has no bars and must abort BEFORE any evaluation."""
    windows = split_into_windows(_TRAIN[0], _TRAIN[1], 3)
    fn = _RecordingBacktestFn(windows=windows)
    holdout_fn = _holdout_fn()
    bars_lo, bars_hi = "2024-02-20", _TRAIN[1]  # last sub-window only

    def _bars_only_last_part(
        catalog_path: str, bar_type: str, start: str, end: str
    ) -> None:
        if end < bars_lo or start > bars_hi:
            raise MissingBarDataError(f"No catalog bars for {bar_type} in window {start}..{end}")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _bars_only_last_part)
    pipe = DiscoveryPipeline(
        _cfg(eval_windows=3), fn, holdout_backtest_fn=holdout_fn
    )
    with pytest.raises(MissingBarDataError, match=windows[0][0]):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_single_fn_window_preflights_the_full_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """backtest_fn runs multi-window only with 2+ entries: with a single
    window the worker runs the full span, so preflight checks the full span."""
    fn = _RecordingBacktestFn(windows=[("2024-02-20", _TRAIN[1])])
    holdout_fn = _holdout_fn()
    checked: list[tuple[str, str]] = []

    def _ok(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        checked.append((start, end))

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _ok)
    pipe = DiscoveryPipeline(_cfg(), fn, holdout_backtest_fn=holdout_fn)
    pipe.run()
    assert checked == [_TRAIN, _HOLDOUT]


def test_cross_window_missing_bars_fails_before_any_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate windows count too: a missing cross window aborts before the GA."""
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[1]),
        fn,
        holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _RecordingBacktestFn(s, e),
    )
    cross = [(ws, we) for _, ws, we in pipe.cross_window_ranges()]
    assert cross

    def _raise_for_cross(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        if (start, end) in cross:
            raise MissingBarDataError(f"No catalog bars for {bar_type} in window {start}..{end}")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _raise_for_cross)
    with pytest.raises(MissingBarDataError, match=cross[0][0]):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_cross_window_range_error_does_not_abort_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config error in a gate-window helper is NOT missing data coverage.

    The bar preflight skips the un-derivable windows and the gate fails closed
    on the same error after the GA — preflight must not turn config problems
    into early aborts (chief, 2026-10-09).
    """
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    checked: list[tuple[str, str]] = []

    def _ok(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        checked.append((start, end))

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _ok)
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[3]),  # +3mo window falls outside the train range
        fn,
        holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _RecordingBacktestFn(s, e),
    )
    result = pipe.run()  # must NOT raise
    assert fn.calls > 0  # the GA ran — no early abort
    assert checked == [(_TRAIN[0], _TRAIN[1]), (_HOLDOUT[0], _HOLDOUT[1])]
    # ...and the cross-window gate still fails closed after the GA.
    assert result.top_strategies == []
    assert any(r["stage"] == "cross_window" for r in result.guardrail_rejections)


def test_wfa_window_missing_bars_fails_before_any_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate windows count too: a missing WFA window aborts before the GA."""
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    pipe = DiscoveryPipeline(
        _cfg(wfa_oos_step_days=30),
        fn,
        holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _RecordingBacktestFn(s, e),
    )
    wfa = pipe.wfa_window_ranges(_TRAIN[0], _TRAIN[1])
    assert wfa

    def _raise_for_wfa(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        if (start, end) in wfa:
            raise MissingBarDataError(f"No catalog bars for {bar_type} in window {start}..{end}")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _raise_for_wfa)
    with pytest.raises(MissingBarDataError, match=wfa[0][1]):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_wfa_range_error_does_not_abort_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config ValueError from the WFA range helper is NOT missing data.

    Same policy as the cross-window helper above: preflight skips the
    un-derivable windows and the gate fails closed after the GA.
    """
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    checked: list[tuple[str, str]] = []
    real_wfa = DiscoveryPipeline.wfa_window_ranges
    calls = 0

    def _raise_once(
        self: DiscoveryPipeline, range_start: str, range_end: str
    ) -> list[tuple[str, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:  # preflight only; the gate re-derives windows after the GA
            raise ValueError("WFA: config problem")
        return real_wfa(self, range_start, range_end)

    monkeypatch.setattr(DiscoveryPipeline, "wfa_window_ranges", _raise_once)

    def _ok(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        checked.append((start, end))

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _ok)
    pipe = DiscoveryPipeline(
        _cfg(wfa_oos_step_days=30),
        fn,
        holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _RecordingBacktestFn(s, e),
    )
    pipe.run()  # must NOT raise
    assert fn.calls > 0  # the GA ran — no early abort
    assert checked == [_TRAIN, _HOLDOUT]


def test_wfa_range_data_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """DataUnavailableError from a range helper is data, not config: it must
    abort even though DataUnavailableError subclasses ValueError."""
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    real_wfa = DiscoveryPipeline.wfa_window_ranges
    calls = 0

    def _raise_data_error_once(
        self: DiscoveryPipeline, range_start: str, range_end: str
    ) -> list[tuple[str, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MissingBarDataError("WFA ranges hit missing bars")
        return real_wfa(self, range_start, range_end)

    monkeypatch.setattr(DiscoveryPipeline, "wfa_window_ranges", _raise_data_error_once)

    def _boom(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        raise AssertionError("must fail while deriving gate windows")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _boom)
    pipe = DiscoveryPipeline(
        _cfg(wfa_oos_step_days=30),
        fn,
        holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _RecordingBacktestFn(s, e),
    )
    with pytest.raises(MissingBarDataError, match="missing bars"):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_cross_range_data_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same as the WFA case for the cross-window helper."""
    fn = _RecordingBacktestFn()
    holdout_fn = _holdout_fn()
    real_cross = DiscoveryPipeline.cross_window_ranges
    calls = 0

    def _raise_data_error_once(self: DiscoveryPipeline) -> list[tuple[int, str, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MissingBarDataError("cross ranges hit missing bars")
        return real_cross(self)

    monkeypatch.setattr(DiscoveryPipeline, "cross_window_ranges", _raise_data_error_once)

    def _boom(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        raise AssertionError("must fail while deriving gate windows")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _boom)
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[1]),
        fn,
        holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _RecordingBacktestFn(s, e),
    )
    with pytest.raises(MissingBarDataError, match="missing bars"):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_unknown_timeframe_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        raise AssertionError("must fail before any bar check")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _boom)
    pipe = DiscoveryPipeline(_cfg(), _RecordingBacktestFn(timeframe="7h"))
    with pytest.raises(ValueError, match="unknown timeframe"):
        pipe.run()


def test_dsl_key_is_dsl_body_key_of_compiled_dsl() -> None:
    chrom = _random_chromosome()
    assert DiscoveryPipeline._dsl_key(chrom) == dsl_body_key(chromosome_to_dsl(chrom))


def test_fake_backtest_fn_skips_bar_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipelines built with injected fake fns never touch the default catalog."""

    def _boom(catalog_path: str, bar_type: str, start: str, end: str) -> None:
        raise AssertionError("bar preflight must not run for injected fake backtest fns")

    monkeypatch.setattr(nt_runner, "require_bars_in_window", _boom)
    pipe = DiscoveryPipeline(
        _cfg(), lambda chrom: _metrics(), holdout_backtest_fn=lambda chrom: _metrics()
    )
    pipe.run()
