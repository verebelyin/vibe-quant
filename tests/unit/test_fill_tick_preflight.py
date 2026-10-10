"""Parent-side fill-tick resolve (vibe-quant-yul7u.12).

Parents (discovery preflight, screening sweep, overfitting runners, auto_screen)
resolve the tick set ONCE per (symbols, tick timeframe) over the union of the
worker windows and hand it to ``NTScreeningRunner(fill_ticks=...)``; workers
never build ticks. Missing data aborts before the GA / sweep.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml

from vibe_quant.data.fill_ticks import FillTickSet
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening import nt_runner
from vibe_quant.screening.nt_runner import MissingBarDataError
from vibe_quant.validation.extraction import date_to_ns

_TRAIN = ("2024-01-01", "2024-03-16")
_HOLDOUT = ("2024-03-16", "2024-06-01")
_TEMPLATES = Path(__file__).resolve().parents[2] / "vibe_quant" / "strategies" / "templates"

# A stub set's coverage window: the F3 window check rejects a set whose
# [start_ns, end_ns] does not cover the run window (vibe-quant-yul7u.10).
_SET_START_NS = date_to_ns(_TRAIN[0])
_SET_END_NS = date_to_ns(_HOLDOUT[1])


def _set(tf: str, symbols: list[str]) -> FillTickSet:
    return FillTickSet(
        timeframe=tf,
        paths={s: f"/ticks/{tf}/{s}" for s in symbols},
        missing_boundaries={s: 0 for s in symbols},
        start_ns=_SET_START_NS,
        end_ns=_SET_END_NS,
    )


class _Resolver:
    """Stand-in for ``resolve_fill_ticks``: records calls, returns a set."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[tuple[list[str], str, str, str]] = []
        self.exc = exc

    def __call__(
        self, symbols: Any, timeframe: str, start: Any, end: Any, catalog_path: Any
    ) -> FillTickSet:
        self.calls.append((list(symbols), timeframe, str(start), str(end)))
        if self.exc is not None:
            raise self.exc
        return _set(timeframe, list(symbols))


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch) -> _Resolver:
    r = _Resolver()
    monkeypatch.setattr("vibe_quant.data.fill_ticks.resolve_fill_ticks", r)
    return r


def _metrics() -> dict[str, float | int]:
    import random

    rng = random.Random(7)
    return {
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.08,
        "profit_factor": 2.0,
        "total_trades": 150,
        "total_return": 0.5,
        "trade_returns": tuple(rng.gauss(0.0033, 0.001) for _ in range(150)),  # type: ignore[dict-item]
    }


class _Fn(NTBacktestFn):
    def __init__(self, start: str, end: str, **kw: Any) -> None:
        super().__init__(kw.pop("symbols", ["BTCUSDT"]), kw.pop("timeframe", "4h"), start, end, **kw)
        self.calls = 0

    def __call__(self, chromosome: Any) -> dict[str, float | int]:
        self.calls += 1
        return _metrics()


def _cfg(**over: Any) -> DiscoveryConfig:
    d: dict[str, Any] = {
        "population_size": 4, "max_generations": 2, "elite_count": 1, "tournament_size": 2,
        "convergence_generations": 1, "top_k": 1, "min_trades": 1, "max_workers": 1,
        "symbols": ["BTCUSDT"], "timeframe": "4h",
        "start_date": _TRAIN[0], "end_date": _TRAIN[1], "train_test_split": 0.5,
        "holdout_start_date": _HOLDOUT[0], "holdout_end_date": _HOLDOUT[1],
        "require_dsr": False,
    }
    d.update(over)
    return DiscoveryConfig(**d)


@pytest.fixture(autouse=True)
def _no_aux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_aux_data", lambda self: None)


@pytest.fixture
def bars_ok(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Bar checks pass; returns the windows they were asked about."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        nt_runner, "require_bars_in_window", lambda c, b, s, e: seen.append((s, e))
    )
    return seen


def test_discovery_preflight_resolves_once_over_union_of_all_windows(
    resolver: _Resolver, bars_ok: list[tuple[str, str]]
) -> None:
    from vibe_quant.utils import split_into_windows

    windows = split_into_windows(_TRAIN[0], _TRAIN[1], 3)
    fn = _Fn(*_TRAIN, windows=windows, symbols=["BTCUSDT", "ETHUSDT"])
    holdout_fn = _Fn(*_HOLDOUT, symbols=["BTCUSDT", "ETHUSDT"])
    pipe = DiscoveryPipeline(
        _cfg(symbols=["BTCUSDT", "ETHUSDT"], cross_window_months=[1], wfa_oos_step_days=30,
             eval_windows=3),
        fn,
        holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _Fn(s, e, symbols=["BTCUSDT", "ETHUSDT"]),
    )
    pipe.run()
    assert len(set(bars_ok)) > 4  # train subs + holdout + cross + WFA all counted
    assert len(resolver.calls) == 1
    symbols, tf, start, end = resolver.calls[0]
    assert (symbols, tf) == (["BTCUSDT", "ETHUSDT"], "4h")
    assert start == min(s for s, _ in bars_ok)
    assert end == max(e for _, e in bars_ok)
    assert fn.calls > 0


def test_discovery_preflight_union_spans_min_start_max_end(
    resolver: _Resolver, bars_ok: list[tuple[str, str]]
) -> None:
    """The resolve spans min(start)..max(end), not the FIRST window (r1).

    A holdout window that starts BEFORE the train window makes ``windows[0]``
    the wrong answer: a ``windows[0][0]`` mutant resolves ticks for a window
    that does not cover the earliest worker, and the F3 coverage check would
    then raise MissingBarDataError mid-GA.
    """
    fn = _Fn("2024-02-01", "2024-03-16")  # train window starts later
    holdout_fn = _Fn("2024-01-01", "2024-06-01")  # holdout starts earliest
    pipe = DiscoveryPipeline(_cfg(), fn, holdout_backtest_fn=holdout_fn)
    pipe.run()
    assert min(s for s, _ in bars_ok) == "2024-01-01"
    assert max(e for _, e in bars_ok) == "2024-06-01"
    assert len(resolver.calls) == 1
    symbols, tf, start, end = resolver.calls[0]
    assert (symbols, tf) == (["BTCUSDT"], "4h")
    assert (start, end) == ("2024-01-01", "2024-06-01")


def test_set_reaches_every_fn_and_factory_fn(
    resolver: _Resolver, bars_ok: list[tuple[str, str]]
) -> None:
    fn, holdout_fn = _Fn(*_TRAIN), _Fn(*_HOLDOUT)
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[1]), fn, holdout_backtest_fn=holdout_fn,
        backtest_fn_factory=lambda s, e: _Fn(s, e),
    )
    pipe.run()
    expect = _set("4h", ["BTCUSDT"])
    assert fn.fill_ticks == holdout_fn.fill_ticks == expect
    assert pipe._backtest_fn_factory is not None
    assert pipe._backtest_fn_factory("2024-01-01", "2024-02-01").fill_ticks == expect  # type: ignore[attr-defined]


def test_fn_pickles_with_set_and_worker_never_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker path: the pickled fn hands its set to the runner; no resolve."""
    from vibe_quant.discovery.genome import chromosome_to_dsl
    from vibe_quant.discovery.operators import _random_chromosome

    ticks = _set("4h", ["BTCUSDT"])
    fn = pickle.loads(pickle.dumps(NTBacktestFn(["BTCUSDT"], "4h", *_TRAIN, fill_ticks=ticks)))
    assert fn.fill_ticks == ticks

    def _boom(*a: Any, **k: Any) -> None:
        raise AssertionError("worker resolved fill ticks")

    monkeypatch.setattr("vibe_quant.data.fill_ticks.resolve_fill_ticks", _boom)
    seen: dict[str, Any] = {}

    class _Runner:
        def __init__(self, **kw: Any) -> None:
            seen.update(kw)

        def __call__(self, params: dict[str, Any]) -> Any:
            raise RuntimeError("stop after construction")

    monkeypatch.setattr(nt_runner, "NTScreeningRunner", _Runner)
    chrom = _random_chromosome()
    del chromosome_to_dsl
    with pytest.raises(RuntimeError, match="stop after"):
        fn._run_single(chrom, *_TRAIN)
    assert seen["fill_ticks"] == ticks


def test_runner_with_supplied_set_does_not_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real runner: a supplied matching set short-circuits the in-process resolve."""
    def _boom(*a: Any, **k: Any) -> None:
        raise AssertionError("resolved in worker")

    monkeypatch.setattr("vibe_quant.data.fill_ticks.resolve_fill_ticks", _boom)
    runner = nt_runner.NTScreeningRunner(
        {}, ["BTCUSDT"], *_TRAIN, fill_ticks=_set("4h", ["BTCUSDT"])
    )
    got = runner._resolve_fill_ticks("4h", Path("/nonexistent"))
    assert got.paths == {"BTCUSDT": "/ticks/4h/BTCUSDT"}


def test_1m_strategy_resolves_nothing(
    resolver: _Resolver, bars_ok: list[tuple[str, str]]
) -> None:
    fn = _Fn(*_TRAIN, timeframe="1m")
    pipe = DiscoveryPipeline(_cfg(timeframe="1m"), fn, holdout_backtest_fn=_Fn(*_HOLDOUT, timeframe="1m"))
    pipe.run()
    assert resolver.calls == []
    assert fn.fill_ticks is None


def test_loaded_1h_indicator_tf_picks_1h_ticks(resolver: _Resolver) -> None:
    from vibe_quant.screening.pipeline import resolve_parent_fill_ticks

    raw = yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text())
    raw["additional_timeframes"] = ["1h"]
    raw["indicators"] = {**raw["indicators"], "ema_1h": {"type": "EMA", "period": 20, "timeframe": "1h"}}
    dsl_dict = validate_strategy_dict(raw).model_dump()
    assert dsl_dict["timeframe"] == "4h"
    ticks = resolve_parent_fill_ticks(dsl_dict, ["BTCUSDT"], *_TRAIN)
    assert ticks is not None and ticks.timeframe == "1h"
    assert resolver.calls == [(["BTCUSDT"], "1h", _TRAIN[0], _TRAIN[1])]


def test_loaded_1h_additional_timeframe_picks_1h_ticks(resolver: _Resolver) -> None:
    """An additional_timeframe (no 1h indicator) still loads 1h -> 1h ticks."""
    from vibe_quant.screening.pipeline import resolve_parent_fill_ticks

    raw = yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text())
    raw["additional_timeframes"] = ["1h"]
    dsl_dict = validate_strategy_dict(raw).model_dump()
    ticks = resolve_parent_fill_ticks(dsl_dict, ["BTCUSDT"], *_TRAIN)
    assert ticks is not None and ticks.timeframe == "1h"
    assert resolver.calls == [(["BTCUSDT"], "1h", _TRAIN[0], _TRAIN[1])]


def test_loaded_indicator_tf_alone_picks_1h_ticks(resolver: _Resolver) -> None:
    """Indicator timeframes are read directly, not only via additional_timeframes.

    A validated DSL always declares an indicator's tf in additional_timeframes
    too, so this reads the raw dict the parent is handed (it does not validate):
    an indicator-only 1h override must still resolve 1h ticks, which is what
    kills a "drop indicator timeframes" helper mutant (r1).
    """
    from vibe_quant.screening.pipeline import resolve_parent_fill_ticks

    dsl_dict = {
        "timeframe": "4h",
        "indicators": {
            "ema": {"type": "EMA", "timeframe": "4h"},
            "rsi_1h": {"type": "RSI", "timeframe": "1h"},
        },
    }
    ticks = resolve_parent_fill_ticks(dsl_dict, ["BTCUSDT"], *_TRAIN)
    assert ticks is not None and ticks.timeframe == "1h"
    assert resolver.calls == [(["BTCUSDT"], "1h", _TRAIN[0], _TRAIN[1])]


def test_runner_and_parent_agree_on_loaded_timeframes(resolver: _Resolver) -> None:
    """The runner and the parent derive the loaded set from ONE helper (r1).

    A divergence makes the parent resolve ticks for the wrong tick_timeframe
    (the finest LOADED tf), and the runner then rejects the set with
    MissingBarDataError at run time.
    """
    from vibe_quant.screening.nt_runner import NTScreeningRunner, loaded_timeframes
    from vibe_quant.screening.pipeline import resolve_parent_fill_ticks

    raw = yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text())
    raw["additional_timeframes"] = ["1h"]
    raw["indicators"] = {
        **raw["indicators"],
        "ema_1h": {"type": "EMA", "period": 20, "timeframe": "1h"},
    }
    dsl_dict = validate_strategy_dict(raw).model_dump()

    runner = NTScreeningRunner(dsl_dict, ["BTCUSDT"], *_TRAIN)
    runner._ensure_compiled()
    resolve_parent_fill_ticks(dsl_dict, ["BTCUSDT"], *_TRAIN)

    assert runner._all_timeframes == {"4h", "1h"}
    assert loaded_timeframes(dsl_dict) == runner._all_timeframes
    # The parent resolved for the finest loaded tf the runner uses.
    assert resolver.calls == [(["BTCUSDT"], "1h", _TRAIN[0], _TRAIN[1])]


def test_loaded_1m_indicator_uses_bar_mode_no_ticks(resolver: _Resolver) -> None:
    from vibe_quant.screening.pipeline import resolve_fill_ticks_for

    assert resolve_fill_ticks_for("4h", {"4h", "1m"}, ["BTCUSDT"], *_TRAIN) is None
    assert resolver.calls == []


def test_missing_1m_data_aborts_before_the_ga(
    monkeypatch: pytest.MonkeyPatch, bars_ok: list[tuple[str, str]]
) -> None:
    r = _Resolver(MissingBarDataError("No 1m bars for BTCUSDT"))
    monkeypatch.setattr("vibe_quant.data.fill_ticks.resolve_fill_ticks", r)
    fn, holdout_fn = _Fn(*_TRAIN), _Fn(*_HOLDOUT)
    pipe = DiscoveryPipeline(_cfg(), fn, holdout_backtest_fn=holdout_fn)
    with pytest.raises(MissingBarDataError, match="1m bars"):
        pipe.run()
    assert fn.calls == 0
    assert holdout_fn.calls == 0


def test_screening_sweep_parent_passes_the_set(resolver: _Resolver) -> None:
    from vibe_quant.screening.pipeline import create_screening_pipeline

    raw = yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text())
    dsl = validate_strategy_dict(raw)
    pipe = create_screening_pipeline(
        dsl, symbols=["BTCUSDT", "ETHUSDT"], start_date=_TRAIN[0], end_date=_TRAIN[1]
    )
    runner = pipe._runner  # type: ignore[attr-defined]
    assert resolver.calls == [(["BTCUSDT", "ETHUSDT"], "4h", _TRAIN[0], _TRAIN[1])]
    assert runner._fill_ticks == _set("4h", ["BTCUSDT", "ETHUSDT"])
    pickle.loads(pickle.dumps(runner))  # the worker pool pickles it


def test_screening_sweep_missing_data_aborts_at_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe_quant.screening.pipeline import create_screening_pipeline

    monkeypatch.setattr(
        "vibe_quant.data.fill_ticks.resolve_fill_ticks", _Resolver(MissingBarDataError("no 1m"))
    )
    dsl = validate_strategy_dict(yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text()))
    with pytest.raises(MissingBarDataError):
        create_screening_pipeline(dsl, symbols=["BTCUSDT"], start_date=_TRAIN[0], end_date=_TRAIN[1])


def _db(tmp_path: Path, dsl: dict[str, Any]) -> Path:
    db = tmp_path / "s.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT, dsl_config TEXT);
        CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, strategy_id INTEGER, symbols TEXT,
                                    timeframe TEXT);
        """
    )
    conn.execute("INSERT INTO strategies VALUES (1,'s',?)", (json.dumps(dsl),))
    conn.execute("INSERT INTO backtest_runs VALUES (42,1,?,?)", (json.dumps(["BTCUSDT"]), "4h"))
    conn.commit()
    conn.close()
    return db


def _dsl4h() -> dict[str, Any]:
    raw = yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text())
    return validate_strategy_dict(raw).model_dump()  # type: ignore[no-any-return]


class _Capture:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, **kw: Any) -> Any:
        self.kwargs.append(kw)
        return lambda params: type("M", (), {"sharpe_ratio": 1.0, "total_return": 0.1})()


def test_wfa_runner_passes_set_and_resolves_per_span(
    resolver: _Resolver, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import date

    from vibe_quant.overfitting.nt_runner import NTWFARunner

    cap = _Capture()
    monkeypatch.setattr(nt_runner, "NTScreeningRunner", cap)
    runner = NTWFARunner(42, _db(tmp_path, _dsl4h()))
    runner.backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})
    runner.backtest("x", date(2024, 2, 5), date(2024, 2, 20), {})  # inside: reuse
    runner.backtest("x", date(2024, 1, 1), date(2024, 2, 10), {})  # widens
    assert [c[2:] for c in resolver.calls] == [
        ("2024-02-01", "2024-03-01"),
        ("2024-01-01", "2024-03-01"),
    ]
    assert all(k["fill_ticks"] == _set("4h", ["BTCUSDT"]) for k in cap.kwargs)
    assert len(cap.kwargs) == 3


def test_cv_runner_passes_set(
    resolver: _Resolver, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    cap = _Capture()
    monkeypatch.setattr(nt_runner, "NTScreeningRunner", cap)
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda self: None)
    runner = NTPurgedKFoldRunner(42, _db(tmp_path, _dsl4h()), catalog_path=tmp_path / "catalog")
    runner._backtest("2024-02-01", "2024-03-01")
    runner.bind_params({"a": 1})._backtest("2024-02-05", "2024-02-20")
    assert len(resolver.calls) == 1
    assert [k["fill_ticks"] for k in cap.kwargs] == [_set("4h", ["BTCUSDT"])] * 2


def test_auto_screen_passes_explicit_set(
    resolver: _Resolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe_quant.research import auto_screen

    cap = _Capture()
    monkeypatch.setattr(nt_runner, "NTScreeningRunner", cap)
    auto_screen._run_single_metrics(_dsl4h(), *_TRAIN)
    assert len(resolver.calls) == 1
    assert cap.kwargs[0]["fill_ticks"] == _set("4h", list(auto_screen.DEFAULT_SYMBOLS))
