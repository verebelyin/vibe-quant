"""Contract tests for the shared engine-run lifecycle (rabmo.4 E1).

The node/engine doubles are local on purpose: ``tests/unit/fake_engine.py`` has no
kernel log guard, analyzer, ``reset`` or ``get_result``, and E4a/E4b own its migration.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vibe_quant.engine_run import (
    SCREENING_ENGINE,
    VALIDATION_ENGINE,
    EnginePolicy,
    EngineRun,
    EngineSpec,
    bar_data_config,
    loaded_timeframes,
    run_engine,
)
from vibe_quant.validation.venue import VenueConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Engine:
    """Fake NT engine: kernel log guard + analyzer, ``reset``, ``get_result``; no ``trader``."""

    def __init__(
        self, calls: list[Any], name: str, *, reset_error: BaseException | None = None
    ) -> None:
        self.name = name
        self._calls = calls
        self._reset_error = reset_error
        self.kernel = SimpleNamespace(
            get_log_guard=lambda: None,
            portfolio=SimpleNamespace(analyzer=SimpleNamespace(register_statistic=self._register)),
        )

    def _register(self, stat: object) -> None:
        self._calls.append(("register", self.name, type(stat).__name__, id(stat)))

    def reset(self) -> None:
        self._calls.append(("reset", self.name))
        if self._reset_error is not None:
            raise self._reset_error

    def get_result(self) -> object:
        self._calls.append(("get_result", self.name))
        return SimpleNamespace(marker=self.name)


def _node_factory(
    calls: list[Any],
    engines: list[object],
    *,
    engine: object | None = None,
    seen: dict[str, Any] | None = None,
    init_error: BaseException | None = None,
    build_error: BaseException | None = None,
) -> type:
    class _Node:
        def __init__(self, configs: object) -> None:
            if init_error is not None:
                raise init_error
            self.configs = configs
            if seen is not None:
                seen["configs"] = configs

        def build(self) -> None:
            calls.append(("build",))
            if build_error is not None:
                raise build_error

        def run(self) -> None:
            calls.append(("run",))

        def get_engines(self) -> list[object]:
            return list(engines)

        def get_engine(self, run_config_id: str) -> object | None:
            calls.append(("get_engine", run_config_id))
            return engine

        def dispose(self) -> None:
            calls.append(("dispose",))

    return _Node


def _spec(
    *,
    policy: EnginePolicy = SCREENING_ENGINE,
    cleanup_dirs: tuple[Path, ...] = (),
    venue: VenueConfig | None = None,
) -> EngineSpec:
    return EngineSpec(
        strategy_configs=(),
        data_configs=(),
        venue=venue if venue is not None else VenueConfig(),
        start="2024-01-01",
        end="2024-07-01",
        policy=policy,
        cleanup_dirs=cleanup_dirs,
    )


# ----------------------------------------------------------------- R36: teardown


def test_teardown_runs_when_consume_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R36: a raising consumer still resets, disposes and cleans every dir, in order."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    monkeypatch.setattr(nt_node, "BacktestNode", _node_factory(calls, [engine], engine=engine))
    cleaned: list[Path] = []
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", cleaned.append)

    def _consume(_run: EngineRun) -> object:
        calls.append(("consume",))
        raise ValueError("consumer failed")

    spec = _spec(cleanup_dirs=(tmp_path / "a", tmp_path / "b"))
    with pytest.raises(ValueError, match="consumer failed"):
        run_engine(spec, _consume)

    assert calls.index(("reset", "e1")) < calls.index(("dispose",))
    assert cleaned == [tmp_path / "a", tmp_path / "b"]


def test_cleanup_failure_replaces_consume_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R36: a plain ``finally`` — the cleanup error replaces the consumer's error."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    monkeypatch.setattr(nt_node, "BacktestNode", _node_factory(calls, [engine], engine=engine))

    def _cleanup(_path: Path) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", _cleanup)

    def _consume(_run: EngineRun) -> object:
        raise ValueError("consumer failed")

    with pytest.raises(OSError, match="cleanup failed"):
        run_engine(_spec(cleanup_dirs=(tmp_path / "a",)), _consume)


def test_teardown_runs_when_run_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R36: ``node.run()`` raising still tears down every engine and every dir."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    node_cls = _node_factory(calls, [engine], engine=engine)

    def _run_raises(self: object) -> None:
        calls.append(("run",))
        raise RuntimeError("engine run failed")

    monkeypatch.setattr(node_cls, "run", _run_raises)
    monkeypatch.setattr(nt_node, "BacktestNode", node_cls)
    cleaned: list[Path] = []
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", cleaned.append)

    def _consume(_run: EngineRun) -> object:
        raise AssertionError("consume must not run")

    with pytest.raises(RuntimeError, match="engine run failed"):
        run_engine(_spec(cleanup_dirs=(tmp_path / "a",)), _consume)

    assert ("reset", "e1") in calls
    assert ("dispose",) in calls
    assert cleaned == [tmp_path / "a"]


def test_build_failure_still_tears_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R36: ``node.build()`` raising (partial engines exist) still resets, disposes, cleans."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    monkeypatch.setattr(
        nt_node,
        "BacktestNode",
        _node_factory(calls, [engine], engine=engine, build_error=OSError("build failed")),
    )
    cleaned: list[Path] = []
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", cleaned.append)

    with pytest.raises(OSError, match="build failed"):
        run_engine(_spec(cleanup_dirs=(tmp_path / "a",)), lambda _run: pytest.fail("no consume"))

    assert [c[0] for c in calls] == ["build", "reset", "dispose"]
    assert cleaned == [tmp_path / "a"]


def test_reset_failure_does_not_stop_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R36: a raising ``reset`` on engine 1 is suppressed; engine 2, dispose, cleanup still run."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    first = _Engine(calls, "e1", reset_error=RuntimeError("reset failed"))
    second = _Engine(calls, "e2")
    monkeypatch.setattr(
        nt_node, "BacktestNode", _node_factory(calls, [first, second], engine=first)
    )
    cleaned: list[Path] = []
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", cleaned.append)

    assert run_engine(_spec(cleanup_dirs=(tmp_path / "a",)), lambda _run: "kept") == "kept"

    assert ("reset", "e2") in calls
    assert calls.index(("reset", "e2")) < calls.index(("dispose",))
    assert cleaned == [tmp_path / "a"]


def test_log_guard_retained_before_reset_and_dispose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each engine's log guard is retained, before its reset and before node disposal."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engines: list[object] = [_Engine(calls, "e1"), _Engine(calls, "e2")]
    monkeypatch.setattr(nt_node, "BacktestNode", _node_factory(calls, engines, engine=engines[0]))
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", lambda _p: None)
    monkeypatch.setattr(
        "vibe_quant.nt_compat.retain_log_guard", lambda eng: calls.append(("retain", eng.name))
    )

    run_engine(_spec(), lambda _run: None)

    for name in ("e1", "e2"):
        assert calls.index(("retain", name)) + 1 == calls.index(("reset", name))
        assert calls.index(("reset", name)) < calls.index(("dispose",))
    assert [c for c in calls if isinstance(c, tuple) and c[0] == "retain"] == [
        ("retain", "e1"),
        ("retain", "e2"),
    ]


# ------------------------------------------------- R37: node construction point


def test_node_constructor_failure_skips_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R37: the node is built before the ``try`` — a constructor error tears down nothing."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    monkeypatch.setattr(
        nt_node,
        "BacktestNode",
        _node_factory(calls, [], init_error=RuntimeError("no node")),
    )
    cleaned: list[Path] = []
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", cleaned.append)

    with pytest.raises(RuntimeError, match="no node"):
        run_engine(_spec(cleanup_dirs=(tmp_path / "a",)), lambda _run: None)

    assert calls == []
    assert cleaned == []


# ------------------------------------------------------------- R23 / R24 / R39


def test_statistics_registered_in_policy_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """R23: policy order, on every engine, one instance per class shared across engines."""
    import nautilus_trader.backtest.node as nt_node

    for policy, expected in (
        (SCREENING_ENGINE, ["SharpeRatio", "SortinoRatio", "WinRate", "ProfitFactor"]),
        (
            VALIDATION_ENGINE,
            [
                "SharpeRatio",
                "SortinoRatio",
                "WinRate",
                "ProfitFactor",
                "Expectancy",
                "LongRatio",
                "AvgWinner",
                "AvgLoser",
                "MaxDrawdown",
            ],
        ),
    ):
        calls: list[Any] = []
        engines: list[object] = [_Engine(calls, "e1"), _Engine(calls, "e2")]
        monkeypatch.setattr(
            nt_node, "BacktestNode", _node_factory(calls, engines, engine=engines[0])
        )
        monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", lambda _p: None)

        consumed: list[EngineRun] = []
        run_engine(_spec(policy=policy), consumed.append)

        for name in ("e1", "e2"):
            names = [
                c[2] for c in calls if isinstance(c, tuple) and c[0] == "register" and c[1] == name
            ]
            assert names == expected
        # The SAME statistic instances reach both engines.
        first = [
            c[3] for c in calls if isinstance(c, tuple) and c[0] == "register" and c[1] == "e1"
        ]
        second = [
            c[3] for c in calls if isinstance(c, tuple) and c[0] == "register" and c[1] == "e2"
        ]
        assert first == second
        assert [c for c in calls if c == ("reset", "e1")] == [("reset", "e1")]
        assert [c for c in calls if c == ("reset", "e2")] == [("reset", "e2")]


def test_engine_none_reaches_consume_and_teardown_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R24: ``get_engine`` None is the consumer's decision; ``get_result`` is never read."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    monkeypatch.setattr(nt_node, "BacktestNode", _node_factory(calls, [engine], engine=None))
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", lambda _p: None)

    received: list[EngineRun] = []
    assert run_engine(_spec(), received.append) is None

    assert len(received) == 1
    assert received[0].engine is None
    assert received[0].bt_result is None
    assert received[0].run_config_id
    assert ("get_result", "e1") not in calls
    assert ("reset", "e1") in calls
    assert ("dispose",) in calls


def test_engine_without_trader_reaches_consume(monkeypatch: pytest.MonkeyPatch) -> None:
    """R39: the lifecycle never reads ``engine.trader`` (margin reads stay in the tiers)."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    assert not hasattr(engine, "trader")
    monkeypatch.setattr(nt_node, "BacktestNode", _node_factory(calls, [engine], engine=engine))
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", lambda _p: None)

    received: list[EngineRun] = []
    run_engine(_spec(), received.append)

    run = received[0]
    assert run.engine is engine
    assert run.bt_result == SimpleNamespace(marker="e1")
    assert not hasattr(run.engine, "trader")


# ------------------------------------------------------- lifecycle + config shape


def test_lifecycle_call_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build -> register -> prepare -> run -> get_engine -> get_result -> consume -> teardown."""
    import nautilus_trader.backtest.node as nt_node

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    monkeypatch.setattr(nt_node, "BacktestNode", _node_factory(calls, [engine], engine=engine))
    monkeypatch.setattr(
        "vibe_quant.data.catalog.cleanup_epoch_parquet",
        lambda path: calls.append(("cleanup", path)),
    )

    def _prepare(node: object) -> None:
        calls.append(("prepare", type(node).__name__))

    def _consume(run: EngineRun) -> str:
        calls.append(("consume", run.run_config_id))
        return "done"

    one_stat = EnginePolicy(("SharpeRatio",), "VIBE_QUANT_TEST_LOG_LEVEL", "WARNING")
    assert (
        run_engine(_spec(policy=one_stat, cleanup_dirs=(tmp_path,)), _consume, prepare=_prepare)
        == "done"
    )

    assert [c[0] for c in calls] == [
        "build",
        "register",
        "prepare",
        "run",
        "get_engine",
        "get_result",
        "consume",
        "reset",
        "dispose",
        "cleanup",
    ]


def test_log_level_env_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine's log level is ``os.environ.get(policy.log_level_env, default)``."""
    import nautilus_trader.backtest.node as nt_node

    def _logging_for(policy: EnginePolicy, env: dict[str, str]) -> str:
        for name in (SCREENING_ENGINE.log_level_env, VALIDATION_ENGINE.log_level_env):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        calls: list[Any] = []
        engine = _Engine(calls, "e1")
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            nt_node,
            "BacktestNode",
            _node_factory(calls, [engine], engine=engine, seen=seen),
        )
        monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", lambda _p: None)
        run_engine(_spec(policy=policy), lambda _run: None)
        run_config = seen["configs"][0]
        return str(run_config.engine.logging.log_level)

    assert _logging_for(SCREENING_ENGINE, {}) == "WARNING"
    assert _logging_for(VALIDATION_ENGINE, {}) == "INFO"
    assert _logging_for(SCREENING_ENGINE, {"VIBE_QUANT_NT_LOG_LEVEL_SCREENING": "DEBUG"}) == "DEBUG"
    assert _logging_for(VALIDATION_ENGINE, {"VIBE_QUANT_NT_LOG_LEVEL": "ERROR"}) == "ERROR"


def test_run_config_flags_and_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flags, one venue, and ``data``/``strategies`` forwarded in the given order."""
    import nautilus_trader.backtest.node as nt_node
    from nautilus_trader.config import BacktestDataConfig, ImportableStrategyConfig
    from nautilus_trader.model.data import Bar

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        nt_node, "BacktestNode", _node_factory(calls, [engine], engine=engine, seen=seen)
    )
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", lambda _p: None)

    strategies = (
        ImportableStrategyConfig(strategy_path="m:A", config_path="m:AC", config={}),
        ImportableStrategyConfig(strategy_path="m:B", config_path="m:BC", config={}),
    )
    data = tuple(
        BacktestDataConfig(
            catalog_path="/tmp/c",
            data_cls=Bar,
            bar_types=[bar_type],
            start_time="2024-01-01",
            end_time="2024-07-01",
        )
        for bar_type in (
            "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL",
            "ETHUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL",
        )
    )
    spec = EngineSpec(
        strategy_configs=strategies,
        data_configs=data,
        venue=VenueConfig(),
        start="2023-02-03",
        end="2024-03-04",
        policy=VALIDATION_ENGINE,
        cleanup_dirs=(),
    )
    run_engine(spec, lambda _run: None)

    configs = seen["configs"]
    assert len(configs) == 1
    run_config = configs[0]
    assert run_config.dispose_on_completion is False
    assert run_config.raise_exception is True
    assert len(run_config.venues) == 1
    assert str(run_config.venues[0].name) == "BINANCE"
    assert str(run_config.start) == "2023-02-03"
    assert str(run_config.end) == "2024-03-04"
    assert [s.config_path for s in run_config.engine.strategies] == ["m:AC", "m:BC"]
    assert [cfg.bar_types[0] for cfg in run_config.data] == [
        "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL",
        "ETHUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL",
    ]


@pytest.mark.parametrize(("balance", "leverage"), [(4321, Decimal("7")), (9876, Decimal("3"))])
def test_custom_venue_forwarded_and_analysis_on(
    monkeypatch: pytest.MonkeyPatch, balance: int, leverage: Decimal
) -> None:
    """The supplied venue (not a default) reaches the run config; analysis stays on."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.validation.venue as venue_mod

    calls: list[Any] = []
    engine = _Engine(calls, "e1")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        nt_node, "BacktestNode", _node_factory(calls, [engine], engine=engine, seen=seen)
    )
    monkeypatch.setattr("vibe_quant.data.catalog.cleanup_epoch_parquet", lambda _p: None)
    inputs: list[VenueConfig] = []
    real = venue_mod.create_backtest_venue_config

    def _spy(config: VenueConfig) -> Any:
        inputs.append(config)
        return real(config)

    monkeypatch.setattr(venue_mod, "create_backtest_venue_config", _spy)

    venue = VenueConfig(starting_balance_usdt=balance, default_leverage=leverage)
    run_engine(_spec(venue=venue), lambda _run: None)

    assert len(inputs) == 1
    assert inputs[0] is venue
    run_config = seen["configs"][0]
    assert run_config.engine.run_analysis is True
    assert list(run_config.venues[0].starting_balances) == [f"{balance} USDT"]
    assert run_config.venues[0].default_leverage == float(leverage)


# ----------------------------------------------------- R-iso and the two helpers


def test_import_loads_no_nautilus() -> None:
    """R-iso: importing the module (and ``loaded_timeframes``) stays NT-free."""
    code = textwrap.dedent(
        """
        import sys

        from vibe_quant.engine_run import loaded_timeframes  # noqa: F401

        leaked = sorted(m for m in sys.modules if m.startswith("nautilus_trader"))
        print(leaked)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout + completed.stderr


def test_loaded_timeframes_matches_nt_runner() -> None:
    """The copy answers exactly like the screening original it will replace in E2."""
    import vibe_quant.screening.nt_runner as ntr

    class _Indicator:
        def __init__(self, timeframe: str | None) -> None:
            self.timeframe = timeframe

    cases: list[dict[str, Any]] = [
        {"timeframe": "4h"},
        {"timeframe": "4h", "additional_timeframes": ["1d", "1h"]},
        {"timeframe": "4h", "indicators": {"rsi": {"timeframe": "1m"}}},
        {"timeframe": "4h", "indicators": {"rsi": _Indicator("15m")}},
        {"timeframe": "1h", "additional_timeframes": None, "indicators": None},
        {"timeframe": "1d", "indicators": {}},
    ]
    for dsl in cases:
        assert loaded_timeframes(dsl) == ntr.loaded_timeframes(dsl), dsl
    assert loaded_timeframes({"timeframe": "4h", "additional_timeframes": ["1d"]}) == {"4h", "1d"}


def test_bar_data_config_fields() -> None:
    """One bar data config: catalog string, the ``Bar`` CLASS, one bar type, the window."""
    from nautilus_trader.model.data import Bar

    config = bar_data_config(
        "/abs/catalog",
        "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        "2024-01-01",
        "2024-07-01",
    )
    assert config.catalog_path == "/abs/catalog"
    assert config.data_cls is Bar
    assert config.bar_types == ["BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"]
    assert config.start_time == "2024-01-01"
    assert config.end_time == "2024-07-01"

    other = bar_data_config(
        "/x", "ETHUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL", "2020-01-01", "2021-01-01"
    )
    assert other.bar_types == ["ETHUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL"]
    assert other.start_time == "2020-01-01"
