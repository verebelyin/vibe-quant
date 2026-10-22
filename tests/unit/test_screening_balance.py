"""Screening honours ``parameters.initial_balance`` (bead vibe-quant-zpbmo, task Z2).

Binding contract rows: R7 (an invalid stored value fails the run through the
``__main__`` except with the ``BalanceConfigError`` text), R8 (the venue,
``_extract_metrics`` and ``last_starting_balance`` all use the parsed balance),
R9 (no kwarg -> 1000) and R12 (every sweep row, the ``-999`` sentinel included,
carries ``starting_balance = float(parsed)``).

R8 is asserted along the whole path, not only at the parser: the venue and
``_extract_metrics`` (below), the runner and the pipeline ``create_screening_pipeline``
builds, and the rows ``save_results`` writes.

R16/R16b (the cross wallet and the tier-cap failure) live in
``tests/unit/test_screening_liquidation.py``, next to the liquidation fixtures.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.balance_settings import DEFAULT_STARTING_BALANCE_USDT, BalanceConfigError
from vibe_quant.launch_settings import PARSE_ORDER, LaunchSettings
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.screening.types import BacktestMetrics
from vibe_quant.validation.venue import create_backtest_venue_config

if TYPE_CHECKING:
    from pathlib import Path

# No funding archive: these trades are minutes long and cross no settlement, so
# funding is 0 and total_return is exactly NT's pnl% (same seam as
# tests/unit/test_audit_metrics_pf.py).
_NO_ARCHIVE = "/nonexistent/vq_test_funding_archive.db"


def _bt_result(pnl_pct: float, n_positions: int = 0) -> SimpleNamespace:
    """NT BacktestResult stub: ``PnL% (total)`` only, no PnL stats otherwise."""
    return SimpleNamespace(
        total_positions=n_positions,
        elapsed_time=0.0,
        stats_pnls={"USDT": {"PnL% (total)": pnl_pct, "Win Rate": 0.5}},
        stats_returns={},
    )


def _fake_engine() -> SimpleNamespace:
    cache = SimpleNamespace(positions=list, position_snapshots=list, bars=list, accounts=list)
    return SimpleNamespace(kernel=SimpleNamespace(cache=cache))


def _runner(**kwargs: Any) -> NTScreeningRunner:
    return NTScreeningRunner(
        dsl_dict={},
        symbols=["BTCUSDT"],
        start_date="",
        end_date="",
        funding_archive_path=_NO_ARCHIVE,
        **kwargs,
    )


def test_runner_rejects_bad_balance_via_launch() -> None:
    """A bad balance arriving through the runner's ``launch`` fails before any engine.

    The runner takes one :class:`LaunchSettings`; whatever builds it (here the screening
    pipeline's parse table) validates the balance, so ``initial_balance`` 0 can never
    reach a venue.
    """
    launch = LaunchSettings(starting_balance=10000)
    runner = NTScreeningRunner(
        dsl_dict={},
        symbols=["BTCUSDT"],
        start_date="",
        end_date="",
        funding_archive_path=_NO_ARCHIVE,
        launch=launch,
    )
    assert runner.launch is launch
    assert runner.launch.starting_balance == 10000

    with pytest.raises(
        BalanceConfigError,
        match=r"initial_balance must be a whole number in \[1, 1000000000\], got 0",
    ):
        LaunchSettings.from_params(
            {"initial_balance": 0}, PARSE_ORDER["screening_pipeline_pre_tick"]
        )


def test_screening_venue_and_metrics_use_initial_balance() -> None:
    """R8: ``initial_balance`` 10000 -> venue "10000 USDT" and metrics scaled by 10000.

    Probe: NT pnl% 5.0 with zero funding gives total_return 0.05.
    """
    runner = _runner(launch=LaunchSettings(starting_balance=10000))
    assert create_backtest_venue_config(runner._venue_config()).starting_balances == ["10000 USDT"]

    metrics = runner._extract_metrics(
        {}, _bt_result(5.0), _fake_engine(), time.time(), starting_balance=10000.0
    )
    assert runner.last_starting_balance == 10000.0
    assert metrics.total_return == 0.05


def test_default_starting_balance_is_1000_usdt() -> None:
    """R9: no kwarg -> the venue keeps the one default 1000 (exactness 239 holds)."""
    runner = _runner()
    assert runner._venue_config().starting_balance_usdt == DEFAULT_STARTING_BALANCE_USDT == 1000
    assert create_backtest_venue_config(runner._venue_config()).starting_balances == ["1000 USDT"]


def _guarded_engine() -> SimpleNamespace:
    """Engine stand-in that survives ``_run_backtest``'s log-guard/reset cleanup."""
    analyzer = SimpleNamespace(register_statistic=lambda _stat: None)
    kernel = SimpleNamespace(
        get_log_guard=lambda: None, portfolio=SimpleNamespace(analyzer=analyzer)
    )
    return SimpleNamespace(kernel=kernel, trader=SimpleNamespace(strategies=list))


def test_run_backtest_hands_extract_metrics_the_runners_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: ``_run_backtest`` divides by the venue's balance, never by a literal.

    The runner is built at 10000, so the value ``_run_backtest`` passes to
    ``_extract_metrics`` — the divisor of every return — must be 10000.0. A
    hard-coded 1000.0 would leave the venue at ``"10000 USDT"`` (R8) while every
    metric was computed against 1000.
    """
    import nautilus_trader.backtest.node as nt_node

    seen: list[float] = []

    class _FakeNode:
        """``BacktestNode`` stand-in: no engine is built and no data is read."""

        def __init__(self, configs: list[Any]) -> None:
            self._config = configs[0]

        def build(self) -> None:
            return None

        def run(self) -> None:
            return None

        def get_engines(self) -> list[Any]:
            return [_guarded_engine()]

        def get_engine(self, _config_id: Any) -> Any:
            return SimpleNamespace(get_result=lambda: SimpleNamespace(total_orders=0, total_events=0))

        def dispose(self) -> None:
            return None

    def _capture(
        _self: NTScreeningRunner,
        params: dict[str, float | int],
        _bt_result: Any,
        _engine: Any,
        _start_time: float,
        **kwargs: Any,
    ) -> BacktestMetrics:
        seen.append(kwargs["starting_balance"])
        return BacktestMetrics(parameters=params, sharpe_ratio=1.0, total_return=0.1)

    monkeypatch.setattr(nt_node, "BacktestNode", _FakeNode)
    monkeypatch.setattr(
        NTScreeningRunner,
        "_build_run_config",
        lambda _self, _params: (SimpleNamespace(id="probe"), []),
    )
    monkeypatch.setattr(NTScreeningRunner, "_extract_metrics", _capture)

    runner = _runner(launch=LaunchSettings(starting_balance=10000), catalog_path=str(tmp_path))
    # The stubbed config builder skips _ensure_compiled, which is what normally
    # resolves catalog_path into _resolved_catalog_path.
    runner._resolved_catalog_path = tmp_path
    runner._run_backtest({}, time.time())
    assert seen == [10000.0]


def _pipeline_with_mixed_runner(starting_balance: int | None) -> Any:
    """A one-worker pipeline whose sweep has one winner and one failed (sentinel) combo."""
    from vibe_quant.screening.pipeline import ScreeningPipeline

    dsl = _dsl()

    class _MixedRunner:
        def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
            if params.get("rsi.period") == 7:
                raise ValueError("boom")
            return BacktestMetrics(parameters=params, sharpe_ratio=1.0, total_return=0.1)

    kwargs: dict[str, Any] = {"max_workers": 1}
    if starting_balance is not None:
        kwargs["starting_balance"] = starting_balance
    return ScreeningPipeline(dsl=dsl, backtest_runner=_MixedRunner(), **kwargs)


def _dsl() -> Any:
    from vibe_quant.dsl import parse_strategy_string

    return parse_strategy_string(
        """
name: balance_probe
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
sweep:
  rsi.period: [7, 14]
"""
    )


def _saved_rows(tmp_path: Path, starting_balance: int | None) -> list[dict[str, Any]]:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.screening.pipeline import FAILED_BACKTEST_SHARPE

    db = tmp_path / "sweep.db"
    state = StateManager(db)
    try:
        state.create_strategy("balance_probe", {"timeframe": "5m", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-02-01",
            parameters={},
        )
        pipeline = _pipeline_with_mixed_runner(starting_balance)
        pipeline.save_results(pipeline.run(), state, run_id)
        rows = state.get_sweep_results(run_id)
    finally:
        state.close()
    assert any(r["sharpe_ratio"] == FAILED_BACKTEST_SHARPE for r in rows), rows
    return rows


def test_every_sweep_row_carries_the_parsed_balance(tmp_path: Path) -> None:
    """R12: every row of the run, the -999 sentinel included, gets float(parsed)."""
    rows = _saved_rows(tmp_path, 10000)
    assert {row["starting_balance"] for row in rows} == {10000.0}


def test_sweep_rows_default_to_1000_without_the_kwarg(tmp_path: Path) -> None:
    """R9/R12: without the kwarg every row (sentinel included) carries 1000.0."""
    rows = _saved_rows(tmp_path, None)
    assert {row["starting_balance"] for row in rows} == {1000.0}


def test_create_screening_pipeline_threads_the_parsed_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8/R12: the factory hands the parsed balance to BOTH constructors.

    A launch that stored ``parameters={"initial_balance": 10000}`` (the UI form
    sends 10000) must build the runner with ``starting_balance=10000`` and return
    a pipeline at 10000, so the run screens — and records — 10000. Dropping
    either forwarding leaves the run silently at the 1000 default.
    """
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.screening import pipeline as pipeline_mod
    from vibe_quant.screening.pipeline import (
        FAILED_BACKTEST_SHARPE,
        create_screening_pipeline,
    )

    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", lambda *a, **k: None)

    built: list[dict[str, Any]] = []

    class _CapturingRunner:
        """``NTScreeningRunner`` stand-in: records its kwargs, fails one combo."""

        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
            if params.get("rsi.period") == 7:
                raise ValueError("boom")
            return BacktestMetrics(parameters=params, sharpe_ratio=1.0, total_return=0.1)

    monkeypatch.setattr("vibe_quant.screening.nt_runner.NTScreeningRunner", _CapturingRunner)

    pipeline = create_screening_pipeline(
        _dsl(), symbols=["BTCUSDT"], parameters={"initial_balance": 10000}, max_workers=1
    )
    assert [kwargs["launch"].starting_balance for kwargs in built] == [10000]
    assert pipeline.starting_balance == 10000

    db = tmp_path / "factory.db"
    state = StateManager(db)
    try:
        state.create_strategy("balance_probe", {"timeframe": "5m", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-02-01",
            parameters={"initial_balance": 10000},
        )
        pipeline.save_results(pipeline.run(), state, run_id)
        rows = state.get_sweep_results(run_id)
    finally:
        state.close()
    assert any(row["sharpe_ratio"] == FAILED_BACKTEST_SHARPE for row in rows), rows
    assert {row["starting_balance"] for row in rows} == {10000.0}


@pytest.mark.parametrize("stored", [1000.5, 0, "10000", None])
def test_invalid_stored_balance_fails_the_run(
    tmp_path: Path, stored: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """R7: an invalid stored value fails the run through ``__main__`` with the parser's text."""
    import argparse

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.screening.__main__ import cmd_run

    db = tmp_path / "invalid.db"
    state = StateManager(db)
    try:
        state.create_strategy("balance_probe", {"timeframe": "5m", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-02-01",
            parameters={"dsl_override": _dsl().model_dump(), "initial_balance": stored},
        )
        state.update_backtest_run_status(run_id, "running")
    finally:
        state.close()

    assert cmd_run(argparse.Namespace(run_id=run_id, db=str(db))) == 1

    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert run["error_message"].startswith(
        "BalanceConfigError: initial_balance must be a whole number"
    )
    # __main__ logs the failure itself (the except at the end of cmd_run).
    assert "initial_balance must be a whole number" in capsys.readouterr().out


def test_main_prints_the_screening_starting_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run log names the balance every screening run started from."""
    import argparse

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.screening import pipeline as pipeline_mod
    from vibe_quant.screening.__main__ import cmd_run

    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", lambda *a, **k: None)
    # One worker: the stand-in runner is a local class and cannot cross a process pool.
    monkeypatch.setattr(pipeline_mod, "cpu_count", lambda: 1)

    class _OkRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
            return BacktestMetrics(parameters=params, sharpe_ratio=1.0, total_return=0.1)

    monkeypatch.setattr("vibe_quant.screening.nt_runner.NTScreeningRunner", _OkRunner)

    db = tmp_path / "printed.db"
    state = StateManager(db)
    try:
        state.create_strategy("balance_probe", {"timeframe": "5m", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-02-01",
            parameters={"dsl_override": _dsl().model_dump(), "initial_balance": 10000},
        )
    finally:
        state.close()

    assert cmd_run(argparse.Namespace(run_id=run_id, db=str(db))) == 0
    assert "Screening starting balance: 10000 USDT" in capsys.readouterr().out
