"""The discovery engine adapters and the state-backed record store.

Edge tests of the ``BacktestAdapter`` / ``RecordStore`` seams (bead
vibe-quant-rabmo.7, step T14): ``NTBacktests``/``MockBacktests`` build every cell
function from THIS module's globals (so a patched ``run.NTBacktestFn`` reaches
the train, holdout and factory functions), and ``StateRecordStore`` writes the
same rows ``main()`` writes today.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _cell_kwargs(**over: object) -> dict[str, object]:
    from vibe_quant.launch_settings import LaunchSettings

    kwargs: dict[str, object] = {
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "symbol_agg": "portfolio",
        "launch": LaunchSettings(),
        "train_start": "2024-01-01",
        "train_end": "2024-06-01",
        "eval_windows": None,
        "min_trades": 0,
        "holdout": ("2024-06-01", "2024-08-01"),
        "needs_factory": True,
        "evaluator": "nt",
    }
    kwargs.update(over)
    return kwargs


def _nt_fn_kwargs(**over: object) -> dict[str, object]:
    from vibe_quant.launch_settings import LaunchSettings

    kwargs: dict[str, object] = {
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "symbol_agg": "portfolio",
        "launch": LaunchSettings(),
    }
    kwargs.update(over)
    return kwargs


class _FakeNT:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


def test_nt_backtestfn_patch_reaches_every_cell_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run.NTBacktestFn patched to a fake is what train, holdout and factory build."""
    monkeypatch.setattr(run, "NTBacktestFn", _FakeNT)

    adapter = run.NTBacktests()
    fns = adapter.cell_fns(**_cell_kwargs())

    assert isinstance(fns.backtest_fn, _FakeNT)
    assert isinstance(fns.holdout_backtest_fn, _FakeNT)
    assert fns.backtest_fn_factory is not None
    assert isinstance(fns.backtest_fn_factory("2024-01-01", "2024-03-01"), _FakeNT)
    assert isinstance(adapter.nt_fn(**_nt_fn_kwargs()), _FakeNT)
    assert adapter.mock is False


@pytest.mark.parametrize(("max_position_pct", "risk_per_trade"), [(0.15, 0.01), (0.4, 0.03)])
def test_nt_backtests_forward_the_run_sizing(
    monkeypatch: pytest.MonkeyPatch, max_position_pct: float, risk_per_trade: float
) -> None:
    """Every engine fn the NT adapter builds carries the run's sizing, not the defaults."""
    from vibe_quant.launch_settings import LaunchSettings
    from vibe_quant.sizing_settings import SizingSettings

    monkeypatch.setattr(run, "NTBacktestFn", _FakeNT)
    sizing = SizingSettings(max_position_pct=max_position_pct, risk_per_trade=risk_per_trade)
    launch = LaunchSettings(sizing=sizing)
    adapter = run.NTBacktests()
    fns = adapter.cell_fns(**_cell_kwargs(launch=launch))
    assert fns.backtest_fn_factory is not None

    built = [
        fns.backtest_fn,
        fns.holdout_backtest_fn,
        fns.backtest_fn_factory("2024-01-01", "2024-03-01"),
        adapter.nt_fn(**_nt_fn_kwargs(launch=launch)),
    ]

    monkeypatch.setattr("vibe_quant.fast_tier.backtest_fn.FastBacktestFn", _FakeNT)
    built.append(adapter.cell_fns(**_cell_kwargs(launch=launch, evaluator="fast")).backtest_fn)

    for fn in built:
        assert isinstance(fn, _FakeNT)
        got = fn.kwargs["sizing"]
        assert (got.max_position_pct, got.risk_per_trade) == (max_position_pct, risk_per_trade)


def test_nt_backtests_unpack_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """R3: the adapter unpacks ONE LaunchSettings into the engine's unchanged kwargs."""
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.launch_settings import LaunchSettings
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    monkeypatch.setattr(run, "NTBacktestFn", _FakeNT)
    adapter = run.NTBacktests()
    launches = [
        LaunchSettings(
            fee_schedule=FeeSchedule(tier="VIP0", bnb_discount=True),
            liquidation=LiquidationSettings(mode="cross", leverage=7.0),
            sizing=SizingSettings(max_position_pct=2.0),
            entry_order="limit",
        ),
        LaunchSettings(
            fee_schedule=FeeSchedule(),
            liquidation=LiquidationSettings(mode="isolated", leverage=3.0),
            sizing=SizingSettings(risk_per_trade=0.01),
            entry_order="market",
        ),
    ]
    for launch in launches:
        fns = adapter.cell_fns(**_cell_kwargs(launch=launch))
        assert fns.backtest_fn_factory is not None
        built = [
            fns.backtest_fn,
            fns.holdout_backtest_fn,
            fns.backtest_fn_factory("2024-01-01", "2024-03-01"),
            adapter.nt_fn(**_nt_fn_kwargs(launch=launch)),
        ]
        for fn in built:
            assert isinstance(fn, _FakeNT)
            assert fn.kwargs["fee_schedule"] == launch.fee_schedule
            assert fn.kwargs["entry_order"] == launch.entry_order
            assert fn.kwargs["liquidation"] == launch.liquidation
            assert fn.kwargs["sizing"] == launch.sizing


def test_mock_backtests_hand_back_the_mock_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mock adapter's cell fns are the module's _mock_backtest, not the engine."""
    sentinel = object()
    monkeypatch.setattr(run, "_mock_backtest", sentinel)

    adapter = run.MockBacktests()
    fns = adapter.cell_fns(**_cell_kwargs())

    assert adapter.mock is True
    assert fns.backtest_fn is sentinel
    assert fns.holdout_backtest_fn is sentinel
    assert fns.backtest_fn_factory is not None
    assert fns.backtest_fn_factory("2024-01-01", "2024-03-01") is sentinel
    with pytest.raises(RuntimeError, match="mock adapter runs no NT backtest"):
        adapter.nt_fn(**_nt_fn_kwargs())


def _seeded_run(db: Path) -> tuple[StateManager, BacktestJobManager, int]:
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", {}
    )
    jobs.conn.execute(
        """INSERT INTO background_jobs (run_id, pid, job_type, status)
           VALUES (?, ?, 'discovery', 'running')""",
        (run_id, os.getpid()),
    )
    jobs.conn.commit()
    return state, jobs, run_id


def _job_row(state: StateManager, run_id: int) -> tuple[str, str | None]:
    row = state.conn.execute(
        "SELECT status, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return (row[0], row[1])


def test_mark_completed_records_both_rows(tmp_path: Path) -> None:
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        run.StateRecordStore(state, jobs).mark_completed(run_id)

        row = state.get_backtest_run(run_id)
        assert row is not None
        assert row["status"] == "completed"
        assert _job_row(state, run_id) == ("completed", None)
    finally:
        state.close()
        jobs.close()


def test_mark_failed_records_the_error_on_both_rows(tmp_path: Path) -> None:
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        run.StateRecordStore(state, jobs).mark_failed(run_id, "boom")

        row = state.get_backtest_run(run_id)
        assert row is not None
        assert row["status"] == "failed"
        assert row["error_message"] == "boom"
        assert _job_row(state, run_id) == ("failed", "boom")
    finally:
        state.close()
        jobs.close()


def test_mark_job_failed_on_an_unknown_run_touches_only_the_job(tmp_path: Path) -> None:
    """The run-not-found branch: no run row exists, so only the job is recorded."""
    db = tmp_path / "state.db"
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    missing_run_id = 4242
    jobs.conn.execute(
        """INSERT INTO background_jobs (run_id, pid, job_type, status)
           VALUES (?, ?, 'discovery', 'running')""",
        (missing_run_id, os.getpid()),
    )
    jobs.conn.commit()
    try:
        store = run.StateRecordStore(state, jobs)
        assert store.get_run(missing_run_id) is None

        store.mark_job_failed(missing_run_id, "jobboom")

        assert store.get_run(missing_run_id) is None
        assert _job_row(state, missing_run_id) == ("failed", "jobboom")
    finally:
        state.close()
        jobs.close()


@pytest.mark.parametrize(
    ("metrics", "expected_sharpe", "expected_return"),
    [
        ({"sharpe_ratio": 1.25, "total_return": 0.10, "notes": "{}"}, 1.25, 0.10),
        ({"sharpe_ratio": -0.75, "total_return": -0.30, "notes": "{}"}, -0.75, -0.30),
    ],
)
def test_get_run_and_save_result_round_trip(
    tmp_path: Path,
    metrics: dict[str, object],
    expected_sharpe: float,
    expected_return: float,
) -> None:
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        store = run.StateRecordStore(state, jobs)

        row = store.get_run(run_id)
        assert row is not None
        assert row["id"] == run_id

        store.save_result(run_id, metrics)

        stored = state.get_backtest_result(run_id)
        assert stored is not None
        assert stored["sharpe_ratio"] == expected_sharpe
        assert stored["total_return"] == expected_return
    finally:
        state.close()
        jobs.close()


# --- run_discovery: the run body behind the CLI shell (step T15) -------------------------------

_MOCK_ARGV = [
    "--mock", "--seed", "7", "--population-size", "6", "--max-generations", "2",
    "--max-workers", "-1", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
    "--timeframe", "4h", "--symbols", "BTCUSDT",
]  # fmt: skip


def _args(run_id: int, extra: list[str] | None = None, *, mock: bool = True) -> object:
    from vibe_quant.discovery.run_spec import build_parser

    base = _MOCK_ARGV if mock else [a for a in _MOCK_ARGV if a != "--mock"]
    return build_parser().parse_args([*base, *(extra or []), "--run-id", str(run_id)])


class _RecordingStore:
    """Delegates to the real store and records the order of the status writes."""

    def __init__(self, inner: run.StateRecordStore, *, fail_mark_failed: bool = False) -> None:
        self.inner = inner
        self.state = inner.state
        self.calls: list[str] = []
        self.fail_mark_failed = fail_mark_failed

    def get_run(self, run_id: int) -> dict[str, object] | None:
        return self.inner.get_run(run_id)

    def save_result(self, run_id: int, metrics: dict[str, object]) -> None:
        self.calls.append("save_result")
        self.inner.save_result(run_id, metrics)

    def mark_completed(self, run_id: int) -> None:
        self.calls.append("mark_completed")
        self.inner.mark_completed(run_id)

    def mark_failed(self, run_id: int, error: str) -> None:
        self.calls.append("mark_failed")
        if self.fail_mark_failed:
            raise RuntimeError("db locked SECRET-MARKER")
        self.inner.mark_failed(run_id, error)

    def mark_job_failed(self, run_id: int, error: str) -> None:
        self.calls.append("mark_job_failed")
        self.inner.mark_job_failed(run_id, error)


def _run(
    state: StateManager,
    jobs: BacktestJobManager,
    run_id: int,
    args: object,
    *,
    backtests: object | None = None,
    fail_mark_failed: bool = False,
) -> tuple[run.RunOutcome, _RecordingStore]:  # type: ignore[name-defined]
    store = _RecordingStore(run.StateRecordStore(state, jobs), fail_mark_failed=fail_mark_failed)
    outcome = run.run_discovery(  # type: ignore[attr-defined]
        args, backtests or run.MockBacktests(), store, run_id=run_id, started_at=0.0
    )
    return outcome, store


def drive_run_discovery(
    db: Path, run_id: int, argv: list[str], backtests: run.BacktestAdapter
) -> run.RunOutcome:
    """Drive ``run_discovery`` from an argv list: what ``main()`` does, minus the CLI shell.

    ``argv`` is the same list a test used to set on ``sys.argv`` without the leading
    ``prog``. The run row and its background job row already exist; both managers are
    closed before returning, so a caller asserts through a fresh ``StateManager``.
    """
    from vibe_quant.discovery.run_spec import build_parser

    args = build_parser().parse_args(argv)
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    try:
        return run.run_discovery(
            args, backtests, run.StateRecordStore(state, jobs), run_id=run_id, started_at=0.0
        )
    finally:
        state.close()
        jobs.close()


@pytest.mark.parametrize(
    ("extra", "champions"),
    [([], 0), (["--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate"], 5)],
    ids=["no_champion", "champions"],
)
def test_status_order_complete(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    champions: int,
) -> None:
    """Design R41: result row first, then run completed, then job completed (both save sites)."""
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    events: list[str] = []

    def _record(obj: object, name: str, event: Callable[..., str]) -> None:
        real = getattr(obj, name)

        def wrapper(*args: object, **kwargs: object) -> object:
            events.append(event(*args, **kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(obj, name, wrapper)

    _record(state, "save_backtest_result", lambda *_a, **_k: "save_result")
    _record(state, "update_backtest_run_status", lambda _rid, st, **_k: f"run_{st}")
    _record(jobs, "mark_completed", lambda *_a, **_k: "job_completed")
    try:
        outcome, _store = _run(state, jobs, run_id, _args(run_id, extra))
        assert len((outcome.notes or {}).get("top_strategies", [])) == champions  # type: ignore[arg-type]

        assert events == ["save_result", "run_completed", "job_completed"]
        assert outcome.exit_code == 0
        assert outcome.message.startswith("Discovery complete: ")
        assert capsys.readouterr().out.strip() == outcome.message
        row = state.get_backtest_run(run_id)
        assert row is not None and row["status"] == "completed"
        assert state.get_backtest_result(run_id) is not None
        assert _job_row(state, run_id) == ("completed", None)
    finally:
        state.close()
        jobs.close()


def test_run_not_found_records_only_the_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state, jobs, _ = _seeded_run(tmp_path / "state.db")
    missing = 4242
    jobs.conn.execute(
        "INSERT INTO background_jobs (run_id, pid, job_type, status) VALUES (?, ?, 'discovery', 'running')",
        (missing, os.getpid()),
    )
    jobs.conn.commit()
    try:
        outcome, store = _run(state, jobs, missing, _args(missing))

        assert outcome.exit_code == 1
        assert capsys.readouterr().out == "Run 4242 not found\n"
        assert state.get_backtest_run(missing) is None
        assert state.get_backtest_result(missing) is None
        assert _job_row(state, missing) == ("failed", "Run 4242 not found")
    finally:
        state.close()
        jobs.close()


def test_failure_records_run_and_job_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run, "_check_data_available", lambda symbols: False)
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        outcome, _ = _run(
            state, jobs, run_id, _args(run_id, mock=False), backtests=run.NTBacktests()
        )

        error = (
            "RuntimeError: No catalog bar data for symbols=['BTCUSDT'] — download data first "
            "(Data Management) or pass --mock for a synthetic run"
        )
        assert outcome.exit_code == 1
        captured = capsys.readouterr()
        assert captured.out == f"Discovery failed: {error.split(': ', 1)[1]}\n"
        assert "Traceback (most recent call last)" in captured.err
        row = state.get_backtest_run(run_id)
        assert row is not None
        assert (row["status"], row["error_message"]) == ("failed", error)
        assert _job_row(state, run_id) == ("failed", error)
    finally:
        state.close()
        jobs.close()


def test_drive_run_discovery_completes_the_run_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared driver runs ``run_discovery`` off an argv list and completes the run row."""
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    db = tmp_path / "state.db"
    state, jobs, run_id = _seeded_run(db)
    state.close()
    jobs.close()

    def fake_run(self: object) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[], top_strategies=[], total_candidates_evaluated=7,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)

    outcome = drive_run_discovery(
        db, run_id, [*_MOCK_ARGV, "--db", str(db), "--run-id", str(run_id)], run.MockBacktests()
    )

    assert outcome.exit_code == 0
    state = StateManager(db)
    row = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()
    assert row is not None and row["status"] == "completed"
    assert result is not None
    assert json.loads(result["notes"])["evaluated"] == 7


def test_bankroll_grid_error_exits_2(tmp_path: Path) -> None:
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        outcome, _ = _run(state, jobs, run_id, _args(run_id, ["--bankroll-modes", "isolated"]))

        assert outcome.exit_code == 2
        row = state.get_backtest_run(run_id)
        assert row is not None and row["status"] == "failed"
        assert row["error_message"].startswith("BankrollGridError: ")
    finally:
        state.close()
        jobs.close()


def test_failed_write_failure_logs_class_only_and_still_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(run, "_check_data_available", lambda symbols: False)
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        with caplog.at_level(logging.ERROR, logger="vibe_quant.discovery.run"):
            outcome, _ = _run(
                state,
                jobs,
                run_id,
                _args(run_id, mock=False),
                backtests=run.NTBacktests(),
                fail_mark_failed=True,
            )

        assert outcome.exit_code == 1
        assert f"could not record run {run_id} as failed (RuntimeError)" in caplog.text
        assert "SECRET-MARKER" not in caplog.text
        assert capsys.readouterr().out.startswith("Discovery failed: ")
    finally:
        state.close()
        jobs.close()
