"""Discovery gets one leverage setting, independent of liquidation.

Contract rows owned here:

- C6: the default (no flag) leaves every runner at 10 and the notes untouched
  (no ``leverage`` / ``liquidation`` key at all).
- C7: ``--leverage 3`` reaches every ``NTBacktestFn`` (training, holdout,
  cross-window, WFA factories) and is recorded as ``notes.leverage``.
- R4: an explicit ``--leverage 10`` is the default and writes no note.
- R7: with the mode on the leverage lives only in ``notes.liquidation``.
- A1: two disagreeing flags fail the run with a ``LiquidationConfigError``.

``--liquidation-leverage`` stays as a deprecated alias (it must still work).

regression: vibe-quant-6wq6w
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import Any

import pytest  # noqa: TCH002

from tests.unit.test_discovery_liquidation import (
    _chrom,
    _create_run,
    _fit,
    _result,
    _run_cli,
)
from vibe_quant.discovery.backtest_fn import NTBacktestFn  # noqa: TCH001
from vibe_quant.discovery.pipeline import DiscoveryResult, HoldoutResult

# Drives the training fn, the post-search factory (cross-window / WFA), the
# holdout and the full-range headline in one run.
_EVERY_RUNNER_ARGS = [
    "--start-date",
    "2024-01-01",
    "--end-date",
    "2025-01-01",
    "--eval-windows",
    "2",
    "--cross-window-months",
    "1",
    "--train-test-split",
    "0.8",
]


def _holdout() -> HoldoutResult:
    return HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2,
        total_trades=30, total_return=0.02,
    )  # fmt: skip


def _nt_call(self: Any, chrom: Any) -> dict[str, float | int]:
    return {
        "sharpe_ratio": 1.0, "total_return": 0.05, "max_drawdown": 0.05,
        "profit_factor": 1.5, "total_trades": 100,
    }  # fmt: skip


def _drives_every_runner(chrom: Any, fit: Any) -> Any:
    def run_fn(self: Any) -> DiscoveryResult:
        assert self._backtest_fn_factory is not None
        self._backtest_fn_factory("2024-02-01", "2024-03-01")
        return _result(
            top=[(chrom, fit)], holdout_results=[_holdout()],
            train_dates=("2024-01-01", "2024-11-01"),
            holdout_dates=("2024-11-01", "2025-01-01"),
        )

    return run_fn


# ---------------------------------------------------------------------------
# C6 / C7 / R4 / R7 -- the CLI flag
# ---------------------------------------------------------------------------


def test_default_run_runners_and_notes_are_ten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6: no flag -> every runner at 10, no leverage/liquidation in the notes."""
    db = tmp_path / "c6.db"
    run_id = _create_run(db)

    built: list[NTBacktestFn] = []
    rc, run, notes = _run_cli(
        db, run_id, list(_EVERY_RUNNER_ARGS), monkeypatch,
        _drives_every_runner(_chrom(), _fit()),
        nt_call=_nt_call, record_built=built,
    )  # fmt: skip
    assert rc == 0
    assert run["status"] == "completed"
    assert built
    assert all(fn.liquidation.leverage == 10.0 for fn in built)
    assert notes is not None
    assert "leverage" not in notes
    assert "liquidation" not in notes


def test_cli_leverage_reaches_every_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C7: --leverage 3, off -> every runner at 3 and notes.leverage == 3.0."""
    db = tmp_path / "c7.db"
    run_id = _create_run(db)

    built: list[NTBacktestFn] = []
    rc, run, notes = _run_cli(
        db, run_id, [*_EVERY_RUNNER_ARGS, "--leverage", "3"], monkeypatch,
        _drives_every_runner(_chrom(), _fit()),
        nt_call=_nt_call, record_built=built,
    )  # fmt: skip
    assert rc == 0
    assert run["status"] == "completed"
    assert built
    assert all(fn.liquidation.leverage == 3.0 for fn in built)
    spans = {(fn.start_date, fn.end_date) for fn in built}
    assert ("2024-02-01", "2024-03-01") in spans  # cross-window / WFA factory
    assert notes is not None
    assert notes["leverage"] == 3.0
    assert "liquidation" not in notes


def test_deprecated_alias_still_sets_off_leverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1/Q2: --liquidation-leverage keeps working, with the same off-mode effect."""
    db = tmp_path / "alias.db"
    run_id = _create_run(db)

    built: list[NTBacktestFn] = []
    rc, _run, notes = _run_cli(
        db, run_id, [*_EVERY_RUNNER_ARGS, "--liquidation-leverage", "3"], monkeypatch,
        _drives_every_runner(_chrom(), _fit()),
        nt_call=_nt_call, record_built=built,
    )  # fmt: skip
    assert rc == 0
    assert built and all(fn.liquidation.leverage == 3.0 for fn in built)
    assert notes is not None and notes["leverage"] == 3.0


def test_explicit_default_leverage_leaves_notes_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --leverage 10 is the default; no notes.leverage key."""
    db = tmp_path / "r4.db"
    run_id = _create_run(db)

    built: list[NTBacktestFn] = []
    rc, _run, notes = _run_cli(
        db, run_id, [*_EVERY_RUNNER_ARGS, "--leverage", "10"], monkeypatch,
        _drives_every_runner(_chrom(), _fit()),
        nt_call=_nt_call, record_built=built,
    )  # fmt: skip
    assert rc == 0
    assert built and all(fn.liquidation.leverage == 10.0 for fn in built)
    assert notes is not None
    assert "leverage" not in notes
    assert "liquidation" not in notes


def test_mode_on_keeps_leverage_in_liquidation_notes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mode on at 20 -> notes.liquidation == {mode, leverage}; no top-level key."""
    db = tmp_path / "r7.db"
    run_id = _create_run(db)

    rc, run, notes = _run_cli(
        db, run_id,
        [*_EVERY_RUNNER_ARGS, "--liquidation-mode", "isolated", "--leverage", "20"],
        monkeypatch, _drives_every_runner(_chrom(), _fit()), nt_call=_nt_call,
    )  # fmt: skip
    assert rc == 0
    assert run["status"] == "completed"
    assert notes is not None
    assert notes["liquidation"] == {"mode": "isolated", "leverage": 20.0}
    assert "leverage" not in notes


def test_conflicting_leverage_flags_fail_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1: --leverage 3 with --liquidation-leverage 20 fails, naming both."""
    db = tmp_path / "a1.db"
    run_id = _create_run(db)

    rc, run, _notes = _run_cli(
        db, run_id, ["--leverage", "3", "--liquidation-leverage", "20"],
        monkeypatch, _drives_every_runner(_chrom(), _fit()),
    )  # fmt: skip
    assert rc == 1
    assert run["status"] == "failed"
    assert "LiquidationConfigError" in run["error_message"]
    assert "conflicting leverage 3.0 and liquidation_leverage 20.0" in run["error_message"]


def test_conflicting_equal_leverage_flags_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1: both given but equal (3 vs 3.0) is fine -- the shared rule normalises."""
    db = tmp_path / "a1eq.db"
    run_id = _create_run(db)

    built: list[NTBacktestFn] = []
    rc, _run, notes = _run_cli(
        db, run_id,
        [*_EVERY_RUNNER_ARGS, "--leverage", "3", "--liquidation-leverage", "3.0"],
        monkeypatch, _drives_every_runner(_chrom(), _fit()),
        nt_call=_nt_call, record_built=built,
    )  # fmt: skip
    assert rc == 0
    assert built and all(fn.liquidation.leverage == 3.0 for fn in built)
    assert notes is not None and notes["leverage"] == 3.0
