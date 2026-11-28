"""A screening reference must share the validation run's fee schedule (R28).

The same strategy at another fee schedule (the BNB discount) is a different cost model,
not an overfitting signal, so the consistency check skips such a reference the way it
skips one at other sizing. A legacy run that recorded no fee keys ran at VIP0; a run
whose recorded schedule cannot be read is skipped, never fatal and never matched.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.fees import FeeSchedule
from vibe_quant.validation.consistency import find_screening_reference

if TYPE_CHECKING:
    from pathlib import Path

WINDOW = ("2024-01-01", "2025-01-01")
SYMBOLS = ["BTCUSDT"]
VIP0 = FeeSchedule()
BNB = FeeSchedule(bnb_discount=True)


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    s = StateManager(db_path=tmp_path / "t.db")
    yield s
    s.close()


def _screening(state: StateManager, sid: int, parameters: dict[str, object], sharpe: float) -> int:
    """A completed screening run over WINDOW/SYMBOLS with one sweep row."""
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="screening",
        symbols=SYMBOLS,
        timeframe="4h",
        start_date=WINDOW[0],
        end_date=WINDOW[1],
        parameters=parameters,
    )
    state.update_backtest_run_status(run_id, "completed")
    state.conn.execute(
        "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
        " VALUES (?, ?, ?, ?)",
        (run_id, json.dumps({}), sharpe, 50),
    )
    state.conn.commit()
    return run_id


def _ref(state: StateManager, sid: int, name: str = "s1", **kwargs: object) -> object:
    return find_screening_reference(
        state,
        sid,
        name,
        val_window=WINDOW,
        val_symbols=SYMBOLS,
        **kwargs,  # type: ignore[arg-type]
    )


def test_reference_run_must_share_the_fee_schedule(state: StateManager) -> None:
    """A BNB validation skips the newer VIP0 run for the older BNB one, and vice versa."""
    sid = state.create_strategy("s1", {"name": "s1"})
    bnb = _screening(state, sid, {"bnb_discount": True}, 1.5)
    vip0 = _screening(state, sid, {}, 9.0)
    assert vip0 > bnb

    with_bnb = _ref(state, sid, val_fee=BNB)
    assert with_bnb is not None
    assert (with_bnb.source, with_bnb.sharpe) == (f"screening_run:{bnb}", 1.5)

    plain = _ref(state, sid, val_fee=VIP0)
    assert plain is not None
    assert (plain.source, plain.sharpe) == (f"screening_run:{vip0}", 9.0)

    # val_fee=None keeps the legacy behaviour: no fee filter, the newest run wins.
    unfiltered = _ref(state, sid)
    assert unfiltered is not None
    assert unfiltered.source == f"screening_run:{vip0}"


def test_legacy_run_without_fee_keys_is_vip0(state: StateManager) -> None:
    """No fee keys stored = VIP0 without BNB; an explicit VIP0 tier reads the same."""
    sid = state.create_strategy("s1", {"name": "s1"})
    legacy = _screening(state, sid, {"initial_balance": 1000}, 2.0)
    ref = _ref(state, sid, val_fee=VIP0)
    assert ref is not None
    assert ref.source == f"screening_run:{legacy}"
    assert _ref(state, sid, val_fee=BNB) is None

    explicit = _screening(state, sid, {"fee_tier": "VIP0", "bnb_discount": False}, 3.0)
    ref2 = _ref(state, sid, val_fee=VIP0)
    assert ref2 is not None
    assert ref2.source == f"screening_run:{explicit}"


@pytest.mark.parametrize(
    "parameters",
    [{"fee_tier": "VIP12"}, {"fee_tier": ["VIP0"]}, {"bnb_discount": "yes"}],
    ids=["unknown-tier", "list-tier", "bnb-not-bool"],
)
def test_unreadable_recorded_fee_is_skipped(
    state: StateManager, parameters: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable stored schedule skips its run (newer, higher Sharpe) for the good one."""
    sid = state.create_strategy("s1", {"name": "s1"})
    good = _screening(state, sid, {}, 2.0)
    bad = _screening(state, sid, parameters, 9.9)
    assert bad > good

    with caplog.at_level(logging.DEBUG, logger="vibe_quant.validation.consistency"):
        ref = _ref(state, sid, val_fee=VIP0)
    assert ref is not None
    assert ref.source == f"screening_run:{good}", "an unreadable fee schedule was matched"
    assert "Reference run has an unreadable fee schedule; skipping it" in caplog.text
    # The skip is logged without the exception's own text.
    assert "unknown fee tier" not in caplog.text
    assert "must be a bool" not in caplog.text


def _discovery(state: StateManager, notes: dict[str, object], sharpe: float) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=SYMBOLS,
        timeframe="4h",
        start_date=WINDOW[0],
        end_date=WINDOW[1],
        parameters={},
    )
    state.update_backtest_run_status(run_id, "completed")
    entry = {"dsl": {"name": "genome_abc123"}, "sharpe": sharpe, "trades": 64}
    state.save_backtest_result(
        run_id, {"sharpe_ratio": sharpe, "notes": json.dumps({**notes, "top_strategies": [entry]})}
    )
    return run_id


def test_discovery_fallback_must_share_the_fee_schedule(state: StateManager) -> None:
    """Same rules on the discovery-notes path: ``notes.fee_schedule`` absent = VIP0."""
    sid = state.create_strategy("genome_abc123", {"name": "genome_abc123"})
    bnb = _discovery(state, {"fee_schedule": {"fee_tier": "VIP0", "bnb_discount": True}}, 1.1)
    legacy = _discovery(state, {}, 1.2)
    unreadable = _discovery(state, {"fee_schedule": {"fee_tier": "VIP12"}}, 1.3)
    not_object = _discovery(state, {"fee_schedule": "VIP0"}, 1.4)
    assert bnb < legacy < unreadable < not_object

    with_bnb = find_screening_reference(state, sid, "genome_abc123", val_fee=BNB)
    assert with_bnb is not None
    assert (with_bnb.source, with_bnb.sharpe) == (f"discovery_run:{bnb}", 1.1)

    plain = find_screening_reference(state, sid, "genome_abc123", val_fee=VIP0)
    assert plain is not None
    assert (plain.source, plain.sharpe) == (f"discovery_run:{legacy}", 1.2)

    # No filter: the newest run wins, whatever it recorded.
    unfiltered = find_screening_reference(state, sid, "genome_abc123")
    assert unfiltered is not None
    assert unfiltered.source == f"discovery_run:{not_object}"


def test_runner_check_filters_by_the_runs_fee_schedule(state: StateManager) -> None:
    """The production path: the runner compares a BNB validation against the BNB run only."""
    from types import SimpleNamespace

    from vibe_quant.validation.runner import ValidationRunner

    sid = state.create_strategy("s1", {"name": "s1"})
    bnb = _screening(state, sid, {"bnb_discount": True}, 3.0)
    vip0 = _screening(state, sid, {}, 9.0)
    runner = object.__new__(ValidationRunner)
    runner._state = state  # type: ignore[attr-defined]
    cfg: dict[str, object] = {
        "parameters": {"bnb_discount": True},
        "symbols": SYMBOLS,
        "start_date": WINDOW[0],
        "end_date": WINDOW[1],
    }

    def check(fee: FeeSchedule) -> dict[str, object] | None:
        return runner._check_screening_consistency(  # type: ignore[no-any-return]
            999,
            sid,
            "s1",
            SimpleNamespace(sharpe_ratio=-2.0, total_trades=50),  # type: ignore[arg-type]
            run_config=cfg,
            fee=fee,
        )

    report = check(BNB)
    assert report is not None
    assert (report["screen_sharpe"], report["screen_source"]) == (3.0, f"screening_run:{bnb}")
    report_vip0 = check(VIP0)
    assert report_vip0 is not None
    assert (report_vip0["screen_sharpe"], report_vip0["screen_source"]) == (
        9.0,
        f"screening_run:{vip0}",
    )
