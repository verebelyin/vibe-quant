"""vibe-quant-8vc7h: a guarded position close checkpoints immediately.

A SIGKILL between the close and the next periodic checkpoint (<= 60 s) loses the
losing streak the guard just booked, so the next session under-counts it. The
checkpoint must run AFTER the guard has updated ``consecutive_losses`` and
``evaluate_risk`` has run (the guard books the streak after its listener callback),
which is what ``on_position_close_settled`` gives us.

The real ``TradingGuard._on_position_event`` is driven with a SimpleNamespace
stand-in (the F2 pattern of ``test_paper_trade_alerts.py``) and REAL NT events.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_paper_halt_restore import _trace_node
from tests.unit.test_paper_halt_restore import seeded as _restore_seeded
from tests.unit.test_paper_trade_alerts import _Trip
from vibe_quant.paper.guard import TradingGuard
from vibe_quant.paper.node import _GuardBridge
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    return _restore_seeded.__wrapped__(tmp_path)  # type: ignore[attr-defined,no-any-return]


def _close_guard(
    node: Any,
    trace: Any,
    *,
    losses: int,
    guarded: bool = True,
) -> SimpleNamespace:
    """The real ``_on_position_event`` dependencies, with every step traced."""
    return SimpleNamespace(
        _risk=SimpleNamespace(consecutive_losses=losses),
        _listener=_GuardBridge(node),
        _is_guarded_strategy=lambda _sid: guarded,
        _stop_exited_strategies=lambda: trace.calls.append(("stop_exited",)),
        evaluate_risk=lambda: trace.calls.append(("evaluate_risk",)),
    )


def test_close_checkpoints_after_losing_streak_is_booked(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint is LAST and records the streak the close just booked."""
    node, trace = _trace_node(monkeypatch, seeded)
    guard = _close_guard(node, trace, losses=0)
    monkeypatch.setattr(
        node,
        "_save_checkpoint_now",
        lambda: trace.calls.append(("checkpoint", guard._risk.consecutive_losses)),
    )

    TradingGuard._on_position_event(guard, _Trip().closed)  # type: ignore[arg-type]

    assert [call[0] for call in trace.calls] == [
        "write",
        "stop_exited",
        "evaluate_risk",
        "checkpoint",
    ]
    assert trace.calls[-1][1] == 1, "the checkpoint saw the updated streak, not 0"


def test_winning_close_resets_streak_and_still_checkpoints(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A win resets a running streak to 0 and checkpoints exactly once."""
    node, trace = _trace_node(monkeypatch, seeded)
    guard = _close_guard(node, trace, losses=3)
    monkeypatch.setattr(
        node,
        "_save_checkpoint_now",
        lambda: trace.calls.append(("checkpoint", guard._risk.consecutive_losses)),
    )

    TradingGuard._on_position_event(guard, _Trip(exit_px="67700.10").closed)  # type: ignore[arg-type]

    checkpoints = [call for call in trace.calls if call[0] == "checkpoint"]
    assert len(checkpoints) == 1
    assert checkpoints[0][1] == 0
    assert trace.calls[-1][0] == "checkpoint"


def test_unguarded_close_does_not_checkpoint(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unguarded close books no streak, so it must not trigger the checkpoint."""
    node, trace = _trace_node(monkeypatch, seeded)
    guard = _close_guard(node, trace, losses=0, guarded=False)
    monkeypatch.setattr(
        node, "_save_checkpoint_now", lambda: trace.calls.append(("checkpoint",))
    )

    TradingGuard._on_position_event(guard, _Trip().closed)  # type: ignore[arg-type]

    assert [call[0] for call in trace.calls] == ["stop_exited", "evaluate_risk"]


def test_checkpoint_row_holds_the_updated_streak(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real persistence: the written row carries the booked streak."""
    node, trace = _trace_node(monkeypatch, seeded)
    guard = _close_guard(node, trace, losses=0)
    trader_id = node._config.trader_id
    persistence = StatePersistence(seeded["db"], trader_id)
    node._persistence = persistence
    monkeypatch.setattr(
        node,
        "_capture_checkpoint",
        lambda: StateCheckpoint(
            trader_id=trader_id,
            node_status={"consecutive_losses": guard._risk.consecutive_losses},
        ),
    )
    try:
        TradingGuard._on_position_event(guard, _Trip().closed)  # type: ignore[arg-type]
        assert node._db_worker.submit(lambda: None).result(5) is None

        latest = persistence.load_latest_checkpoint()
        assert latest is not None
        assert latest.node_status["consecutive_losses"] == 1
    finally:
        node._db_worker.begin_stop(persistence.close).result(5)
        persistence.close()


class _ExplodingPersistence:
    def save_checkpoint(self, _checkpoint: StateCheckpoint) -> None:
        raise RuntimeError("ZZBOOM")


def test_failing_checkpoint_never_breaks_the_close(
    seeded: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising store is logged class-only and never escapes the guard callback."""
    node, trace = _trace_node(monkeypatch, seeded)
    guard = _close_guard(node, trace, losses=0)
    node._persistence = _ExplodingPersistence()  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR):
        TradingGuard._on_position_event(guard, _Trip().closed)  # type: ignore[arg-type]
        assert node._db_worker.submit(lambda: None).result(5) is None

    assert guard._risk.consecutive_losses == 1
    assert "immediate checkpoint failed (RuntimeError)" in caplog.text
    assert "ZZBOOM" not in caplog.text, "class-only log"
