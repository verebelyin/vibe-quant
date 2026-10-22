"""One Telegram alert per symbol when the guard's entry check keeps raising (bead vibe-quant-ii8et).

When ``TradingGuard._checked_denial_reason`` catches an exception it denies the order with
``f"{GATE_DENIAL_PREFIX}: entry check failed ({type(exc).__name__}); order denied"``
(``ENTRY_CHECK_FAILED_PREFIX``). A strategy that keeps retrying then produces that denial
on every bar, so ``PaperTradingNode._on_order_denied`` must alert ONCE per symbol and
re-arm the latch only when an entry for that symbol passes the margin limit again
(``GuardListener.on_entry_allowed``) -- the same latch pattern the margin denial uses.

Fixtures reused: ``_trace_node``/``_Trace`` and ``seeded`` come from
``tests/unit/test_paper_halt_restore.py`` (the same pair
``test_paper_guard_margin.py::test_margin_alert_once_per_symbol`` uses) and ``_denied`` is
copied from ``test_paper_guard_margin.py``; ``_guard``/``_Cache``/``_Order`` come from
``test_paper_guard_margin.py`` for the guard-level reason-literal pin.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_guard_margin import ETH, _Cache, _guard, _Order
from tests.unit.test_paper_halt_restore import _trace_node
from tests.unit.test_paper_halt_restore import seeded as _restore_seeded
from vibe_quant.logging.events import EventType
from vibe_quant.paper import guard as guard_module
from vibe_quant.paper.guard import (
    ENTRY_CHECK_FAILED_PREFIX,
    GATE_DENIAL_PREFIX,
)
from vibe_quant.paper.node import PaperTradingNode, _GuardBridge

if TYPE_CHECKING:
    from pathlib import Path

BTC = "BTCUSDT-PERP.BINANCE"
#: ``node._ns_to_dt(1_700_000_000_000_000_000)`` -- the denial event's ``ts_init``.
_DENIED_TS = datetime.fromtimestamp(1_700_000_000, tz=UTC)
#: The byte-for-byte reason the guard builds for a raising entry check (pinned literal).
_PREFIX_LITERAL = "VQ_GUARD: entry check failed ("


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    return _restore_seeded.__wrapped__(tmp_path)  # type: ignore[attr-defined,no-any-return]


def _denied(symbol: str, reason: str) -> Any:
    return SimpleNamespace(
        client_order_id="C-1",
        instrument_id=symbol,
        strategy_id="S-1",
        reason=reason,
        ts_init=1_700_000_000_000_000_000,
    )


def _entry_check_reason(class_name: str = "RuntimeError") -> str:
    return f"{_PREFIX_LITERAL}{class_name}); order denied"


def _alerts(trace: Any) -> list[Any]:
    return [c for c in trace.calls if c[0] == "alert"]


def _expected_alert(reason: str, symbol: str) -> tuple[Any, ...]:
    return ("alert", "error", f"Entry check failed (order denied): {reason}\n{symbol}", False)


# ------------------------------------------------------------- E1: prefix export


def test_entry_check_prefix_is_the_gate_prefix_plus_its_literal() -> None:
    """E1: the exported prefix is ``GATE_DENIAL_PREFIX`` plus the exact literal, so the
    muk2e reason is built from a named constant."""
    expected = f"{GATE_DENIAL_PREFIX}: entry check failed ("
    assert GATE_DENIAL_PREFIX == "VQ_GUARD"
    assert ENTRY_CHECK_FAILED_PREFIX == _PREFIX_LITERAL
    assert expected == ENTRY_CHECK_FAILED_PREFIX
    assert "ENTRY_CHECK_FAILED_PREFIX" in guard_module.__all__


# ------------------------------------------------- E2: one alert per symbol (latch)


def test_entry_check_failure_alerts_once_per_symbol(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2: ETH denial, ETH denial, BTC denial, ETH allowed, ETH denial -> alerts 1, 0, 1, -, 2."""
    node, trace = _trace_node(monkeypatch, seeded)
    bridge = _GuardBridge(node)
    eth, btc = ETH.__str__(), BTC
    reason = _entry_check_reason()

    node._on_order_denied(_denied(eth, reason), by_gate=True)
    assert _alerts(trace) == [_expected_alert(reason, eth)]
    assert trace.calls[0] == (
        "write",
        EventType.ORDER,
        {
            "action": "denied",
            "by_guard": True,
            "order_id": "C-1",
            "symbol": eth,
            "strategy_id": "S-1",
            "reason": reason,
        },
        _DENIED_TS,
    ), "the ORDER event is written with its full payload and ts_init before the alert"

    node._on_order_denied(_denied(eth, reason), by_gate=True)
    assert len(_alerts(trace)) == 1, "the second failure of the same symbol does not alert"
    assert len([c for c in trace.calls if c[0] == "write"]) == 2, "but it is still written"
    assert eth in node._entry_check_alerted, "the latch holds the symbol"

    node._on_order_denied(_denied(btc, reason), by_gate=True)
    assert len(_alerts(trace)) == 2, "another symbol has its own latch"
    assert _alerts(trace)[1] == _expected_alert(reason, btc)

    bridge.on_entry_allowed(eth)
    node._on_order_denied(_denied(eth, reason), by_gate=True)
    assert len(_alerts(trace)) == 3, "an allowed entry re-arms the entry-check latch"
    assert _alerts(trace)[2] == _expected_alert(reason, eth)
    assert eth in node._entry_check_alerted


# --------------------------------------------------------------- E3: re-arm both


def test_entry_allowed_discards_the_symbol_from_both_latches(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """E3: ``on_entry_allowed`` clears the symbol from the margin AND entry-check latches."""
    node, _trace = _trace_node(monkeypatch, seeded)
    bridge = _GuardBridge(node)
    symbol = ETH.__str__()
    node._margin_alerted.add(symbol)
    node._entry_check_alerted.add(symbol)

    bridge.on_entry_allowed(symbol)

    assert symbol not in node._margin_alerted
    assert symbol not in node._entry_check_alerted


# ------------------------------------------------------------- E4: unchanged rows


def test_margin_denial_behaviour_unchanged(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """E4/E5: a margin denial still alerts once with kind "warning" and its own latch."""
    node, trace = _trace_node(monkeypatch, seeded)
    symbol = ETH.__str__()
    reason = f"{GATE_DENIAL_PREFIX}: margin limit: margin in use 2060 + new 100 > limit 2000.0"

    node._on_order_denied(_denied(symbol, reason), by_gate=True)

    assert _alerts(trace) == [
        ("alert", "warning", f"Entry refused (margin limit): {reason}\n{symbol}", False)
    ]
    assert symbol in node._margin_alerted
    assert symbol not in node._entry_check_alerted, "the entry-check latch is untouched"


def test_other_gate_denial_still_writes_without_alerting(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """E4/E5: a gate denial that is neither a margin nor an entry-check failure only writes."""
    node, trace = _trace_node(monkeypatch, seeded)
    node._on_order_denied(
        _denied(ETH.__str__(), f"{GATE_DENIAL_PREFIX}: trading halted (manual)"), by_gate=True
    )
    assert [c[0] for c in trace.calls] == ["write"]


# ------------------------------------------- E1/E5: the muk2e reason text is unchanged


@pytest.mark.parametrize(
    ("exc_cls", "class_name"),
    [
        pytest.param(RuntimeError, "RuntimeError", id="runtime-error"),
        pytest.param(ZeroDivisionError, "ZeroDivisionError", id="zero-division"),
    ],
)
def test_guard_reason_literal_is_built_from_the_prefix(
    monkeypatch: pytest.MonkeyPatch, exc_cls: type[Exception], class_name: str
) -> None:
    """E1/E5: the guard still produces the byte-for-byte muk2e reason, naming the RAISED
    class, built from ``ENTRY_CHECK_FAILED_PREFIX``."""
    g = _guard(_Cache(), {})

    def _boom(_order: Any) -> str | None:
        raise exc_cls("boom")

    monkeypatch.setattr(g, "_denial_reason", _boom)

    assert g._checked_denial_reason(_Order("O-1", ETH)) == (
        f"{_PREFIX_LITERAL}{class_name}); order denied"
    )


def test_denied_event_still_carries_the_full_payload_and_ts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2/E5: the alerting branch writes before it alerts, with the ts_init-derived
    timestamp (a dropped timestamp/field or an early return fails here)."""
    node, trace = _trace_node(monkeypatch, seeded)
    reason = _entry_check_reason()

    node._on_order_denied(_denied(BTC, reason), by_gate=True)

    assert [c[0] for c in trace.calls] == ["write", "alert"]
    assert trace.calls[0][3] == _DENIED_TS
    assert trace.calls[1] == _expected_alert(reason, BTC)
    assert isinstance(node, PaperTradingNode)


# --------------------- E2 (round 2): the latch holds while the event write keeps failing


def test_entry_check_alert_latch_holds_when_the_event_write_fails(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2 (risk-gr-alert-1): with the REAL ``_write_then_alert`` and a persistently raising
    ``record`` on the node's journal (disk full), 20 entry-check denials for one symbol
    still alert exactly ONCE -- the latch is set BEFORE the write, so a failed write
    cannot re-arm it. The other tests here go through ``_trace_node`` (a ``FakeJournal``
    that accepts every write) and so cannot see this."""
    node = PaperTradingNode(_config(seeded))
    alerts: list[tuple[str, str, bool]] = []
    attempts: list[Any] = []
    error = OSError("disk full")

    def raising_write(
        event_type: Any, data: dict[str, object], *, alert: Any = None, timestamp: Any = None
    ) -> None:
        attempts.append((event_type, data, timestamp))
        raise error

    monkeypatch.setattr(node._journal, "record", raising_write)
    monkeypatch.setattr(
        node._journal,
        "alert",
        lambda alert: alerts.append((alert.kind, alert.message, alert.critical)),
    )

    reason = _entry_check_reason()
    symbol = ETH.__str__()

    for _ in range(20):
        with pytest.raises(OSError) as caught:
            node._on_order_denied(_denied(symbol, reason), by_gate=True)
        assert caught.value is error, "the write's own error keeps propagating (retry semantics)"

    assert len(attempts) == 20, "every denial still attempts the write"
    assert alerts == [("error", f"Entry check failed (order denied): {reason}\n{symbol}", False)]
    assert symbol in node._entry_check_alerted

    # An entry for the symbol that passes again re-arms the latch even while the writer
    # keeps failing, so the NEXT failure alerts anew (one more alert, not a second storm).
    _GuardBridge(node).on_entry_allowed(symbol)
    assert symbol not in node._entry_check_alerted
    with pytest.raises(OSError):
        node._on_order_denied(_denied(symbol, reason), by_gate=True)
    assert len(alerts) == 2, "the allowed entry re-armed the latch despite the failing writer"
