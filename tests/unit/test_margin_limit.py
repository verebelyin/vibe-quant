"""The ONE margin rule shared by the paper guard and the compiled template (bead vibe-quant-9r0m5).

Binding contract rows exercised here: R20/R20b (equality AT the limit is allowed; the
non-terminating 1000/3 + 500/3 lands exactly on 500 only because the sum is a Fraction),
R21 (one ulp over is refused, message verbatim), R21b (heterogeneous per-item leverage),
R21c (an unverified leverage refuses, BEFORE the equity checks), R22 (equity unknown),
R23 (equity 0 / -5), R24 (every invalid input is a ``MarginLimitError`` naming the field),
R25 (pct 1.0 at 1x) and R25b (the pending union by ``client_order_id``).

The rule's callers arrive in T4 (paper guard) and T5 (generated backtest code); it is a pure
stdlib module, so every fake here is a plain record with no NT import.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import pytest

from vibe_quant.margin_limit import (
    DEFAULT_PAPER_MAX_MARGIN_PCT,
    MarginDecision,
    MarginItem,
    MarginLimitError,
    check_entry_margin,
    unique_pending_entries,
)


def _item(notional: str, leverage: str | None, label: str = "BTCUSDT-PERP.BINANCE") -> MarginItem:
    """One open position / pending entry / candidate entry as the rule sees it."""
    return MarginItem(
        notional=Decimal(notional),
        leverage=None if leverage is None else Decimal(leverage),
        label=label,
    )


def _check(
    open_items: list[MarginItem],
    pending_items: list[MarginItem],
    new_item: MarginItem,
    *,
    equity: Decimal | None = Decimal("1000"),
    max_margin_pct: Decimal = Decimal("0.5"),
) -> MarginDecision:
    return check_entry_margin(
        open_items=open_items,
        pending_items=pending_items,
        new_item=new_item,
        equity=equity,
        max_margin_pct=max_margin_pct,
    )


def test_equality_allowed() -> None:
    """R20: 400 in use + 100 new is EXACTLY the 500 limit, so the entry is allowed."""
    decision = _check(
        [_item("2000", "10"), _item("2000", "10")],
        [],
        _item("1000", "10"),
    )
    assert decision.allowed is True
    assert decision.reason is None
    assert decision.in_use == Decimal("400")
    assert decision.new_margin == Decimal("100")
    assert decision.limit == Decimal("500.0")


def test_equality_exact_non_terminating() -> None:
    """R20b: 1000/3 + 500/3 == 500 exactly — Decimal division would round below the limit."""
    decision = _check([_item("1000", "3")], [], _item("500", "3"))
    assert decision.allowed is True
    assert decision.reason is None
    # The claim the rule rests on: the sum is a rational, and it is the limit exactly.
    assert Fraction(Decimal("1000")) / 3 + Fraction(Decimal("500")) / 3 == Fraction(500)


def test_just_over_refused() -> None:
    """R21: one ulp above the limit is refused, with the verbatim reason."""
    decision = _check(
        [_item("2000", "10"), _item("2000", "10")],
        [],
        _item("1000.0000001", "10"),
    )
    assert decision.allowed is False
    assert decision.reason == (
        "margin in use 400 + new 100.00000001 > limit 500.0 (50% of equity 1000)"
    )


def test_heterogeneous_leverage() -> None:
    """R21b: each item keeps its OWN leverage (4000@5x = 800, 1000@10x = 100)."""
    decision = _check([_item("4000", "5")], [], _item("1000", "10"))
    assert decision.allowed is False
    assert decision.reason == "margin in use 800 + new 100 > limit 500.0 (50% of equity 1000)"
    assert decision.in_use == Decimal("800")
    assert decision.new_margin == Decimal("100")


@pytest.mark.parametrize("where", ["open", "pending", "new"])
def test_unverified_leverage_refused(where: str) -> None:
    """R21c: an item whose verified leverage is unknown refuses, wherever it sits."""
    unverified = _item("500", None, "ETHUSDT-PERP.BINANCE")
    open_items: list[MarginItem] = []
    pending_items: list[MarginItem] = []
    new_item = _item("100", "10")
    if where == "open":
        open_items = [unverified]
    elif where == "pending":
        pending_items = [unverified]
    else:
        new_item = unverified
    decision = _check(open_items, pending_items, new_item)
    assert decision.allowed is False
    assert decision.reason == "leverage unverified for ETHUSDT-PERP.BINANCE"


def test_unverified_leverage_precedes_equity_checks() -> None:
    """R21c: an unknown leverage refuses BEFORE the equity checks, even with no equity."""
    decision = _check([_item("500", None, "ETHUSDT-PERP.BINANCE")], [], _item("100", "10"), equity=None)
    assert decision.allowed is False
    assert decision.reason == "leverage unverified for ETHUSDT-PERP.BINANCE"


def test_unknown_equity_refused() -> None:
    """R22: no account equity -> refuse (never treat it as 0 or as unlimited)."""
    decision = _check([_item("100", "10")], [], _item("100", "10"), equity=None)
    assert decision.allowed is False
    assert decision.reason == "equity unknown"


@pytest.mark.parametrize(
    ("equity", "reason"),
    [("0", "equity 0 <= 0"), ("-5", "equity -5 <= 0")],
)
def test_non_positive_equity_refused(equity: str, reason: str) -> None:
    """R23: zero and negative equity refuse with the value quoted."""
    decision = _check([], [], _item("100", "10"), equity=Decimal(equity))
    assert decision.allowed is False
    assert decision.reason == reason


def test_invalid_inputs_raise() -> None:
    """R24: pct outside (0, 1], leverage outside [1, 125] and a negative notional raise."""
    base: dict[str, object] = {
        "open_items": [],
        "pending_items": [],
        "new_item": _item("100", "10"),
        "equity": Decimal("1000"),
        "max_margin_pct": Decimal("0.5"),
    }
    cases: list[tuple[dict[str, object], str]] = [
        ({"max_margin_pct": Decimal("0")}, "max_margin_pct"),
        ({"max_margin_pct": Decimal("1.01")}, "max_margin_pct"),
        ({"max_margin_pct": Decimal("NaN")}, "max_margin_pct"),
        ({"new_item": _item("100", "0.5")}, "leverage"),
        ({"new_item": _item("100", "126")}, "leverage"),
        ({"open_items": [_item("100", "126")]}, "leverage"),
        ({"new_item": _item("-1", "10")}, "notional"),
    ]
    for override, field in cases:
        with pytest.raises(MarginLimitError) as excinfo:
            check_entry_margin(**{**base, **override})  # type: ignore[arg-type]
        assert field in str(excinfo.value)


def test_invalid_inputs_are_value_errors() -> None:
    """The guard turns the error into its own denial path through the ValueError branch."""
    assert issubclass(MarginLimitError, ValueError)
    with pytest.raises(ValueError):
        _check([], [], _item("100", "10"), max_margin_pct=Decimal("2"))


def test_full_equity_at_1x() -> None:
    """R25: at 1x the whole equity may be the position, exactly at the limit."""
    decision = _check([], [], _item("1000", "1"), max_margin_pct=Decimal("1"))
    assert decision.allowed is True
    assert decision.reason is None
    assert decision.limit == Decimal("1000")
    assert decision.new_margin == Decimal("1000")


def test_leverage_bounds_inclusive() -> None:
    """R24: 1 and 125 are valid leverages; 0.5 and 126 are not (equality edge kept)."""
    for leverage in ("1", "125"):
        assert _check([], [], _item("100", leverage)).allowed is True


def test_zero_notional_allowed() -> None:
    """R24: a zero notional is valid (0 >= 0) and contributes no margin."""
    decision = _check([], [], _item("0", "10"))
    assert decision.allowed is True
    assert decision.new_margin == Decimal("0")


def test_unique_pending_union() -> None:
    """R25b: union by client_order_id; reduce-only and the excluded id are dropped."""
    pending_cancel = SimpleNamespace(client_order_id="C1", is_reduce_only=False)
    pending_update = SimpleNamespace(client_order_id="U1", is_reduce_only=False)
    only_open = SimpleNamespace(client_order_id="O1", is_reduce_only=False)
    reduce_only = SimpleNamespace(client_order_id="R1", is_reduce_only=True)
    excluded = SimpleNamespace(client_order_id="X1", is_reduce_only=False)

    # PENDING_CANCEL and PENDING_UPDATE live in BOTH sets; each must count exactly once.
    open_orders = [pending_cancel, only_open, reduce_only, excluded]
    inflight_orders = [pending_cancel, pending_update]

    assert unique_pending_entries(
        open_orders, inflight_orders, exclude_client_order_id="X1"
    ) == [pending_cancel, only_open, pending_update]


def test_unique_pending_empty() -> None:
    """R25b: nothing pending, nothing excluded."""
    assert unique_pending_entries([], [], exclude_client_order_id=None) == []


def test_default_paper_max_margin_pct() -> None:
    """The paper/live default cushion the guard falls back to (R33 in T4)."""
    assert isinstance(DEFAULT_PAPER_MAX_MARGIN_PCT, Decimal)
    assert Decimal("0.5") == DEFAULT_PAPER_MAX_MARGIN_PCT


def test_excess_below_display_precision_refused() -> None:
    """R21: 1000/3 + (500 + 1e-30)/3 is over 500 by far less than a Decimal displays; still refused."""
    decision = _check([_item("1000", "3")], [], _item("500.000000000000000000000000000001", "3"))
    assert decision.allowed is False
    assert decision.reason is not None
    assert decision.reason.startswith("margin in use")


def test_verified_pending_margin_counts() -> None:
    """R21b: a pending entry with verified leverage is what pushes the candidate over the limit."""
    open_items = [_item("300", "1")]
    pending = [_item("150", "1")]
    new_item = _item("100", "1")

    refused = _check(open_items, pending, new_item)
    assert refused.allowed is False
    assert refused.in_use == Decimal("450")

    allowed = _check(open_items, [], new_item)
    assert allowed.allowed is True
    assert allowed.in_use == Decimal("300")


def test_unique_pending_same_client_id_first_seen_wins() -> None:
    """R25b: distinct records sharing a client_order_id count once; the first one is kept."""
    first = SimpleNamespace(client_order_id="D1", is_reduce_only=False)
    second = SimpleNamespace(client_order_id="D1", is_reduce_only=False)
    third = SimpleNamespace(client_order_id="D1", is_reduce_only=False)

    result = unique_pending_entries([first, second], [third], exclude_client_order_id=None)

    assert len(result) == 1
    assert result[0] is first
