"""Bankroll size grid: per-cell sizing under the 50% risk limit, admission, risk facts.

Contract rows C7, C8, C9 and C24a of design 2gyk1.3 (bankroll sizing report).
"""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

import pytest

from vibe_quant.bankroll.stress import (
    RISK_LIMIT,
    CellPlan,
    RowOutcome,
    plan_cells,
    row_risk_facts,
)
from vibe_quant.sizing_settings import MAX_RISK_PER_TRADE, SizingConfigError, SizingSettings

GRID = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 10.0]


def _by_k(plans: list[CellPlan]) -> dict[float, CellPlan]:
    return {p.k: p for p in plans}


def test_cell_params() -> None:
    parent = {"max_position_pct": 0.5, "risk_per_trade": 0.02, "leverage": 10.0}
    plans = _by_k(plan_cells(parent, [2.0, 10.0], "isolated"))

    two = plans[2.0]
    assert (two.status, two.effective_cap, two.effective_risk, two.leverage) == (
        "planned",
        1.0,
        0.04,
        10.0,
    )
    assert two.requested_risk == 0.04
    assert two.risk_held_at_limit is False
    assert two.overrides == {
        "max_position_pct": 1.0,
        "risk_per_trade": 0.04,
        "liquidation_mode": "isolated",
    }

    ten = plans[10.0]
    assert (ten.status, ten.effective_cap, ten.effective_risk, ten.leverage) == (
        "planned",
        5.0,
        0.2,
        10.0,
    )
    assert ten.risk_held_at_limit is False
    assert ten.mode == "isolated"
    # The parent's parameters are never modified.
    assert parent == {"max_position_pct": 0.5, "risk_per_trade": 0.02, "leverage": 10.0}


def test_cell_params_forward_mode_leverage_and_defaults() -> None:
    """Defaults (cap 0.5, risk 0.02, L10) and the requested mode reach every cell."""
    plans = _by_k(plan_cells({}, [3.0], "cross"))
    cell = plans[3.0]
    assert (cell.effective_cap, cell.effective_risk, cell.leverage, cell.mode) == (
        1.5,
        0.02 * 3.0,
        10.0,
        "cross",
    )
    assert cell.overrides["liquidation_mode"] == "cross"
    other = _by_k(plan_cells({"leverage": 20, "max_position_pct": 2.0}, [0.5], "isolated"))[0.5]
    assert (other.effective_cap, other.leverage, other.mode) == (1.0, 20.0, "isolated")


def test_per_cell_admission() -> None:
    parent = {"max_position_pct": 4.0, "leverage": 5.0}
    plans = plan_cells(parent, [2.0, 0.5, 1.0], "cross")
    assert [p.k for p in plans] == [2.0, 0.5, 1.0]
    by_k = _by_k(plans)
    assert by_k[2.0].status == "not_runnable"
    assert by_k[2.0].error == "max_position_pct (8.0) must not exceed leverage (5.0)"
    assert by_k[2.0].overrides == {}
    assert by_k[0.5].status == "planned"
    assert by_k[1.0].status == "planned"
    assert by_k[0.5].error is None

    # cap * k == leverage is runnable (equality edge).
    edge = _by_k(plan_cells({"max_position_pct": 2.5, "leverage": 5.0}, [2.0], "cross"))[2.0]
    assert (edge.status, edge.effective_cap) == ("planned", 5.0)

    # At the maximum leverage 125: cap * k == 125 runs, just above it does not.
    at_max = _by_k(plan_cells({"max_position_pct": 12.5, "leverage": 125}, [10.0], "cross"))
    assert (at_max[10.0].status, at_max[10.0].effective_cap) == ("planned", 125.0)
    above = _by_k(plan_cells({"max_position_pct": 62.50000005, "leverage": 125}, [2.0], "cross"))[
        2.0
    ]
    assert above.effective_cap == 125.0000001
    assert above.status == "not_runnable"
    assert above.error is not None and "max_position_pct" in above.error


def test_per_cell_admission_liquidation_error_is_not_runnable() -> None:
    """A leverage the liquidation parser refuses fails the cell, never the plan."""
    plans = plan_cells({"leverage": 10.0}, [1.0, 2.0], "bogus")  # type: ignore[arg-type]
    assert [p.status for p in plans] == ["not_runnable", "not_runnable"]
    assert plans[0].error is not None and "liquidation_mode must be one of" in plans[0].error


def test_risk_held_at_limit() -> None:
    assert RISK_LIMIT == MAX_RISK_PER_TRADE == 0.5

    # A parent already at the limit: k=2 asks 1.0, runs at 0.50, flagged.
    at_limit = _by_k(plan_cells({"risk_per_trade": 0.5, "max_position_pct": 0.5}, [2.0], "cross"))
    cell = at_limit[2.0]
    assert (cell.requested_risk, cell.effective_risk, cell.effective_cap) == (1.0, 0.5, 1.0)
    assert cell.risk_held_at_limit is True
    assert cell.status == "planned"

    # A parent below the limit whose scaled risk crosses it: 0.30 * 2 = 0.60 -> 0.50.
    crossing = _by_k(plan_cells({"risk_per_trade": 0.3}, [2.0], "cross"))[2.0]
    assert crossing.requested_risk == 0.6
    assert crossing.effective_risk == 0.5
    assert crossing.risk_held_at_limit is True
    assert crossing.overrides["risk_per_trade"] == 0.5

    # Equality: 0.25 * 2 == 0.50 is NOT flagged and runs at exactly 0.50.
    equal = _by_k(plan_cells({"risk_per_trade": 0.25}, [2.0], "cross"))[2.0]
    assert equal.requested_risk == 0.5
    assert (equal.effective_risk, equal.risk_held_at_limit) == (0.5, False)

    # Just below the limit: not flagged, risk scaled exactly.
    below = _by_k(plan_cells({"risk_per_trade": 0.24}, [2.0], "cross"))[2.0]
    assert (below.effective_risk, below.risk_held_at_limit) == (0.48, False)

    # Property: no planned cell asks the sizer for more than the limit, and every planned
    # cell passes the existing sizing admission unchanged.
    for risk in (0.01, 0.02, 0.1, 0.25, 0.3, 0.5):
        for cap in (0.1, 0.5, 1.0):
            for plan in plan_cells(
                {"risk_per_trade": risk, "max_position_pct": cap}, GRID, "cross"
            ):
                assert plan.effective_risk <= RISK_LIMIT
                assert plan.risk_held_at_limit is (risk * plan.k > RISK_LIMIT)
                if plan.status == "planned":
                    SizingSettings.from_params(plan.overrides)
    # The existing risk parser keeps refusing anything above the limit.
    with pytest.raises(SizingConfigError):
        SizingSettings(risk_per_trade=0.5000001)


def _compiled_qty(
    *, equity: float, price: float, stop: float, risk: float, cap: float, min_qty: float
) -> float | None:
    """The compiled template's risk-mode sizing (``dsl/templates.py`` sizing block).

    ``min(equity*risk/stop_distance, equity*cap/price)``, clamped UP to the instrument
    minimum while the minimum fits under the cap, else the entry is skipped (``None``);
    rounded down to whole units (step 1 in these examples).
    """
    size = min(equity * risk / abs(price - stop), equity * cap / price)
    if size < min_qty:
        if min_qty > equity * cap / price:
            return None
        size = min_qty
    return float(math.floor(size))


def test_risk_limit_is_a_sizing_budget_not_a_loss_bound() -> None:
    """The 50% is a budget at the signal price; minimum quantity and next-bar fills can
    lose more, so a row carries no loss-bound field (C9 cases)."""
    # NEARUSDT, balance 10, signal 50, stop 20 (distance 30), risk at the limit (budget 5).
    assert _compiled_qty(equity=10, price=50, stop=20, risk=0.5, cap=4, min_qty=1) is None
    qty = _compiled_qty(equity=10, price=50, stop=20, risk=0.5, cap=8, min_qty=1)
    assert qty == 1.0
    assert qty * (50 - 20) == 30 > 10 * 0.5

    # BTCUSDT balance 1,000, signal 100, fixed 25% stop: cap 4 and 8 both size 20 units;
    # the next-bar fill at 120 arms the stop at 90 and loses 600 > the 500 budget.
    for cap in (4.0, 8.0):
        assert (
            _compiled_qty(equity=1000, price=100, stop=75, risk=0.5, cap=cap, min_qty=0.001) == 20
        )
    fill = 120.0
    assert 20 * (fill - fill * 0.75) == 600 > 1000 * 0.5

    names = {f.name for f in dataclasses.fields(CellPlan)} | {
        f.name for f in dataclasses.fields(RowOutcome)
    }
    assert not [n for n in names if "loss" in n or "bound" in n]


def test_no_loss_cap_wording_in_bankroll_package() -> None:
    root = Path(__file__).resolve().parents[2] / "vibe_quant" / "bankroll"
    pattern = re.compile(r"caps? (the )?loss|max(imum)? loss", re.IGNORECASE)
    hits = [
        f"{path.name}:{n}"
        for path in sorted(root.glob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []


_START = 1_704_067_200_000_000_000  # 2024-01-01T00:00:00Z


def _liq(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mode": "cross",
        "count": 0,
        "bust_ts": None,
        "path_divergence": False,
        "events": [],
    }
    base.update(over)
    return base


def test_time_to_ruin() -> None:
    off = row_risk_facts(RowOutcome(k=1.0, status="ok", liquidation=None), _START)
    assert (off.liquidation_count, off.time_to_first_liquidation_s, off.time_to_ruin_s) == (
        None,
        None,
        None,
    )

    clean = row_risk_facts(RowOutcome(k=1.0, status="ok", liquidation=_liq()), _START)
    assert (clean.liquidation_count, clean.time_to_first_liquidation_s, clean.time_to_ruin_s) == (
        0,
        None,
        None,
    )

    # Cross bust 1 ns after the start: integer ns arithmetic survives (a float subtraction
    # of two 1.7e18 values would give 0 or 256 ns).
    bust = row_risk_facts(
        RowOutcome(
            k=2.0,
            status="ok",
            liquidation=_liq(
                count=2,
                bust_ts=_START + 1,
                events=[{"ts": _START + 7_000_000_001}, {"ts": _START + 3_000_000_001}],
            ),
        ),
        _START,
    )
    assert bust.liquidation_count == 2
    assert bust.time_to_ruin_s == 1e-9
    assert bust.time_to_first_liquidation_s == 3.000000001

    # Another start and bust: 2 days and 1 ns.
    later = row_risk_facts(
        RowOutcome(
            k=3.0, status="ok", liquidation=_liq(count=1, bust_ts=_START + 172_800_000_000_001)
        ),
        _START,
    )
    assert later.time_to_ruin_s == 172_800.000000001
    assert later.time_to_first_liquidation_s is None  # no events recorded

    # An isolated hit counts and dates its first liquidation but is not account ruin.
    isolated = row_risk_facts(
        RowOutcome(
            k=5.0,
            status="ok",
            liquidation=_liq(mode="isolated", count=1, events=[{"ts": _START + 60_000_000_000}]),
        ),
        _START,
    )
    assert (isolated.liquidation_count, isolated.time_to_first_liquidation_s) == (1, 60.0)
    assert isolated.time_to_ruin_s is None
