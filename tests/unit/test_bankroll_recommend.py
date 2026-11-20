"""Row eligibility, Kelly actionability and the tested-row-only recommendation.

Contract rows C3 (recommendation side), C10, C11, C12, C13 and C14 of design 2gyk1.3.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any

import pytest

from vibe_quant.bankroll.kelly import KellyStats, kelly_stats
from vibe_quant.bankroll.recommend import (
    BINDING_TEXT,
    kelly_actionable,
    recommend,
    row_eligibility,
)
from vibe_quant.bankroll.stress import RowOutcome


def _liq(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mode": "isolated",
        "count": 0,
        "bust_ts": None,
        "path_divergence": False,
        "events": [],
    }
    base.update(over)
    return base


def _row(k: float = 1.0, **over: object) -> RowOutcome:
    """An eligible row: ran ok, profitable, MTM DD 0.2, liquidation modeled and clean."""
    fields: dict[str, object] = {
        "k": k,
        "status": "ok",
        "trades": 12,
        "total_return": 0.4,
        "mtm_paths": "ok",
        "mtm_dd": 0.2,
        "liquidation": _liq(),
    }
    fields.update(over)
    return RowOutcome(**fields)  # type: ignore[arg-type]


def _stats(f_half: float) -> KellyStats:
    return KellyStats(
        status="ok",
        n=400,
        target_vol=0.2,
        mean=0.001,
        variance=0.0004,
        sigma_ann=math.sqrt(0.146),
        f_full=2 * f_half,
        f_half=f_half,
        vol_target=0.2 / math.sqrt(0.146),
        excess_kurtosis=0.0,
        fat_tail=False,
    )


def _two_point(mean: float, var: float, n: int) -> list[float]:
    d = math.sqrt(var * (n - 1) / n)
    return [mean + d, mean - d] * (n // 2)


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"status": "failed", "error": "RuntimeError: boom"}, "failed"),
        ({"status": "not_runnable"}, "not_runnable"),
        ({"engine_force_stopped": True}, "engine_force_stopped"),
        ({"trades": 0}, "no_trades"),
        ({"total_return": None}, "non_finite"),
        ({"total_return": math.nan}, "non_finite"),
        ({"total_return": math.inf}, "non_finite"),
        ({"total_return": 0.0}, "non_positive_return"),
        ({"total_return": -0.1}, "non_positive_return"),
        ({"mtm_paths": "none"}, "mtm_unavailable"),
        ({"mtm_paths": "unavailable"}, "mtm_unavailable"),
        ({"mtm_paths": None}, "mtm_unavailable"),
        ({"mtm_dd": None}, "non_finite"),
        ({"mtm_dd": math.nan}, "non_finite"),
        ({"mtm_dd": -0.01}, "dd_out_of_range"),
        ({"mtm_dd": math.nextafter(1.0, 2.0)}, "dd_out_of_range"),
        ({"mtm_dd": 0.31}, "dd_over_limit"),
        ({"mtm_dd": math.nextafter(0.3, 1.0)}, "dd_over_limit"),
        ({"liquidation": None}, "liquidation_not_modeled"),
        ({"liquidation": _liq(count=None)}, "liquidation_not_modeled"),
        ({"liquidation": _liq(count=True)}, "liquidation_not_modeled"),
        ({"liquidation": _liq(count=1)}, "liquidated"),
        ({"bust_ts": 1_700_000_000_000_000_000}, "bust"),
        ({"liquidation": _liq(bust_ts=1_700_000_000_000_000_000)}, "bust"),
        ({"path_divergence": True}, "path_divergence"),
        ({"liquidation": _liq(path_divergence=True)}, "path_divergence"),
    ],
)
def test_eligibility(over: dict[str, Any], reason: str) -> None:
    assert row_eligibility(_row(), dd_limit=0.3) is None
    assert row_eligibility(dataclasses.replace(_row(), **over), dd_limit=0.3) == reason


def test_eligibility_edges() -> None:
    # DD exactly at the limit is eligible; DD 0 is in range; DD 1 is in range but over 0.99.
    assert row_eligibility(_row(mtm_dd=0.3), dd_limit=0.3) is None
    assert row_eligibility(_row(mtm_dd=0.0), dd_limit=0.3) is None
    assert row_eligibility(_row(mtm_dd=1.0), dd_limit=0.99) == "dd_over_limit"
    assert row_eligibility(_row(trades=1), dd_limit=0.3) is None
    assert row_eligibility(_row(total_return=5e-324), dd_limit=0.3) is None
    # A force-stopped cell that also failed reads as force-stopped.
    stopped = _row(status="failed", engine_force_stopped=True)
    assert row_eligibility(stopped, dd_limit=0.3) == "engine_force_stopped"


def test_kelly_actionable() -> None:
    stats = _stats(1.0)
    assert kelly_actionable(_row(), stats) is None
    # Mode off is allowed for the baseline (no liquidation notes).
    assert kelly_actionable(_row(liquidation=None), stats) is None

    cases: list[tuple[RowOutcome, KellyStats | None, str]] = [
        (_row(status="failed"), stats, "failed"),
        (_row(engine_force_stopped=True), stats, "engine_force_stopped"),
        (_row(trades=0), stats, "no_trades"),
        (_row(mtm_paths="unavailable"), stats, "mtm_unavailable"),
        (_row(mtm_paths="none"), stats, "mtm_unavailable"),
        (_row(), None, "daily_unavailable"),
        (_row(mtm_dd=None), stats, "non_finite"),
        (_row(mtm_dd=math.inf), stats, "non_finite"),
        (_row(mtm_dd=-0.1), stats, "dd_out_of_range"),
        # Intraday wipeout: DD 1.0 even though daily closes recovered (stats are ok).
        (_row(mtm_dd=1.0), stats, "intraday_wipeout"),
        (_row(mtm_dd=1.0, liquidation=None), stats, "intraday_wipeout"),
        # bust_ts is checked on its own, whatever the float DD says.
        (_row(mtm_dd=0.1, bust_ts=1_700_000_000_000_000_000), stats, "bust"),
        (_row(mtm_dd=0.1, liquidation=None, bust_ts=1), stats, "bust"),
        (_row(liquidation=_liq(bust_ts=5)), stats, "bust"),
        (_row(liquidation=_liq(count=3)), stats, "liquidated"),
        (_row(liquidation=_liq(count=None)), stats, "liquidation_not_modeled"),
        (_row(liquidation=_liq(path_divergence=True)), stats, "path_divergence"),
        (_row(path_divergence=True), stats, "path_divergence"),
        (_row(), kelly_stats([0.01] * 59), "insufficient_data"),
        (_row(), kelly_stats([0.01] * 80), "zero_variance"),
        (_row(), kelly_stats([1e200, 0.0] * 40), "numeric"),
    ]
    for row, case_stats, reason in cases:
        assert kelly_actionable(row, case_stats) == reason, (row, reason)
    # Equality edge: DD just under 1 is still actionable (no DD limit on the baseline).
    assert kelly_actionable(_row(mtm_dd=math.nextafter(1.0, 0.0)), stats) is None
    # Mode off: divergence is only defined with liquidation modeled, so it is not checked.
    assert kelly_actionable(_row(liquidation=None, path_divergence=True), stats) is None

    # Not actionable -> descriptive stats kept, recommendation null with that reason.
    rec = recommend(
        [_row(0.5), _row(1.0)], stats, baseline=_row(mtm_dd=1.0), user_cap=1.0, dd_limit=0.3
    )
    assert (rec.k, rec.reason, rec.ceiling) == (None, "intraday_wipeout", None)
    assert [r for _, r in rec.row_reasons] == [None, None]


def test_recommend_tested_only() -> None:
    rows = [_row(0.25), _row(0.5, trades=0), _row(1.0)]
    rec = recommend(rows, _stats(0.4567), baseline=_row(), user_cap=1.0, dd_limit=0.3)
    assert rec.k == 0.25
    assert rec.ceiling == 0.4567
    assert rec.reason is None
    assert rec.binding == "half_kelly"
    assert rec.row_reasons == ((0.25, None), (0.5, "no_trades"), (1.0, None))

    # Ceiling exactly on a row picks that row (k <= C edge); the user cap binds when lower.
    on_row = recommend(
        [_row(0.25), _row(1.0)], _stats(1.0), baseline=_row(), user_cap=5.0, dd_limit=0.3
    )
    assert (on_row.k, on_row.ceiling, on_row.binding) == (1.0, 1.0, "half_kelly")
    capped = recommend(
        [_row(0.25), _row(0.75), _row(2.0)],
        _stats(3.0),
        baseline=_row(),
        user_cap=0.75,
        dd_limit=0.3,
    )
    assert (capped.k, capped.ceiling, capped.binding) == (0.75, 0.75, "user_cap")
    assert capped.text == BINDING_TEXT["user_cap"]
    # f_half == user_cap: the half-Kelly ceiling is the named binding.
    tie = recommend([_row(0.5)], _stats(0.8), baseline=_row(), user_cap=0.8, dd_limit=0.3)
    assert (tie.k, tie.binding) == (0.5, "half_kelly")
    # An ineligible tested row between the pick and the ceiling binds as row_ineligible.
    blocked = recommend(
        [_row(0.5), _row(1.0, liquidation=_liq(count=2))],
        _stats(3.0),
        baseline=_row(),
        user_cap=5.0,
        dd_limit=0.3,
    )
    assert (blocked.k, blocked.binding) == (0.5, "row_ineligible")

    # Properties over a grid of eligibility patterns, ceilings and caps: the pick is a
    # tested eligible row, at most the ceiling and the user cap, and never rounded.
    grid = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 10.0]
    for mask, f_half, cap in itertools.product(
        itertools.product([True, False], repeat=4), [0.3, 0.8, 1.7, 4.0, 12.0], [0.5, 1.0, 7.5]
    ):
        rows = [
            _row(k) if (k not in grid[:4] or mask[grid.index(k)]) else _row(k, mtm_dd=0.9)
            for k in grid
        ]
        got = recommend(rows, _stats(f_half), baseline=_row(), user_cap=cap, dd_limit=0.3)
        eligible = [r.k for r in rows if r.mtm_dd == 0.2]
        under = [k for k in eligible if k <= min(f_half, cap)]
        assert got.ceiling == min(f_half, cap)
        if under:
            assert got.k == max(under)
            assert got.k in grid and got.k <= cap
        else:
            assert (got.k, got.reason) == (None, "no_eligible_row")


def test_dd_binds() -> None:
    rows = [_row(0.5, mtm_dd=0.22), _row(1.0, mtm_dd=0.41)]
    rec = recommend(rows, _stats(3.0), baseline=_row(), user_cap=5.0, dd_limit=0.30)
    assert rec.k == 0.5
    assert rec.ceiling == 3.0
    assert rec.binding == "dd_limit"
    assert rec.text == "Kelly overstates: drawdown limit binds"
    assert rec.row_reasons == ((0.5, None), (1.0, "dd_over_limit"))
    # A DD-failing row exactly AT the ceiling binds (k <= C edge) ...
    at_ceiling = recommend(rows, _stats(1.0), baseline=_row(), user_cap=5.0, dd_limit=0.30)
    assert (at_ceiling.k, at_ceiling.binding) == (0.5, "dd_limit")
    # ... and a DD-failing row ABOVE the ceiling does not.
    above = recommend(rows, _stats(0.6), baseline=_row(), user_cap=5.0, dd_limit=0.30)
    assert (above.k, above.binding) == (0.5, "half_kelly")


def test_none() -> None:
    rows = [_row(0.5, mtm_dd=0.5), _row(1.0, trades=0)]
    rec = recommend(rows, _stats(3.0), baseline=_row(), user_cap=5.0, dd_limit=0.3)
    assert (rec.k, rec.reason, rec.binding, rec.text) == (None, "no_eligible_row", None, None)
    assert rec.ceiling == 3.0
    assert rec.row_reasons == ((0.5, "dd_over_limit"), (1.0, "no_trades"))

    # Eligible rows exist but all lie above the ceiling.
    high = recommend([_row(2.0)], _stats(1.0), baseline=_row(), user_cap=5.0, dd_limit=0.3)
    assert (high.k, high.reason) == (None, "no_eligible_row")


def test_non_positive_mean_recommendation() -> None:
    """C3: mean 0 and mean -0.0005 both give ceiling 0 and a null recommendation."""
    rows = [_row(0.25), _row(1.0)]
    for mean, f_full in ((0.0, 0.0), (-0.0005, -1.25)):
        stats = kelly_stats(_two_point(mean, 0.0004, 400))
        assert stats.f_full == pytest.approx(f_full, abs=1e-9)
        rec = recommend(rows, stats, baseline=_row(), user_cap=1.0, dd_limit=0.3)
        assert rec.ceiling == 0.0
        assert (rec.k, rec.reason, rec.binding) == (None, "non_positive_mean", None)
        assert rec.row_reasons == ((0.25, None), (1.0, None))


def test_recommendation_fields_carry_no_loss_bound() -> None:
    from vibe_quant.bankroll.recommend import Recommendation

    names = {f.name for f in dataclasses.fields(Recommendation)}
    assert not [n for n in names if "loss" in n]
