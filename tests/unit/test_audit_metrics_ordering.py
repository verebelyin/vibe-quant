"""Same-bar SL/TP ambiguity must not favour longs over shorts (bd vibe-quant-e70tl.12).

NT's default bar processing is always O->H->L->C, so when one bar touches both
the SL and the TP, longs book the TP and shorts book the SL. SPEC asks for
SL-first (pessimistic) but NT cannot express an order-dependent ordering (the
choice is per bar, inside the Cython matching engine), so both venues enable
NT's ``bar_adaptive_high_low_ordering``: the extreme nearer the open is
processed first. That is direction-symmetric: a mirrored series gives a short
exactly the outcome the original gives a long.
"""

from __future__ import annotations

import pytest
from nautilus_trader.backtest.models import FillModel

from tests.unit.test_audit_metrics_fills import _bar, _exit, _run
from vibe_quant.validation.venue import (
    create_backtest_venue_config,
    create_venue_config_for_screening,
    create_venue_config_for_validation,
)

P0 = 10_000.0


def _mirror(o: float, h: float, lo: float, c: float) -> tuple[float, float, float, float]:
    """Reflect a bar around P0: highs become lows."""
    return 2 * P0 - o, 2 * P0 - lo, 2 * P0 - h, 2 * P0 - c


def _model() -> FillModel:
    return FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42)


def _adaptive() -> bool:
    """Ordering flag exactly as our venue factory configures it."""
    flags = {
        create_backtest_venue_config(create_venue_config_for_screening()).bar_adaptive_high_low_ordering,
        create_backtest_venue_config(
            create_venue_config_for_validation()
        ).bar_adaptive_high_low_ordering,
    }
    assert len(flags) == 1, "screening and validation must resolve ambiguity the same way"
    return flags.pop()


def test_both_venues_enable_adaptive_ordering() -> None:
    assert _adaptive() is True


# Bar 1 touches both TP (+1% = 10100) and SL (-1% = 9900) of a long entered at 10000.
LOW_NEARER = (10_000.0, 10_150.0, 9_880.0, 10_000.0)  # |L-O|=120 < |H-O|=150
HIGH_NEARER = (10_000.0, 10_120.0, 9_850.0, 10_000.0)  # |H-O|=120 < |L-O|=150


def _series(bar1: tuple[float, float, float, float]) -> list[object]:
    return [
        _bar(P0, P0 + 5, P0 - 5, P0, 0),
        _bar(*bar1, 1),
        _bar(P0, P0 + 5, P0 - 5, P0, 2),
    ]


@pytest.mark.parametrize(
    ("bar1", "long_exit_type"),
    [(LOW_NEARER, "STOP_MARKET"), (HIGH_NEARER, "LIMIT")],
    ids=["low-nearer->SL", "high-nearer->TP"],
)
def test_long_vs_mirrored_short_symmetric(
    bar1: tuple[float, float, float, float], long_exit_type: str
) -> None:
    adaptive = _adaptive()
    long_fills = _run(_model(), _series(bar1), side="BUY", tp=1.0, sl=1.0, adaptive=adaptive)
    short_fills = _run(
        _model(), _series(_mirror(*bar1)), side="SELL", tp=1.0, sl=1.0, adaptive=adaptive
    )
    long_exit, short_exit = _exit(long_fills), _exit(short_fills)
    assert long_exit[0] == long_exit_type
    assert short_exit[0] == long_exit_type  # same outcome for the mirrored short
    long_pnl = long_exit[2] - long_fills[0][2]
    short_pnl = short_fills[0][2] - short_exit[2]
    assert long_pnl == pytest.approx(short_pnl)


def test_old_fixed_ordering_was_biased() -> None:
    """Documents the bug: O->H->L->C always books long TP / short SL."""
    long_fills = _run(_model(), _series(LOW_NEARER), side="BUY", tp=1.0, sl=1.0)
    short_fills = _run(
        _model(), _series(_mirror(*LOW_NEARER)), side="SELL", tp=1.0, sl=1.0
    )
    assert _exit(long_fills)[0] == "LIMIT"  # long: TP although low is nearer
    assert _exit(short_fills)[0] == "STOP_MARKET"  # mirrored short: SL


def test_unambiguous_bars_unchanged_by_ordering() -> None:
    """Bars that only touch one of SL/TP fill identically under either ordering."""
    only_tp = [_bar(P0, P0 + 5, P0 - 5, P0, 0), _bar(P0, 10_150.0, 9_950.0, 10_100.0, 1),
               _bar(P0, P0 + 5, P0 - 5, P0, 2)]
    only_sl = [_bar(P0, P0 + 5, P0 - 5, P0, 0), _bar(P0, 10_050.0, 9_850.0, 9_900.0, 1),
               _bar(P0, P0 + 5, P0 - 5, P0, 2)]
    for bars in (only_tp, only_sl):
        fixed = _run(_model(), bars, tp=1.0, sl=1.0, adaptive=False)
        adaptive = _run(_model(), bars, tp=1.0, sl=1.0, adaptive=True)
        assert fixed == adaptive
