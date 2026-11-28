"""Liquidating move of a margin setting: one leg, joint cross books, the margin-limit book.

Contract rows R1-R13c of design arch-bankroll-1 (round 2). Entry price 1.0; tables = the
shipped snapshot; FIX = fixture brackets [0,1000) mmr 0.01 cum 0, [1000,2000) mmr 0.02 cum 10.
Expected numbers are exact floats copied from the design's reference run unless noted.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import pytest

from vibe_quant.bankroll.liq_move import (
    Book,
    Move,
    cell_move,
    cross_move,
    isolated_move,
    margin_book,
)
from vibe_quant.liquidation.tiers import (
    Bracket,
    TierTable,
    TierTableError,
    TierTables,
    cross_liq_price,
    load_tier_tables,
)
from vibe_quant.margin_limit import MarginItem, check_entry_margin

T = load_tier_tables()
BTC, ETH, SOL = T["BTCUSDT"], T["ETHUSDT"], T["SOLUSDT"]
FIX = TierTable(
    "FIX",
    (Bracket(0.0, 1000.0, 0.01, 0.0, None), Bracket(1000.0, 2000.0, 0.02, 10.0, None)),
    None,
)
FIX2 = TierTable("FIX2", FIX.brackets, None)


class _Boom(TierTable):
    def maintenance_margin(self, notional: float) -> float:
        raise TierTableError("BOOM: injected")


def _tables(*extra: TierTable) -> TierTables:
    out = TierTables(version=T.version, fetched_at=T.fetched_at, source=T.source, note=T.note)
    out.update(T)
    out.update({t.symbol: t for t in extra})
    return out


FT = _tables(FIX, FIX2, _Boom("BOOM", FIX.brackets, None))


def _cell(**kw: object) -> tuple[Book | None, Move]:
    args: dict[str, object] = {"starting_balance": 1000.0, "tables": FT, **kw}
    return cell_move(**args)  # type: ignore[arg-type]


def _assert_fixed_point(wallet: float, legs: list[tuple[int, float, TierTable]], x: float) -> None:
    """Each leg's one-position cross LP, given every other leg moved adversely by x, is x."""
    for i, (side, notional, table) in enumerate(legs):
        others = [leg for j, leg in enumerate(legs) if j != i]
        other_mm = math.fsum(
            t.maintenance_margin(n * (1 - x) if s == 1 else n * (1 + x)) for s, n, t in others
        )
        other_upnl = -math.fsum(n * x for _, n, _ in others)
        lp = cross_liq_price(side, notional, 1.0, wallet, other_mm, other_upnl, table)
        assert (1.0 - lp if side == 1 else lp - 1.0) == pytest.approx(x, abs=1e-12)


def test_isolated_long() -> None:
    assert isolated_move(1, 500.0, 10.0, BTC) == ("ok", 0.09638554216867468, None)
    assert isolated_move(1, 500.0, 10.0, SOL) == ("ok", 0.09547738693467334, None)


def test_isolated_short() -> None:
    assert isolated_move(-1, 500.0, 10.0, BTC) == ("ok", 0.095617529880478, None)
    assert isolated_move(-1, 500.0, 10.0, SOL) == ("ok", 0.0945273631840795, None)


def test_isolated_l1() -> None:
    assert isolated_move(1, 500.0, 1.0, BTC) == ("never", None, None)
    assert isolated_move(-1, 500.0, 1.0, BTC) == ("ok", 0.9920318725099602, None)


@pytest.mark.parametrize("leverage", [125.0, 100.0])
@pytest.mark.parametrize("side", [1, -1])
def test_at_entry(side: int, leverage: float) -> None:
    # L100: the isolated margin (5.0) equals the maintenance margin exactly.
    assert isolated_move(side, 500.0, leverage, FIX) == ("at_entry", 0.0, None)


def test_cross_single() -> None:
    long = cross_move(1000.0, [(1, 2000.0, BTC)])
    short = cross_move(1000.0, [(-1, 2000.0, BTC)])
    assert long == ("ok", 0.4979919678714859, None)
    assert short == ("ok", 0.4940239043824701, None)
    assert long[1] == 1.0 - cross_liq_price(1, 2000.0, 1.0, 1000.0, 0.0, 0.0, BTC)
    assert short[1] == cross_liq_price(-1, 2000.0, 1.0, 1000.0, 0.0, 0.0, BTC) - 1.0


def test_cross_wallet_edge() -> None:
    assert cross_move(1000.0, [(1, 1000.0, BTC)]) == ("never", None, None)
    assert cross_move(1000.0, [(1, 1000.0001, BTC)]) == ("ok", 0.9999998995984036, None)
    assert cross_move(1000.0, [(-1, 1000.0, BTC)]) == ("ok", 0.9920318725099602, None)


def test_cross_long_book_at_wallet() -> None:
    assert cross_move(1000.0, [(1, 500.0, BTC), (1, 500.0, SOL)]) == ("never", None, None)
    three = [(1, 300.0, BTC), (1, 300.0, ETH), (1, 300.0, SOL)]
    assert cross_move(900.0, three) == ("never", None, None)
    over = cross_move(1000.0, [(1, 500.0001, BTC), (1, 500.0, SOL)])
    assert over == ("ok", 0.999999899547976, None)


@pytest.mark.parametrize(
    ("sides", "expected"),
    [
        ((-1, -1), 0.49328023892483824),
        ((1, -1), 0.49525237381309345),
        ((1, 1), 0.4977398292315419),
    ],
)
def test_cross_joint(sides: tuple[int, int], expected: float) -> None:
    legs = [(sides[0], 1000.0, BTC), (sides[1], 1000.0, SOL)]
    assert cross_move(1000.0, legs) == ("ok", expected, None)
    _assert_fixed_point(1000.0, legs, expected)


def test_cross_tier_crossing() -> None:
    legs = [(-1, 250000.0, BTC), (-1, 250000.0, ETH)]
    status, value, reason = cross_move(200000.0, legs)
    assert (status, value, reason) == ("ok", 0.39422885572139305, None)
    assert value is not None
    one = cross_liq_price(-1, 250000.0, 1.0, 100000.0, 0.0, 0.0, BTC) - 1.0
    assert one == 0.3942288557213931
    assert abs(value - one) <= 1e-12
    _assert_fixed_point(200000.0, legs, value)


def test_unavailable() -> None:
    assert cross_move(5000.0, [(-1, 1500.0, FIX)]) == ("unavailable", None, "tier_overflow")
    book, move = _cell(
        mode="isolated", leverage=10.0, max_margin_pct=0.5, cap=0.5,
        symbols=["XYZUSDT"], sides=[1],
    )  # fmt: skip
    assert book is None
    assert (move.status, move.value, move.reason, move.legs) == (
        "unavailable",
        None,
        "no_tier_table",
        (),
    )


def test_entry_at_last_cap() -> None:
    overflow = ("unavailable", None, "tier_overflow")
    assert isolated_move(1, 2000.0, 10.0, FIX) == overflow
    assert isolated_move(-1, 2000.0, 10.0, FIX) == overflow
    assert cross_move(1e6, [(1, 2000.0, FIX)]) == overflow
    assert cross_move(1e6, [(-1, 2000.0, FIX), (-1, 2000.0, FIX2)]) == overflow
    assert isolated_move(1, 1999.99, 10.0, FIX) == ("ok", 0.08673471938788269, None)


@pytest.mark.parametrize(
    ("n", "cap", "leverage", "pct", "expected"),
    [
        (4, 0.5, 3.0, 0.5, Book(3, 0.0)),  # 3 x 500/3 == 500: equality admitted
        (4, 0.5, 3.0, 1.0, Book(4, 0.0)),
        (4, 2.0, 3.0, 0.5, Book(0, 1500.0)),
        (3, 1.0, 3.0, 0.7, Book(2, 100.0)),
        (2, 1.0, 3.0, 0.5, Book(1, 500.0)),
        (4, 0.3, 3.0, 0.1, Book(1, 0.0)),
        (2, 5.0, 5.0, 0.5, Book(0, 2500.0)),
    ],
)
def test_margin_book(n: int, cap: float, leverage: float, pct: float, expected: Book) -> None:
    book = margin_book(
        n_symbols=n, notional=cap * 1000.0, leverage=leverage, max_margin_pct=pct, equity=1000.0
    )
    assert book == expected
    lev = Decimal(str(leverage))
    full = [MarginItem(Decimal(str(cap * 1000.0)), lev)] * book.positions
    if book.remainder > 0:
        r = Fraction(book.remainder)
        decision = check_entry_margin(
            open_items=full,
            pending_items=[],
            new_item=MarginItem(Decimal(r.numerator) / Decimal(r.denominator), lev),
            equity=Decimal("1000.0"),
            max_margin_pct=Decimal(str(pct)),
        )
        assert decision.allowed, decision.reason


def test_cell_isolated() -> None:
    iso = {"mode": "isolated", "leverage": 10.0, "max_margin_pct": 0.5, "cap": 0.5}
    book, move = _cell(**iso, symbols=["SOLUSDT", "BTCUSDT"], sides=[1])
    assert book == Book(2, 0.0)
    assert (move.status, move.value, move.side, move.legs) == (
        "ok",
        0.09547738693467334,
        "long",
        (("SOLUSDT", 500.0),),
    )
    _, both = _cell(**iso, symbols=["SOLUSDT", "BTCUSDT"], sides=[1, -1])
    assert (both.value, both.side, both.legs) == (
        0.0945273631840795,
        "short",
        (("SOLUSDT", 500.0),),
    )
    _, tie = _cell(**iso, symbols=["FIX2", "FIX"], sides=[1])
    assert (tie.value, tie.side, tie.legs) == (0.09090909090909094, "long", (("FIX", 500.0),))
    book0, m0 = _cell(
        mode="isolated", leverage=3.0, max_margin_pct=0.5, cap=2.0,
        symbols=["BTCUSDT"], sides=[1],
    )  # fmt: skip
    assert book0 == Book(0, 1500.0)
    assert (m0.status, m0.value, m0.legs) == ("ok", 0.33065595716198126, (("BTCUSDT", 1500.0),))


def test_cell_cross_remainder() -> None:
    book, move = _cell(
        mode="cross", leverage=3.0, max_margin_pct=0.7, cap=1.0,
        symbols=["SOLUSDT", "ETHUSDT", "BTCUSDT"], sides=[-1],
    )  # fmt: skip
    assert book == Book(2, 100.0)
    assert (move.status, move.value, move.side, move.reason) == (
        "ok",
        0.4696122120034133,
        "short",
        None,
    )
    assert move.legs == (("BTCUSDT", 1000.0), ("SOLUSDT", 1000.0), ("ETHUSDT", 100.0))
    assert move.value is not None and move.value < 0.49328023892483824


def test_remainder_symbol_choice() -> None:
    cross = {"mode": "cross", "leverage": 3.0, "max_margin_pct": 0.5, "cap": 1.0}
    book, long = _cell(**cross, symbols=["BTCUSDT", "SOLUSDT"], sides=[1])
    assert book == Book(1, 500.0)
    assert (long.value, long.side, long.legs) == (
        0.6651038178164769,
        "long",
        (("SOLUSDT", 1000.0), ("BTCUSDT", 500.0)),
    )
    # The BTC-first book is the less adverse one, so it must lose.
    assert cross_move(1000.0, [(1, 1000.0, BTC), (1, 500.0, SOL)])[1] == 0.6652159357214597
    _, both = _cell(**cross, symbols=["BTCUSDT", "SOLUSDT"], sides=[1, -1])
    assert (both.value, both.side, both.legs) == (
        0.6589250165892502,
        "short",
        (("SOLUSDT", 1000.0), ("BTCUSDT", 500.0)),
    )


def test_cell_cross_m0() -> None:
    book, move = _cell(
        mode="cross", leverage=5.0, max_margin_pct=0.5, cap=5.0,
        symbols=["BTCUSDT", "SOLUSDT"], sides=[-1],
    )  # fmt: skip
    assert book == Book(0, 2500.0)
    assert (move.value, move.side, move.legs) == (
        0.39303482587064686,
        "short",
        (("SOLUSDT", 2500.0),),
    )
    assert cross_move(1000.0, [(-1, 2500.0, BTC)])[1] == 0.3944223107569722


@pytest.mark.parametrize("balance", [None, 0.0, -1.0, math.nan])
def test_no_balance(balance: float | None) -> None:
    book, move = _cell(
        mode="cross", leverage=3.0, max_margin_pct=0.5, cap=0.5,
        symbols=["BTCUSDT"], sides=[1], starting_balance=balance,
    )  # fmt: skip
    assert book is None
    assert (move.status, move.value, move.side, move.legs, move.reason) == (
        "unavailable",
        None,
        None,
        (),
        "no_starting_balance",
    )


def test_move_errors_stay_in_the_cell() -> None:
    book, boom = _cell(
        mode="cross", leverage=3.0, max_margin_pct=1.0, cap=1.0,
        symbols=["BOOM", "FIX"], sides=[-1],
    )  # fmt: skip
    assert book is None
    assert (boom.status, boom.reason) == (
        "unavailable",
        "move_error: TierTableError: BOOM: injected",
    )
    _, zero = _cell(
        mode="cross", leverage=3.0, max_margin_pct=0.0, cap=0.5,
        symbols=["BTCUSDT"], sides=[1],
    )  # fmt: skip
    assert (zero.status, zero.reason) == (
        "unavailable",
        "move_error: MarginLimitError: max_margin_pct must be a number in (0, 1], got "
        "Decimal('0.0')",
    )
    # No symbols or no side is no book at all: unavailable, never a "never" move.
    for symbols, sides in (([], [1]), (["BTCUSDT"], [])):
        _, empty = _cell(
            mode="cross", leverage=3.0, max_margin_pct=0.5, cap=0.5,
            symbols=symbols, sides=sides,
        )  # fmt: skip
        assert empty.status == "unavailable"
        assert empty.reason is not None and empty.reason.startswith("move_error: ValueError: ")


def test_path_dependent_flag() -> None:
    base = {"leverage": 3.0, "max_margin_pct": 0.5, "cap": 0.5, "sides": [1]}
    assert _cell(mode="cross", symbols=["BTCUSDT", "SOLUSDT"], **base)[1].path_dependent is True
    assert _cell(mode="cross", symbols=["BTCUSDT"], **base)[1].path_dependent is False
    iso = _cell(mode="isolated", symbols=["BTCUSDT", "SOLUSDT"], **base)[1]
    assert iso.path_dependent is False


def test_cross_book_at_entry() -> None:
    # Two legs whose maintenance margins (5400) exceed the wallet: liquidated on entry.
    legs = [(-1, 600000.0, BTC), (-1, 600000.0, ETH)]
    assert cross_move(5000.0, legs) == ("at_entry", 0.0, None)


def test_cross_mixed_book_past_full_move() -> None:
    # The long leg is worthless at x = 1 while the book still stands: refused, not a raw error.
    with pytest.raises(ValueError, match=r"^a mixed long/short cross book survives"):
        cross_move(3000.0, [(1, 1000.0, BTC), (-1, 1000.0, SOL)])


def test_isolated_leg_is_the_cap_with_a_remainder() -> None:
    # Book(1, 500): the cap-sized 1000 leg sits in FIX's 2nd bracket and binds, not the 500 one.
    book, move = _cell(
        mode="isolated", leverage=3.0, max_margin_pct=0.5, cap=1.0,
        symbols=["FIX2", "FIX"], sides=[-1],
    )  # fmt: skip
    assert book == Book(1, 500.0)
    assert (move.status, move.value, move.side, move.legs) == (
        "ok",
        0.31699346405228757,
        "short",
        (("FIX", 1000.0),),
    )


def test_duplicate_symbols_are_one_position() -> None:
    book, move = _cell(
        mode="cross", leverage=3.0, max_margin_pct=1.0, cap=1.0,
        symbols=["BTCUSDT", "BTCUSDT"], sides=[-1],
    )  # fmt: skip
    assert book == Book(1, 0.0)
    assert (move.status, move.value, move.legs, move.path_dependent) == (
        "ok",
        0.9920318725099602,
        (("BTCUSDT", 1000.0),),
        False,
    )
