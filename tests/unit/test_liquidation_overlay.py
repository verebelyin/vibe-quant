"""Liquidation overlay: NT positions -> scanner input -> truncated ledgers (bead vibe-quant-2gyk1.1.5).

Positions are duck-typed stubs, marks synthetic 1m bars. Expectations are hand-computed
from the Binance one-way formula ``LP = (WB + cum - side*q*EP) / (q*MMR - side*q)``
(BTC tier 1: MMR .004, cum 0) and from the ledger identity in ``overlay.py``:
ledger realized PnL of a liquidated position (close + cash) == -hit.loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest

from tests.unit.fake_engine import fake_engine
from vibe_quant.liquidation import OverlayResult, liquidation_overlay, position_views
from vibe_quant.liquidation.overlay import RefinedExitConflict
from vibe_quant.liquidation.scan import MissingMarkSeries
from vibe_quant.liquidation.tiers import load_tier_tables
from vibe_quant.metrics import BarSeries, LedgerBook, LedgerEvent, mark_to_market_equity, pnl_path
from vibe_quant.validation.extraction import (
    mark_to_market_drawdown,
    mark_to_market_paths,
    position_ledger_events,
)

MIN = 60_000_000_000
BASE = 28_333_333 * MIN
BTC = "BTCUSDT-PERP.BINANCE"
ETH = "ETHUSDT-PERP.BINANCE"
TABLES = load_tier_tables()
LP_LONG = 36144.57831325301  # (4000 - 40000) / (0.004 - 1)


class Name(NamedTuple):
    name: str


BUY, SELL = Name("BUY"), Name("SELL")
MARKET, LIMIT = Name("MARKET"), Name("LIMIT")


def bar(i: int) -> int:
    return BASE + i * MIN


@dataclass
class Fill:
    ts_event: int
    order_side: Any
    last_qty: float
    last_px: float
    commission: float = 0.0
    order_type: Any = MARKET


@dataclass
class Pos:
    id: str
    instrument_id: str
    ts_opened: int
    ts_closed: int
    events: list[Fill] = field(default_factory=list)


def make_pos(pid: str, iid: str, fills: list[Fill], closed: bool = True) -> Pos:
    return Pos(pid, iid, fills[0].ts_event, fills[-1].ts_event if closed else 0, fills)


def key_of(p: Pos) -> str:
    return f"{p.id}@{p.ts_opened}"


class Marks(NamedTuple):
    ts: np.ndarray
    low: np.ndarray
    high: np.ndarray
    close: np.ndarray
    open: np.ndarray


def marks(
    lows: list[float],
    highs: list[float] | None = None,
    closes: list[float] | None = None,
    opens: list[float] | None = None,
) -> Marks:
    n = len(lows)
    low = np.asarray(lows, dtype=np.float64)
    high = np.full(n, 41000.0) if highs is None else np.asarray(highs, dtype=np.float64)
    close = low.copy() if closes is None else np.asarray(closes, dtype=np.float64)
    open_ = np.concatenate((close[:1], close[:-1])) if opens is None else np.asarray(opens, float)
    return Marks(BASE + np.arange(n, dtype=np.int64) * MIN, low, high, close, open_)


def exec_series(mk: Marks) -> BarSeries:
    """1m LAST execution bars equal to the mark bars, in production timestamps: a catalog
    bar's ``ts_init`` is its minute close less 1 ms, a mark bar's ``ts`` its OPEN, so exec bar
    ``i`` is stamped ``mark.ts[i] + 1 minute - 1 ms``. A flat warm-up bar closes just before
    ``mark.ts[0]`` (a run's first fill always follows a closed bar)."""
    first = mk.open[:1]
    return BarSeries(
        ts=np.concatenate((mk.ts[:1] - 1_000_000, mk.ts + MIN - 1_000_000)),
        open=np.concatenate((first, mk.open)),
        high=np.concatenate((first, mk.high)),
        low=np.concatenate((first, mk.low)),
        close=np.concatenate((first, mk.close)),
    )


def engine_for(positions: list[Any], series: dict[str, Marks]) -> Any:
    """NT-cache stub serving ``exec_series`` for each instrument."""
    bars_by_type: dict[str, list[SimpleNamespace]] = {}
    for instrument_id, mk in series.items():
        ex = exec_series(mk)
        bars_by_type[f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"] = [
            SimpleNamespace(
                ts_init=int(t), open=float(o), high=float(h), low=float(lo), close=float(c)
            )
            for t, o, h, lo, c in zip(ex.ts, ex.open, ex.high, ex.low, ex.close, strict=True)
        ]
    return fake_engine(positions, bars_by_type=bars_by_type)


def strip(event: LedgerEvent) -> LedgerEvent:
    """A real event without its valuation annotation (``liq_margin``)."""
    return replace(event, liq_margin=0.0)


def ledger_realized(events: list[LedgerEvent]) -> float:
    book = LedgerBook()
    for e in events:
        book.apply(e)
    return book.realized


def long_btc_position(*, sell_bar: int | None = None, commission: float = 0.0) -> Pos:
    fills = [Fill(bar(0), BUY, 1.0, 40000.0, commission)]
    if sell_bar is not None:
        fills.append(Fill(bar(sell_bar), SELL, 1.0, 38000.0))
    return make_pos("P-1", BTC, fills)


def isolated(positions: list[Pos], mk: dict[str, Marks], **kw: Any) -> OverlayResult:
    kw.setdefault("execution_bar_ns", MIN)
    return liquidation_overlay(
        positions,
        mk,
        mode="isolated",
        starting_balance=40000.0,
        leverage=10.0,
        tables=TABLES,
        **kw,
    )


# ---------------------------------------------------------------- position_views


def test_position_views_round_trip_scale_in_and_partial_reduction() -> None:
    """Fills: +1 @ 40000 (fee 2), +0.5 @ 40500 (fee 1), -0.75 @ 41000 (LIMIT, fee 1.5)."""
    p = make_pos(
        "P-7",
        BTC,
        [
            Fill(bar(0), BUY, 1.0, 40000.0, 2.0),
            Fill(bar(3), BUY, 0.5, 40500.0, 1.0),
            Fill(bar(9), SELL, 0.75, 41000.0, 1.5, order_type=LIMIT),
        ],
    )
    (view,) = position_views([p], {key_of(p): [(bar(5), -0.25), (bar(1), -0.5)]})
    assert view.key == f"P-7@{bar(0)}"
    assert (view.symbol, view.instrument_id) == ("BTCUSDT", BTC)
    assert list(view.events) == [
        LedgerEvent(ts=bar(0), qty=1.0, price=40000.0, cash=-2.0, resting=False),
        LedgerEvent(ts=bar(3), qty=0.5, price=40500.0, cash=-1.0, resting=False),
        LedgerEvent(ts=bar(9), qty=-0.75, price=41000.0, cash=-1.5, resting=True),
    ]
    assert list(view.funding) == [(bar(1), -0.5), (bar(5), -0.25)]  # sorted, ledger sign


def test_position_views_match_position_ledger_events_with_real_nt_enums() -> None:
    """Same fill fields as the ledger the metrics code builds today (incl. resting)."""
    from nautilus_trader.model.enums import OrderSide, OrderType

    p = make_pos(
        "P-2",
        BTC,
        [
            Fill(bar(0), OrderSide.BUY, 2.0, 40000.0, 4.0, order_type=OrderType.MARKET),
            Fill(bar(4), OrderSide.SELL, 2.0, 41000.0, 3.0, order_type=OrderType.STOP_MARKET),
        ],
    )
    (view,) = position_views([p])
    assert list(view.events) == position_ledger_events([p])[BTC]


def test_position_views_reject_duplicate_keys_unknown_funding_and_empty_positions() -> None:
    a = make_pos("P-1", BTC, [Fill(bar(0), BUY, 1.0, 40000.0), Fill(bar(1), SELL, 1.0, 40000.0)])
    with pytest.raises(ValueError, match="duplicate position key"):
        position_views([a, a])
    with pytest.raises(ValueError, match="unknown position keys"):
        position_views([a], {"P-9@1": [(bar(0), -1.0)]})
    with pytest.raises(ValueError, match="no fill events"):
        position_views([Pos("P-3", BTC, bar(0), 0, [])])
    bad_side = make_pos("P-4", BTC, [Fill(bar(0), Name("NO_ORDER_SIDE"), 1.0, 40000.0)])
    with pytest.raises(ValueError, match="unknown order side"):
        position_views([bad_side])


# ------------------------------------------------------------------- identity


def test_zero_hits_is_identity_and_ledgers_none_leaves_mtm_untouched() -> None:
    p = long_btc_position(sell_bar=4, commission=2.0)
    mk = marks([39000, 38500, 38000, 38200, 38000])
    ov = isolated([p], {"BTCUSDT": mk})
    assert ov.is_identity and ov.ledgers is None
    assert ov.hits_by_key == {} and ov.dropped_keys == frozenset() and ov.exit_override == {}
    assert ov.result.is_empty

    engine = engine_for([p], {BTC: mk})
    before = pnl_path(exec_series(mk), position_ledger_events([p])[BTC])
    (path,) = mark_to_market_paths(engine, execution_timeframe="1m", ledgers=None) or []
    assert np.array_equal(path[1], before[0]) and np.array_equal(path[2], before[1])
    explicit = mark_to_market_paths(
        engine, execution_timeframe="1m", ledgers=position_ledger_events([p])
    )
    assert explicit is not None
    assert np.array_equal(explicit[0][1], before[0]) and np.array_equal(explicit[0][2], before[1])
    dd = mark_to_market_drawdown(engine, 40000.0, execution_timeframe="1m")
    assert dd == mark_to_market_drawdown(
        engine, 40000.0, execution_timeframe="1m", ledgers=position_ledger_events([p])
    )


def test_ledgers_param_rejects_extra_cash_and_accepts_empty() -> None:
    """The overlay ledger already carries all per-position cash: extra_cash on top would
    double-book it (or charge after a bust), so the combination fails closed."""
    p = long_btc_position(sell_bar=4, commission=2.0)
    mk = marks([39000, 38500, 38000, 38200, 38000])
    engine = engine_for([p], {BTC: mk})
    ledgers = position_ledger_events([p])
    with pytest.raises(ValueError, match="extra_cash cannot be combined with ledgers"):
        mark_to_market_paths(
            engine, execution_timeframe="1m", ledgers=ledgers, extra_cash={BTC: [(bar(2), -0.5)]}
        )
    with pytest.raises(ValueError, match="extra_cash cannot be combined with ledgers"):
        mark_to_market_drawdown(
            engine,
            40000.0,
            execution_timeframe="1m",
            ledgers=ledgers,
            extra_cash={BTC: [(1, -1.0)]},
        )
    assert mark_to_market_paths(engine, execution_timeframe="1m", ledgers={}) == []
    same = mark_to_market_paths(engine, execution_timeframe="1m", ledgers=ledgers, extra_cash={})
    assert same is not None


# ------------------------------------------------------------------- isolated


def test_isolated_liquidation_closes_at_lp_and_books_the_account_loss() -> None:
    """10x long q=1 @ 40000: margin 4000, LP 36144.57831325301; the low 36100 crosses it.
    Price loss 3855.42, clearance (forfeit) 144.58 -> account loss 4000.0 exactly the margin."""
    p = long_btc_position(sell_bar=8, commission=2.0)
    mk = marks([39000, 38000, 36200, 36100, 36000, 36000, 36000, 36000, 36000])
    ov = isolated([p], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    assert not ov.is_identity and ov.dropped_keys == frozenset()
    assert hit.ts == bar(3) and hit.price == LP_LONG
    assert hit.loss == pytest.approx(4000.0, abs=1e-9)
    assert ov.hits_by_key == {key_of(p): hit}
    assert ov.exit_override == {key_of(p): (bar(3), LP_LONG)}

    assert ov.ledgers is not None
    events = ov.ledgers[BTC]
    assert [(e.ts, e.qty, e.price) for e in events if e.qty] == [
        (bar(0), 1.0, 40000.0),
        (bar(3), -1.0, LP_LONG),
    ]  # the strategy's SELL at bar 8 is gone
    assert not [e for e in events if e.qty == 0.0 and e.ts == bar(3)]  # fee rides the close
    assert events[-1].cash == -hit.clearance_fee and events[-1].resting is False
    assert hit.clearance_fee == pytest.approx(144.57831325301, abs=1e-9)
    # entry commission -2 + (close at LP - 40000) - clearance == -2 - 4000
    assert ledger_realized(events) == pytest.approx(-2.0 - 4000.0, abs=1e-9)
    assert ledger_realized(events) == pytest.approx(-2.0 - hit.loss, abs=1e-9)


def test_isolated_funding_after_the_hit_is_removed_and_before_it_is_kept() -> None:
    """Funding -0.5 at bar 2 shrinks the margin to 3999.5 (LP 36144.0763, still crossed at
    bar 3); funding -0.5 at bar 5 is after the hit and must vanish."""
    p = long_btc_position(sell_bar=8)
    mk = marks([39000, 38000, 37000, 36000, 36000, 36000, 36000, 36000, 36000])
    funding = {key_of(p): [(bar(2), -0.5), (bar(5), -0.5)]}
    ov = isolated([p], {"BTCUSDT": mk}, funding_by_position=funding)
    (hit,) = ov.result.hits
    assert hit.ts == bar(3) and hit.loss == pytest.approx(3999.5, abs=1e-9)
    assert ov.ledgers is not None
    events = ov.ledgers[BTC]
    assert [(e.ts, e.qty, e.cash) for e in events] == [
        (bar(0), 1.0, 0.0),
        (bar(2), 0.0, -0.5),
        (bar(3), -1.0, -hit.clearance_fee),
    ]
    assert all(e.ts <= bar(3) for e in events)
    assert ledger_realized(events) == pytest.approx(-0.5 - 3999.5, abs=1e-9)


def test_isolated_gap_past_bankruptcy_charges_the_margin_not_the_gap() -> None:
    """The bar opens at 35000 (gap): price loss 5000 but the isolated loss is the 4000
    margin; the insurance fund bears 1000 and the ledger must book -4000, not -5000. The
    closing event is priced at the bankruptcy-equivalent 36000 (raw loss 4000), so MTM
    equity never dips to 35000 between a close at the gap price and a separate credit."""
    p = long_btc_position(sell_bar=4)
    mk = marks([39000, 35000, 35000, 35000, 35000], opens=[40000, 35000, 35000, 35000, 35000])
    ov = isolated([p], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    assert hit.price == 35000.0 and hit.insurance_deficit == pytest.approx(1000.0, abs=1e-9)
    assert ov.ledgers is not None
    close = ov.ledgers[BTC][-1]
    assert close.price == pytest.approx(36000.0, abs=1e-9) and close.cash == 0.0
    assert ov.exit_override == {key_of(p): (bar(1), 35000.0)}  # the venue fill stays the gap
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(-4000.0, abs=1e-9)
    engine = engine_for([p], {BTC: mk})
    dd = mark_to_market_drawdown(engine, 40000.0, execution_timeframe="1m", ledgers=ov.ledgers)
    assert dd == pytest.approx(0.1, abs=1e-12)  # 40000 -> 36000, not 35000 (0.125)
    equity_close, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_min.min() == pytest.approx(36000.0, abs=1e-9)
    assert equity_close[-1] == pytest.approx(36000.0, abs=1e-9)


def test_isolated_other_instrument_ledger_is_untouched() -> None:
    btc = long_btc_position(sell_bar=8)
    from nautilus_trader.model.enums import OrderSide, OrderType

    eth = make_pos(
        "P-2",
        ETH,
        [
            Fill(bar(0), OrderSide.BUY, 10.0, 2000.0, 1.0, OrderType.MARKET),
            Fill(bar(6), OrderSide.SELL, 10.0, 2100.0, 1.0, OrderType.MARKET),
        ],
    )
    series = {
        "BTCUSDT": marks([39000, 38000, 36200, 36100, 36000, 36000, 36000, 36000, 36000]),
        "ETHUSDT": marks([2000.0] * 9, highs=[2100.0] * 9),
    }
    ov = isolated([btc, eth], series)
    assert set(ov.hits_by_key) == {key_of(btc)}
    assert ov.ledgers is not None
    assert ov.ledgers[ETH] == position_ledger_events([eth])[ETH]
    assert key_of(eth) not in ov.exit_override


# ---------------------------------------------------------------------- cross


def cross(
    positions: list[Pos],
    series: dict[str, Marks],
    starting_balance: float = 1000.0,
    **kw: Any,
) -> OverlayResult:
    kw.setdefault("execution_bar_ns", MIN)
    return liquidation_overlay(
        positions,
        series,
        mode="cross",
        starting_balance=starting_balance,
        leverage=10.0,
        tables=TABLES,
        **kw,
    )


def equity_path(
    ov: OverlayResult,
    positions: list[Pos],
    series: dict[str, Marks],
    starting_balance: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    assert ov.ledgers is not None
    paths = mark_to_market_paths(
        engine_for(positions, series), execution_timeframe="1m", ledgers=ov.ledgers
    )
    assert paths is not None
    (_ts, pnl_close, pnl_min) = paths[0]
    return starting_balance + pnl_close, starting_balance + pnl_min


def test_cross_bust_drops_later_positions_and_floors_equity_at_zero() -> None:
    """Cross, wallet 1000 - 2 fee = 998, long 0.1 BTC @ 40000 (4x): bust when
    998 + .1(P-40000) <= .004*.1*P -> LP = (998 - 4000)/(.0004 - .1) = 30140.56...
    The bar-4 low 30000 crosses it. A later SELL (bar 6) and a whole later position
    (bar 8) are NT's path past the bust and must vanish; equity is 0 from the bust on."""
    a = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 0.1, 40000.0, 2.0), Fill(bar(6), SELL, 0.1, 29000.0, 1.0)],
    )
    b = make_pos(
        "P-2", BTC, [Fill(bar(8), BUY, 0.1, 30000.0, 1.0), Fill(bar(9), SELL, 0.1, 30500.0, 1.0)]
    )
    mk = marks([39500, 39000, 38000, 31000, 30000, 30000, 30000, 30000, 30000, 30000])
    ov = cross([a, b], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    lp = (998.0 - 4000.0) / (0.004 * 0.1 - 0.1)
    assert hit.ts == bar(4) and hit.price == pytest.approx(lp, rel=1e-12)
    assert ov.dropped_keys == frozenset({key_of(b)})
    assert ov.result.bust_ts == bar(4) and ov.exit_override == {key_of(a): (bar(4), hit.price)}
    assert ov.ledgers is not None
    assert all(e.ts <= bar(4) for e in ov.ledgers[BTC])  # nothing after the bust survives
    assert sum(1 for e in ov.ledgers[BTC] if e.qty) == 2  # entry + synthetic close only

    equity_close, equity_min = equity_path(ov, [a, b], {BTC: mk})
    # exec bar 0 is the warm-up and mark bar i is exec bar i + 1 (ts_init = close - 1 ms), so
    # the event at the bar-4 open is replayed at the start of exec bar 5
    assert equity_close[4] == pytest.approx(98.0, abs=1e-9)  # alive until the bust
    assert equity_min[5] == pytest.approx(0.0, abs=1e-9)  # the bust bar itself
    assert equity_close[5:].min() >= 0.0 and equity_min[5:].min() >= 0.0  # never negative
    assert equity_close[-1] == pytest.approx(0.0, abs=1e-9)


def test_cross_insurance_deficit_is_not_charged_to_the_account() -> None:
    """The bust bar gaps to 25000: raw loss .1*(25000-40000) = -1500 against a 996 wallet.
    The 504 beyond the wallet is the insurance fund's: equity floors at 0, never -504."""
    a = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 0.1, 40000.0, 4.0), Fill(bar(5), SELL, 0.1, 25000.0, 1.0)],
    )
    mk = marks(
        [39500, 39000, 25000, 25000, 25000, 25000],
        opens=[40000, 39500, 25000, 25000, 25000, 25000],
    )
    ov = cross([a], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    assert hit.price == 25000.0 and hit.insurance_deficit == pytest.approx(504.0, abs=1e-9)
    assert hit.loss == pytest.approx(996.0, abs=1e-9)  # the wallet, nothing more
    assert ov.ledgers is not None
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(-1000.0, abs=1e-9)
    equity_close, equity_min = equity_path(ov, [a], {BTC: mk})
    assert equity_close[-1] == pytest.approx(0.0, abs=1e-9)
    assert equity_close[2] > 0.0  # the bust is the bar-2 open = exec bar 3
    assert equity_min[3] == pytest.approx(0.0, abs=1e-9)  # the bust bar itself
    assert equity_close[3:].min() >= 0.0 and equity_min[3:].min() >= 0.0  # never -504


# ---------------------------------------------------------------- fail closed


def test_missing_mark_series_propagates() -> None:
    p = long_btc_position(sell_bar=3)
    with pytest.raises(MissingMarkSeries, match="BTCUSDT"):
        isolated([p], {})
    with pytest.raises(MissingMarkSeries, match="BTCUSDT"):
        cross([p], {"ETHUSDT": marks([2000.0, 2000.0])})


def test_nonfinite_inputs_propagate() -> None:
    mk = marks([39000, 38000, float("nan"), 38000])
    with pytest.raises(ValueError, match="non-finite mark low"):
        isolated([long_btc_position(sell_bar=3)], {"BTCUSDT": mk})
    bad_px = make_pos(
        "P-5", BTC, [Fill(bar(0), BUY, 1.0, float("nan")), Fill(bar(2), SELL, 1.0, 1.0)]
    )
    with pytest.raises(ValueError, match="non-finite fill price"):
        isolated([bad_px], {"BTCUSDT": marks([39000, 38000, 38000])})
    p = long_btc_position(sell_bar=2)
    with pytest.raises(ValueError, match="non-finite funding"):
        isolated(
            [p],
            {"BTCUSDT": marks([39000, 38000, 38000])},
            funding_by_position={key_of(p): [(bar(1), float("inf"))]},
        )


def skew_first_hit(monkeypatch: pytest.MonkeyPatch, **changes: float) -> None:
    """Make the scanner's first hit disagree with the fills/ledger (a defect simulation)."""
    import vibe_quant.liquidation.overlay as overlay
    from vibe_quant.liquidation.scan import scan_liquidations as real

    def skewed(*args: Any, **kwargs: Any) -> Any:
        res = real(*args, **kwargs)
        res.hits[0] = type(res.hits[0])(**{**res.hits[0].__dict__, **changes})
        return res

    monkeypatch.setattr(overlay, "scan_liquidations", skewed)


def test_ledger_that_disagrees_with_the_scanner_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthetic close must flatten exactly what the fills left open (fail closed)."""
    p = long_btc_position(sell_bar=8)
    mk = marks([39000, 38000, 36200, 36100, 36000, 36000, 36000, 36000, 36000])
    skew_first_hit(monkeypatch, qty=0.5)
    with pytest.raises(ValueError, match="leave qty"):
        isolated([p], {"BTCUSDT": mk})


def _bust_case() -> tuple[list[Pos], dict[str, Marks]]:
    a = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 0.1, 40000.0, 2.0), Fill(bar(6), SELL, 0.1, 29000.0, 1.0)],
    )
    mk = marks([39500, 39000, 38000, 31000, 30000, 30000, 30000])
    return [a], {"BTCUSDT": mk}


FEE_4 = 12.056224899598192  # the forfeit clearance fee of _bust_case (= wallet left at the LP)


@pytest.mark.parametrize("delta", [0.9e-6, -0.9e-6])
def test_cross_residual_just_below_the_absolute_limit_is_booked(
    monkeypatch: pytest.MonkeyPatch, delta: float
) -> None:
    positions, series = _bust_case()
    skew_first_hit(monkeypatch, clearance_fee=FEE_4 + delta)
    ov = cross(positions, series)
    assert ov.ledgers is not None
    equity_close, _ = equity_path(ov, positions, {BTC: series["BTCUSDT"]})
    assert equity_close[-1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("delta", [1.1e-6, -1.1e-6])
def test_cross_residual_above_the_absolute_limit_raises(
    monkeypatch: pytest.MonkeyPatch, delta: float
) -> None:
    """The limit is absolute (1e-6 quote currency), not scaled by the balance of 1000."""
    positions, series = _bust_case()
    skew_first_hit(monkeypatch, clearance_fee=FEE_4 + delta)
    with pytest.raises(ValueError, match="does not wipe the account"):
        cross(positions, series)


# ------------------------------------------------- same-timestamp entry hit, event order


def entry_hit_position(order_type: Name) -> Pos:
    """A 1 BTC long entered 30 s into minute 0 (pos stays open)."""
    fill = Fill(bar(0) + 30_000_000_000, BUY, 1.0, 40000.0, order_type=order_type)
    return make_pos("P-1", BTC, [fill], closed=False)


def entry_hit_marks() -> Marks:
    """Minute 0: O=H=50000, L=36000 (through LP 36144.578), C=39000; minute 1 flat."""
    return marks([36000, 39000], highs=[50000, 39000], closes=[39000, 39000], opens=[50000, 39000])


def test_entry_minute_hit_keeps_the_entry_fill_and_replays_it_before_the_close() -> None:
    """The scanner stamps the hit at the entry's own ts (never before it). The entry fill
    must survive (``ts <= hit.ts``) and replay BEFORE the close, also for a resting LIMIT
    entry (a non-resting close is replayed first and leaves phantom short exposure:
    minimum equity 26000, drawdown 0.35, instead of 36000 / 0.1)."""
    p = entry_hit_position(LIMIT)
    mk = entry_hit_marks()
    ov = isolated([p], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    entry_ts = bar(0) + 30_000_000_000
    assert hit.ts == entry_ts and hit.price == LP_LONG and hit.loss == pytest.approx(4000.0)
    assert ov.ledgers is not None
    entry, close = ov.ledgers[BTC]
    assert (entry.ts, entry.qty, entry.price, entry.resting) == (entry_ts, 1.0, 40000.0, True)
    assert not entry.liquidation and entry.liq_margin == 0.0  # a real fill, no margin cash
    assert (close.ts, close.qty, close.price) == (entry_ts, -1.0, LP_LONG)
    assert close.liquidation  # the synthetic close, replayed right after the entry
    engine = engine_for([p], {BTC: mk})
    dd = mark_to_market_drawdown(engine, 40000.0, execution_timeframe="1m", ledgers=ov.ledgers)
    assert dd == pytest.approx(0.1, abs=1e-12)
    _, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_min.min() == pytest.approx(36000.0, abs=1e-9)


def test_funding_and_cash_at_the_hit_timestamp_stay_real_events_before_the_close() -> None:
    """Funding -0.5 and other cash -0.25 stamped at the hit ts remain events of their own
    (amount, sign and ts untouched), ordered before the synthetic close."""
    p = long_btc_position(sell_bar=8)
    mk = marks([39000, 38000, 37000, 36000, 36000, 36000, 36000, 36000, 36000])
    ov = isolated(
        [p],
        {"BTCUSDT": mk},
        funding_by_position={key_of(p): [(bar(3), -0.5)]},
        cash_by_position={key_of(p): [(bar(3), -0.25)]},
    )
    (hit,) = ov.result.hits
    assert ov.ledgers is not None
    entry, funding, other, close = ov.ledgers[BTC]
    assert strip(funding) == LedgerEvent(ts=bar(3), cash=-0.5)
    assert strip(other) == LedgerEvent(ts=bar(3), cash=-0.25)
    assert (close.qty, close.cash) == (-1.0, -hit.clearance_fee)
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(-0.75 - hit.loss, abs=1e-9)


# ---------------------------------------------------------------- per-position cash


def test_cash_by_position_is_wallet_only_in_isolated_and_truncated_at_the_hit() -> None:
    """Modeled cash (slippage) never moves an isolated margin/LP, is kept up to the hit and
    removed after it."""
    p = long_btc_position(sell_bar=8)
    mk = marks([39000, 38000, 36200, 36100, 36000, 36000, 36000, 36000, 36000])
    base = isolated([p], {"BTCUSDT": mk})
    cash = {key_of(p): [(bar(1), -1.5), (bar(5), -10.0)]}
    ov = isolated([p], {"BTCUSDT": mk}, cash_by_position=cash)
    assert ov.result.hits == base.result.hits  # same hit: cash is not margin
    assert ov.ledgers is not None
    assert [(e.ts, e.cash) for e in ov.ledgers[BTC] if e.qty == 0.0] == [(bar(1), -1.5)]
    (hit,) = ov.result.hits
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(-1.5 - hit.loss, abs=1e-9)


def test_cash_by_position_enters_the_cross_wallet_and_nothing_survives_the_bust() -> None:
    a = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 0.1, 40000.0, 2.0), Fill(bar(6), SELL, 0.1, 29000.0, 1.0)],
    )
    mk = marks([39500, 39000, 38000, 31000, 30000, 30000, 30000])
    cash = {key_of(a): [(bar(2), -1.5), (bar(5), -10.0)]}
    ov = cross([a], {"BTCUSDT": mk}, cash_by_position=cash)
    (hit,) = ov.result.hits
    lp = (996.5 - 4000.0) / (0.004 * 0.1 - 0.1)  # wallet 1000 - 2 fee - 1.5 slippage
    assert hit.price == pytest.approx(lp, rel=1e-12)
    assert ov.ledgers is not None
    assert all(e.ts <= hit.ts for e in ov.ledgers[BTC])
    equity_close, equity_min = equity_path(ov, [a], {BTC: mk})
    assert equity_close[-1] == pytest.approx(0.0, abs=1e-9)
    assert equity_close[3:].min() >= 0.0 and equity_min[3:].min() >= 0.0


def test_cash_by_position_unknown_key_raises() -> None:
    p = long_btc_position(sell_bar=3)
    with pytest.raises(ValueError, match="cash for unknown position keys"):
        position_views([p], cash_by_position={"nope@1": [(bar(0), -1.0)]})


# ------------------------------------- old resting entry, gap liquidation (production ts)


def one_lot_position(side: Name, order_type: Name) -> Pos:
    fill = Fill(bar(0), side, 1.0, 40000.0, order_type=order_type)
    return make_pos("P-1", BTC, [fill], closed=False)


@pytest.mark.parametrize("is_long", [True, False])
def test_gap_liquidation_never_marks_past_the_post_liquidation_equity(
    is_long: bool,
) -> None:
    """1 BTC @ 40000 entered a bar earlier as a resting LIMIT whose resting flag is stale by
    the time of the hit, 10x, balance 40000. Minute 1 gaps to 35000 (long) / 45000 (short):
    price loss 5000, margin 4000, insurance fund 1000, closing event at the
    bankruptcy-equivalent 36000 / 44000. With production exec timestamps the account is
    never below 36000 (drawdown 0.1); a stale resting close walked the bar and hit 35000 /
    45000 (0.125)."""
    gap = 35000.0 if is_long else 45000.0
    p = one_lot_position(BUY if is_long else SELL, LIMIT)
    if is_long:
        mk = marks([39000, gap, gap, gap], opens=[40000, gap, gap, gap])
    else:
        mk = marks(
            [39000, 39000, 39000, 39000],
            highs=[41000, gap, gap, gap],
            closes=[40000, gap, gap, gap],
            opens=[40000, gap, gap, gap],
        )
    ov = isolated([p], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    assert hit.ts == bar(1) and hit.price == gap
    assert hit.insurance_deficit == pytest.approx(1000.0, abs=1e-9)
    assert ov.ledgers is not None
    close = ov.ledgers[BTC][-1]
    assert close.price == pytest.approx(36000.0 if is_long else 44000.0, abs=1e-9)
    dd = mark_to_market_drawdown(
        engine_for([p], {BTC: mk}), 40000.0, execution_timeframe="1m", ledgers=ov.ledgers
    )
    assert dd == pytest.approx(0.1, abs=1e-12)
    equity_close, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_min.min() == pytest.approx(36000.0, abs=1e-9)
    assert equity_close[-1] == pytest.approx(36000.0, abs=1e-9)
    assert close.resting is False  # the entry is a bar old: replay before the bar's path


def test_stale_limit_entry_with_funding_received_at_the_hit_ts() -> None:
    """Funding +100 received at the hit timestamp grows the margin to 4100 (account loss
    4100); the equity floor is 40000 + 100 - 4100 = 36000, drawdown 0.1 (the short mirror
    moves the same way: LP (4100 + 40000)/1.004 = 43924.3, loss 4100)."""
    p = one_lot_position(BUY, LIMIT)
    mk = marks([39000, 38000, 37000, 36000, 36000])  # LP (4100 - 40000)/(.004 - 1) = 36044.18
    ov = isolated([p], {"BTCUSDT": mk}, funding_by_position={key_of(p): [(bar(3), 100.0)]})
    (hit,) = ov.result.hits
    assert hit.ts == bar(3) and hit.loss == pytest.approx(4100.0, abs=1e-9)
    dd = mark_to_market_drawdown(
        engine_for([p], {BTC: mk}), 40000.0, execution_timeframe="1m", ledgers=ov.ledgers
    )
    assert dd == pytest.approx(0.1, abs=1e-12)
    _, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_min.min() == pytest.approx(36000.0, abs=1e-9)


def gap_marks(is_long: bool) -> Marks:
    """Minute 0 around 40000, then minute 1 gaps to 35000 (long) / 45000 (short)."""
    gap = 35000.0 if is_long else 45000.0
    if is_long:
        return marks([39000, gap, gap, gap], opens=[40000, gap, gap, gap])
    return marks(
        [39000] * 4,
        highs=[41000, gap, gap, gap],
        closes=[40000, gap, gap, gap],
        opens=[40000, gap, gap, gap],
    )


@pytest.mark.parametrize(
    ("channel", "amount"),
    [("funding", -1.0), ("cash", -1.0), ("cash", 100.0)],
)
def test_cash_at_the_gap_hit_timestamp_never_marks_the_open_position_at_the_gap(
    channel: str, amount: float
) -> None:
    """Funding / other cash stamped at the hit ts must not be replayed ahead of the close:
    ``pnl_path`` values a cash-only event at the bar open, i.e. the raw gap price (35000 /
    45000), which is past the post-liquidation equity. Funding is margin (loss = margin, so
    the account ends at 36000 whatever the funding; before it arrives the bankruptcy price
    is that of the smaller margin, so the account is 36000 then too); other cash is wallet
    only (35999 for -1, 36100 for +100). The minimum is the lower of the account before the
    cash (36000: the gap past bankruptcy is the insurance fund's) and the final equity; no
    floor at the final equity, which would hide the 36000 before a +100 arrives. One row per
    channel and cash sign; the short and LIMIT-entry variants move every number in that
    same sign and are covered by the gap close tests below."""
    p = one_lot_position(BUY, MARKET)
    mk = gap_marks(True)
    flows = {key_of(p): [(bar(1), amount)]}
    kw = {"funding_by_position": flows} if channel == "funding" else {"cash_by_position": flows}
    ov = isolated([p], {"BTCUSDT": mk}, **kw)
    (hit,) = ov.result.hits
    assert hit.ts == bar(1) and hit.insurance_deficit > 0.0  # the gap really overshoots
    final = 40000.0 - 4000.0 + (amount if channel == "cash" else 0.0)
    assert ov.ledgers is not None
    *real, close = ov.ledgers[BTC]
    assert [strip(e) for e in real][-1] == LedgerEvent(ts=bar(1), cash=amount)  # untouched
    assert close.qty == -1.0 and close.cash == -hit.clearance_fee
    assert close.price - 40000.0 == pytest.approx(hit.realized_pnl, abs=1e-9)
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(final - 40000.0, abs=1e-9)
    lowest = min(final, 36000.0)
    dd = mark_to_market_drawdown(
        engine_for([p], {BTC: mk}), 40000.0, execution_timeframe="1m", ledgers=ov.ledgers
    )
    assert dd == pytest.approx((40000.0 - lowest) / 40000.0, abs=1e-12)
    equity_close, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_min.min() == pytest.approx(lowest, abs=1e-9)
    assert equity_close[-1] == pytest.approx(final, abs=1e-9)


STOP = Name("STOP_MARKET")


@pytest.mark.parametrize("is_long", [True, False])
def test_partial_stop_exit_at_the_hit_timestamp_does_not_mark_the_rest_at_the_gap(
    is_long: bool,
) -> None:
    """1 BTC @ 40000 (10x, balance 40000) entered a bar earlier; minute 1 gaps to 35000 /
    45000 and a STOP_MARKET sells / buys back 0.5 at the gap price at the hit ts: that part
    loses 2500. The rest (margin 2000, price loss 2500) is floored at 2000 by the insurance
    fund. Account loss 4500 -> final 35500, drawdown 0.1125. The reducing fill is valued
    before it executes with the full lot, so at the raw gap price it would put the account
    at 35000 (0.125)."""
    gap = 35000.0 if is_long else 45000.0
    entry_side, exit_side = (BUY, SELL) if is_long else (SELL, BUY)
    p = make_pos(
        "P-1",
        BTC,
        [
            Fill(bar(0), entry_side, 1.0, 40000.0, order_type=MARKET),
            Fill(bar(1), exit_side, 0.5, gap, order_type=STOP),
        ],
        closed=False,
    )
    mk = gap_marks(is_long)
    ov = isolated([p], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    assert hit.ts == bar(1) and hit.qty == 0.5 and hit.loss == pytest.approx(2000.0, abs=1e-9)
    assert ov.ledgers is not None
    entry, stop, close = ov.ledgers[BTC]
    # the real fills are bit-identical to the input: real price, qty, cash, resting flag
    assert strip(entry) == LedgerEvent(
        ts=bar(0), qty=1.0 if is_long else -1.0, price=40000.0, resting=False
    )
    assert strip(stop) == LedgerEvent(
        ts=bar(1), qty=-0.5 if is_long else 0.5, price=gap, cash=0.0, resting=True
    )
    assert ledger_realized([strip(entry), strip(stop)]) == pytest.approx(-2500.0, abs=1e-9)
    # the synthetic close alone carries -hit.loss (bankruptcy-equivalent 36000 / 44000)
    assert close.price == pytest.approx(36000.0 if is_long else 44000.0, abs=1e-9)
    assert -close.qty * (close.price - 40000.0) + close.cash == pytest.approx(-hit.loss)
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(-4500.0, abs=1e-9)
    dd = mark_to_market_drawdown(
        engine_for([p], {BTC: mk}), 40000.0, execution_timeframe="1m", ledgers=ov.ledgers
    )
    assert dd == pytest.approx(0.1125, abs=1e-12)
    equity_close, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_min.min() == pytest.approx(35500.0, abs=1e-9)
    assert equity_close[-1] == pytest.approx(35500.0, abs=1e-9)


def test_partial_stop_and_limit_at_the_hit_cap_the_liquidated_quantity_only() -> None:
    """1 BTC @ 40000 (10x, balance 40000) bought by a resting LIMIT at the hit bar's ts_init,
    where a STOP sells 0.4 @ 35000 and a LIMIT sells 0.5 @ 50000 (all at exec_close_ts(1)).
    That leaves 0.1 with margin 400 and LP 36144.578, which the hit bar's low 35000 crosses:
    the synthetic close books the 400 margin and the account ends 40000 - 2000 + 5000 - 400 =
    42600. The pre-fill instant still holds the whole lot, but only the liquidated 0.1 is
    valued at the cap, so the walk dips to 40000 - 4885.5421686747 = 35114.4578313253
    (drawdown 0.12213855421686749). Valuing the whole lot at the cap hides that dip
    (35686.74698795181, 0.10783132530120475) while the final equity stays 42600."""
    ts = exec_close_ts(1)
    p = make_pos(
        "P-1",
        BTC,
        [
            Fill(ts, BUY, 1.0, 40000.0, order_type=LIMIT),
            Fill(ts, SELL, 0.4, 35000.0, order_type=STOP),
            Fill(ts, SELL, 0.5, 50000.0, order_type=LIMIT),
        ],
        closed=False,
    )
    mk = marks([39000, 35000], highs=[41000, 50000], closes=[40000, 40000], opens=[40000, 45000])
    ov = isolated([p], {"BTCUSDT": mk})
    (hit,) = ov.result.hits
    assert hit.ts == ts and hit.qty == pytest.approx(0.1) and hit.price == pytest.approx(LP_LONG)
    dd = mark_to_market_drawdown(
        engine_for([p], {BTC: mk}), 40000.0, execution_timeframe="1m", ledgers=ov.ledgers
    )
    equity_close, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_min.min() == pytest.approx(35114.4578313253, abs=1e-9)
    assert dd == pytest.approx(0.12213855421686749, abs=1e-12)
    assert equity_close[-1] == pytest.approx(42600.0, abs=1e-9)


def test_multi_leg_cross_bust_with_gap_keeps_every_leg_above_zero() -> None:
    """Cross, wallet 1000: BTC +0.1 @ 40000 (resting LIMIT) and ETH +1 @ 2000 (MARKET)
    opened a bar before the bust. Minute 1 gaps BTC to 25000 (raw -1500) while ETH closes at
    2100 (+100): the account is -400 past zero, so the venue absorbs 400 (BTC closes at the
    bankruptcy-equivalent 29000). Equity must end 0 and never dip below it, bust bar
    included: close [1000, 1100, 0, 0], min [1000, 800, 0, 0]."""
    btc = make_pos("P-1", BTC, [Fill(bar(0), BUY, 0.1, 40000.0, order_type=LIMIT)], False)
    eth = make_pos("P-2", ETH, [Fill(bar(0), BUY, 1.0, 2000.0, order_type=MARKET)], False)
    btc_mk = marks(
        [39000, 25000, 25000],
        highs=[41000, 25000, 25000],
        closes=[40000, 25000, 25000],
        opens=[40000, 25000, 25000],
    )
    eth_mk = marks(
        [1900, 2100, 2100],
        highs=[2200, 2100, 2100],
        closes=[2100, 2100, 2100],
        opens=[2000, 2100, 2100],
    )
    series = {"BTCUSDT": btc_mk, "ETHUSDT": eth_mk}
    ov = cross([btc, eth], series)
    by_symbol = {h.symbol: h for h in ov.result.hits}
    assert by_symbol["BTCUSDT"].insurance_deficit == pytest.approx(400.0, abs=1e-9)
    assert ov.ledgers is not None
    assert ov.ledgers[BTC][-1].price == pytest.approx(29000.0, abs=1e-9)
    assert ov.ledgers[ETH][-1].price == 2100.0 and ov.ledgers[ETH][-1].resting is False
    paths = mark_to_market_paths(
        engine_for([btc, eth], {BTC: btc_mk, ETH: eth_mk}),
        execution_timeframe="1m",
        ledgers=ov.ledgers,
    )
    assert paths is not None
    equity_close = 1000.0 + sum(path[1] for path in paths)
    equity_min = 1000.0 + sum(path[2] for path in paths)
    assert list(np.round(equity_close, 9)) == [1000.0, 1100.0, 0.0, 0.0]
    assert list(np.round(equity_min, 9)) == [1000.0, 800.0, 0.0, 0.0]


# ----------------------------------------------------------- unsupported input, fail closed


def test_mixed_resting_and_market_fills_at_one_ts_raise() -> None:
    ts = bar(0) + 30_000_000_000
    p = make_pos(
        "P-1",
        BTC,
        [
            Fill(ts, BUY, 0.5, 40000.0, order_type=LIMIT),
            Fill(ts, BUY, 0.5, 40000.0, order_type=MARKET),
        ],
        closed=False,
    )
    with pytest.raises(ValueError, match=r"P-1@\d+: fills at ts \d+ mix resting and non-resting"):
        isolated([p], {"BTCUSDT": entry_hit_marks()})


def test_resting_fill_earlier_in_the_hit_execution_bar_raises_unless_bars_are_finer() -> None:
    """LIMIT entry at +10 s, funding -100 at +40 s tips the LP over the minute's low 36200:
    the hit is stamped at the funding ts, in the same 1m execution bar as the resting
    entry, so the replay order would be inexact. A finer execution bar separates them."""
    entry = Fill(bar(0) + 10_000_000_000, BUY, 1.0, 40000.0, order_type=LIMIT)
    p = make_pos("P-1", BTC, [entry], closed=False)
    mk = marks([36200, 36200], highs=[41000, 41000], opens=[40000, 36200])
    funding = {key_of(p): [(bar(0) + 40_000_000_000, -100.0)]}
    with pytest.raises(ValueError, match="a resting fill precedes the liquidation"):
        isolated([p], {"BTCUSDT": mk}, funding_by_position=funding)
    ov = isolated([p], {"BTCUSDT": mk}, funding_by_position=funding, execution_bar_ns=1_000_000_000)
    assert ov.ledgers is not None and ov.ledgers[BTC][-1].resting is False
    with pytest.raises(ValueError, match="execution_bar_ns must be an int > 0"):
        isolated([p], {"BTCUSDT": mk}, funding_by_position=funding, execution_bar_ns=0)


def test_execution_bar_ns_is_required_and_validated() -> None:
    p = long_btc_position(sell_bar=3)
    mk = {"BTCUSDT": marks([39000, 38000, 38000, 38000])}
    kw: dict[str, Any] = {
        "mode": "isolated",
        "starting_balance": 40000.0,
        "leverage": 10.0,
        "tables": TABLES,
    }
    with pytest.raises(TypeError, match="execution_bar_ns"):
        liquidation_overlay([p], mk, **kw)  # type: ignore[call-arg]
    bads: tuple[Any, ...] = (True, 60.0, float("nan"), None, "60", 0, -MIN)
    for bad in bads:
        with pytest.raises(ValueError, match="execution_bar_ns must be an int > 0"):
            liquidation_overlay([p], mk, execution_bar_ns=bad, **kw)
    assert liquidation_overlay([p], mk, execution_bar_ns=MIN, **kw).is_identity


def test_old_limit_entry_in_the_same_coarse_bar_as_the_hit_raises() -> None:
    """With 5 minute execution bars the LIMIT entry (bar 0) shares the hit's bar (bar 1:
    BASE sits 3 minutes into its 5-minute bar), so the close cannot be replayed exactly; with
    1 minute bars the entry is a bar old and the same input is fine."""
    p = one_lot_position(BUY, LIMIT)
    mk = gap_marks(True)
    with pytest.raises(ValueError, match="a resting fill precedes the liquidation"):
        isolated([p], {"BTCUSDT": mk}, execution_bar_ns=5 * MIN)
    assert isolated([p], {"BTCUSDT": mk}, execution_bar_ns=MIN).ledgers is not None


@pytest.mark.parametrize("channel", ["funding", "cash"])
def test_multi_leg_cross_cash_at_the_bust_ts_keeps_every_bar_above_zero(
    channel: str,
) -> None:
    """The multi-leg cross bust of above, plus +100 stamped at the bust ts on the BTC leg
    (funding or other cash). It joins the BTC closing event, so the summed minimum equity
    at and after the bust bar is 0 (not -400), the close path [1000, 1100, 0, 0]."""
    btc = make_pos("P-1", BTC, [Fill(bar(0), BUY, 0.1, 40000.0, order_type=MARKET)], False)
    eth = make_pos("P-2", ETH, [Fill(bar(0), BUY, 1.0, 2000.0, order_type=MARKET)], False)
    btc_mk = marks(
        [39000, 25000, 25000],
        highs=[41000, 25000, 25000],
        closes=[40000, 25000, 25000],
        opens=[40000, 25000, 25000],
    )
    eth_mk = marks(
        [1900, 2100, 2100],
        highs=[2200, 2100, 2100],
        closes=[2100, 2100, 2100],
        opens=[2000, 2100, 2100],
    )
    flows = {key_of(btc): [(bar(1), 100.0)]}
    kw: dict[str, Any] = (
        {"funding_by_position": flows} if channel == "funding" else {"cash_by_position": flows}
    )
    ov = cross([btc, eth], {"BTCUSDT": btc_mk, "ETHUSDT": eth_mk}, **kw)
    assert ov.ledgers is not None
    paths = mark_to_market_paths(
        engine_for([btc, eth], {BTC: btc_mk, ETH: eth_mk}),
        execution_timeframe="1m",
        ledgers=ov.ledgers,
    )
    assert paths is not None
    equity_close = 1000.0 + sum(path[1] for path in paths)
    equity_min = 1000.0 + sum(path[2] for path in paths)
    assert list(np.round(equity_close, 9)) == [1000.0, 1100.0, 0.0, 0.0]
    assert list(np.round(equity_min, 9)) == [1000.0, 800.0, 0.0, 0.0]


def test_every_real_event_is_bit_identical_after_the_overlay() -> None:
    """Real fills (price, qty, commission cash, resting flag) and real funding / cash
    events (amount, sign, ts) come out exactly as they went in; only the synthetic close
    is added; ``liq_margin`` marks the margin cash of the liquidation instant (isolated:
    the funding, not the other cash nor the commission)."""
    entry = Fill(bar(0), BUY, 1.0, 40000.0, 2.0, order_type=LIMIT)
    stop = Fill(bar(1), SELL, 0.5, 35000.0, 1.5, order_type=STOP)
    p = make_pos("P-1", BTC, [entry, stop], closed=False)
    mk = gap_marks(True)
    funding = [(bar(0) + 1, -0.125), (bar(1), 100.0)]
    cash = [(bar(0) + 2, -0.0625), (bar(1), -1.0)]
    ov = isolated(
        [p],
        {"BTCUSDT": mk},
        funding_by_position={key_of(p): funding},
        cash_by_position={key_of(p): cash},
    )
    (hit,) = ov.result.hits
    assert ov.ledgers is not None
    *real, close = [strip(e) for e in ov.ledgers[BTC]]
    assert real == [
        LedgerEvent(ts=bar(0), qty=1.0, price=40000.0, cash=-2.0, resting=True),
        LedgerEvent(ts=bar(0) + 1, cash=-0.125),
        LedgerEvent(ts=bar(0) + 2, cash=-0.0625),
        LedgerEvent(ts=bar(1), qty=-0.5, price=35000.0, cash=-1.5, resting=True),
        LedgerEvent(ts=bar(1), cash=100.0),
        LedgerEvent(ts=bar(1), cash=-1.0),
    ]
    assert close.qty == -0.5 and close.cash == -hit.clearance_fee and not close.resting
    assert close.price == pytest.approx(40000.0 + hit.realized_pnl / 0.5, abs=1e-9)
    assert [e.liq_margin for e in ov.ledgers[BTC]] == [0.0] * 4 + [100.0, 0.0, 0.0]
    assert [e.liquidation for e in ov.ledgers[BTC]] == [False] * 6 + [True]
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(
        -2.0 - 0.125 - 0.0625 - 1.5 + 100.0 - 1.0 - 2500.0 - hit.loss, abs=1e-9
    )


def test_scale_in_and_flip_at_the_liquidation_ts_raise() -> None:
    """Out of the DSL's scope and not replayable exactly: fail closed, naming the position."""
    scale = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 0.5, 40000.0), Fill(bar(1), BUY, 0.5, 35000.0)],
        closed=False,
    )
    mk = marks([39000, 33000, 33000], opens=[40000, 33000, 33000])
    with pytest.raises(ValueError, match=r"P-1@\d+: a scale-in fill at the liquidation ts"):
        isolated([scale], {"BTCUSDT": mk})
    flip = make_pos(
        "P-2",
        BTC,
        [Fill(bar(0), BUY, 0.5, 40000.0), Fill(bar(1), SELL, 1.0, 40000.0)],
        closed=False,
    )
    short_mk = marks(
        [39000] * 3,
        highs=[41000, 45000, 45000],
        closes=[40000, 45000, 45000],
        opens=[40000, 45000, 45000],
    )
    with pytest.raises(ValueError, match=r"P-2@\d+: a fill at the liquidation ts .* flips"):
        isolated([flip], {"BTCUSDT": short_mk})


# ------------------------------------- genuine losses before the liquidation stay visible


def exec_close_ts(i: int) -> int:
    """Production ts of an event stamped by NT while processing mark minute ``i``: the 1m
    execution bar's ``ts_init`` (its close less 1 ms)."""
    return bar(i) + MIN - 1_000_000


def test_profitable_partial_reduction_at_the_hit_keeps_the_genuine_bar_open_loss() -> None:
    """1 BTC @ 40000 (10x, balance 40000) rides to 50000: equity 50000. The hit minute gaps
    to 45000 (equity 45000, a genuine loss of 5000 on the full lot), then a resting LIMIT
    takes 0.9 off at 50000 and funding -200 is paid, both at the bar's ts_init. That leaves
    0.1 with margin 200 and LP 38152.61, which the low 37000 crosses: hit at that ts, no
    insurance deficit, loss 200. Final 40000 + 9000 - 200 - 200 = 48600. The bar-open equity
    45000 precedes the liquidation and must stay the minimum (drawdown 0.1); a floor at the
    instant's final equity reports 48600 / 0.028. The short mirror is the same walk with the
    sign of every price flipped."""
    entry_side, exit_side = BUY, SELL
    target = 50000.0
    ts = exec_close_ts(2)
    p = make_pos(
        "P-1",
        BTC,
        [
            Fill(bar(0), entry_side, 1.0, 40000.0),
            Fill(ts, exit_side, 0.9, target, order_type=LIMIT),
        ],
        closed=False,
    )
    mk = marks(
        [40000, 40000, 37000, 38000],
        highs=[41000, 50000, 50000, 38000],
        closes=[40000, 50000, 38000, 38000],
        opens=[40000, 40000, 45000, 38000],
    )
    ov = isolated([p], {"BTCUSDT": mk}, funding_by_position={key_of(p): [(ts, -200.0)]})
    (hit,) = ov.result.hits
    assert hit.ts == ts and hit.qty == pytest.approx(0.1) and hit.insurance_deficit == 0.0
    assert hit.loss == pytest.approx(200.0, abs=1e-9)
    assert ov.ledgers is not None
    assert ledger_realized(ov.ledgers[BTC]) == pytest.approx(8600.0, abs=1e-9)
    dd = mark_to_market_drawdown(
        engine_for([p], {BTC: mk}), 40000.0, execution_timeframe="1m", ledgers=ov.ledgers
    )
    assert dd == pytest.approx(0.1, abs=1e-12)
    equity_close, equity_min = equity_path(ov, [p], {BTC: mk}, 40000.0)
    assert equity_close[2] == pytest.approx(50000.0, abs=1e-9)  # the peak before the hit bar
    assert equity_min[3] == pytest.approx(45000.0, abs=1e-9)  # hit bar: the gapped open
    assert equity_min[3:].min() == pytest.approx(45000.0, abs=1e-9)  # nothing lower after it
    assert equity_close[-1] == pytest.approx(48600.0, abs=1e-9)


def test_cross_bust_inside_the_bar_shows_the_other_leg_dip_before_it() -> None:
    """Cross, wallet 1000: BTC +0.1 @ 40000 and ETH +1 @ 2000. In minute 1 BTC dips to 38000
    (-200) and ETH to 1850 (-150), margin balance 650: alive. Funding -650 on BTC at the
    bar's ts_init busts the account there: BTC closes at its cross LP, ETH at its bar open
    2000 (the scanner's mid-minute exit), forfeit -> equity 0. ETH's dip happened BEFORE the
    bust, so its own intrabar minimum in that bar is -150 (moving the bust to the bar open
    hides it: 0). That dip plus BTC's post-bust -1000 sums to -150 on the bust bar; the
    account cannot be below 0 there, so with ``floor_from_ns=bust_ts`` the account minimum
    is clamped at 0 from the bust bar on, and nowhere else."""
    btc = make_pos("P-1", BTC, [Fill(bar(0), BUY, 0.1, 40000.0)], False)
    eth = make_pos("P-2", ETH, [Fill(bar(0), BUY, 1.0, 2000.0)], False)
    btc_mk = marks(
        [39500, 38000, 38500],
        highs=[40500, 39000, 38500],
        closes=[40000, 38500, 38500],
        opens=[40000, 39000, 38500],
    )
    eth_mk = marks(
        [1950, 1850, 1900],
        highs=[2050, 2000, 1900],
        closes=[2000, 1900, 1900],
        opens=[2000, 2000, 1900],
    )
    ts = exec_close_ts(1)
    ov = cross(
        [btc, eth],
        {"BTCUSDT": btc_mk, "ETHUSDT": eth_mk},
        funding_by_position={key_of(btc): [(ts, -650.0)]},
    )
    by_symbol = {h.symbol: h for h in ov.result.hits}
    assert ov.result.bust_ts == ts and set(by_symbol) == {"BTCUSDT", "ETHUSDT"}
    assert by_symbol["ETHUSDT"].price == 2000.0 and by_symbol["BTCUSDT"].insurance_deficit == 0.0
    assert ov.ledgers is not None
    paths = mark_to_market_paths(
        engine_for([btc, eth], {BTC: btc_mk, ETH: eth_mk}),
        execution_timeframe="1m",
        ledgers=ov.ledgers,
    )
    assert paths is not None
    (btc_path, eth_path) = sorted(paths, key=lambda path: float(path[1][1]))  # ETH ends at 0
    assert eth_path[2][2] == pytest.approx(-150.0, abs=1e-9)  # ETH's dip before the bust
    assert btc_path[2][2] == pytest.approx(-1000.0, abs=1e-9)  # BTC ends at the wallet loss
    _grid, raw_close, raw_min = mark_to_market_equity(1000.0, paths)
    assert list(np.round(raw_min, 9)) == [1000.0, 900.0, -150.0, 0.0]  # conservative sum
    grid, equity_close, equity_min = mark_to_market_equity(
        1000.0, paths, floor_from_ns=ov.result.bust_ts
    )
    assert int(grid[2]) == ts  # the bust bar is the one holding bust_ts
    assert list(np.round(equity_close, 9)) == [1000.0, 1000.0, 0.0, 0.0]
    assert np.array_equal(equity_close, raw_close)
    assert list(np.round(equity_min, 9)) == [1000.0, 900.0, 0.0, 0.0]
    dd = mark_to_market_drawdown(
        engine_for([btc, eth], {BTC: btc_mk, ETH: eth_mk}),
        1000.0,
        execution_timeframe="1m",
        ledgers=ov.ledgers,
        floor_from_ns=ov.result.bust_ts,
    )
    assert dd == 1.0


def test_equity_floor_clamps_only_bars_at_or_after_the_bust_ts() -> None:
    """``floor_from_ns`` clamps ``equity_min`` at 0 from the bar holding the bust ts on, never
    on the bar before it. The floor 60000000000 sits strictly between the bars at 59999000000
    and 119999000000: the first bar keeps its loss (1000 - 1050 = -50, negative and still not
    clamped), the following bars floor their 1000 - 1150 = -150 at 0. Equity closes are
    [1100, 0, 0] and the drawdown stays 1.0 either way, so only ``equity_min`` shows whether
    the clamp leaked onto the earlier bar (a ``<=`` clamp returns [0, -150, 0])."""
    ts = np.array([59_999_000_000, 119_999_000_000, 179_999_000_000], dtype=np.int64)
    path = (
        ts,
        np.array([100.0, -1000.0, -1000.0]),
        np.array([-1050.0, -1150.0, -1000.0]),
    )
    grid, _equity_close, equity_min = mark_to_market_equity(
        1000.0, [path], floor_from_ns=60_000_000_000
    )
    assert np.array_equal(grid, ts)
    assert list(equity_min) == [-50.0, 0.0, 0.0]


def test_another_position_in_the_liquidation_bar_raises() -> None:
    """A's real exit (+20 s) is after its hit at the minute-3 open and is dropped; B opens at
    +40 s in the same execution bar. pnl_path would replay B inside A's liquidation instant
    and net the two, so the overlay fails closed naming both positions."""
    a = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 1.0, 40000.0), Fill(bar(3) + 20_000_000_000, SELL, 1.0, 36500.0)],
    )
    b = make_pos(
        "P-2",
        BTC,
        [Fill(bar(3) + 40_000_000_000, BUY, 1.0, 36500.0), Fill(bar(5), SELL, 1.0, 36000.0)],
    )
    mk = marks([39000, 38000, 37000, 36000, 36000, 36000])
    with pytest.raises(ValueError, match=r"P-1@\d+: position P-2@\d+ has an event at ts"):
        isolated([a, b], {"BTCUSDT": mk})


def test_pnl_path_rejects_an_event_after_the_liquidation_close_in_its_bar() -> None:
    mk = marks([39000, 36000, 36000])
    series = exec_series(mk)
    events = [
        LedgerEvent(ts=bar(0), qty=1.0, price=40000.0),
        LedgerEvent(ts=bar(1), qty=-1.0, price=LP_LONG, liquidation=True),
        LedgerEvent(ts=bar(1) + 1, cash=-1.0),
    ]
    with pytest.raises(ValueError, match="event after the liquidation"):
        pnl_path(series, events)
    with pytest.raises(ValueError, match="position still open after its liquidation"):
        pnl_path(series, [events[0], replace(events[1], qty=-0.5)])


# --------------------------------------------- refined scan timestamps (R24, R24b, R31, vjno2)


def test_partial_refined_exit_then_hit_fails_closed_with_refined_exit_conflict() -> None:
    """R24b isolated probe: a partial SELL stop of 0.5 out of 1.0 is refined to the ``bar(1)``
    open (``r``) while its NT timestamp stays the ``bar(2)`` close (``n``). The mark crosses the
    LP at ``bar(2)``, so the scanner (which sees 0.5 open from ``bar(1)``) liquidates the
    remaining half there, but the ledger keeps NT timestamps and still holds the full lot at
    ``bar(2)``. The overlay must fail closed with ``RefinedExitConflict`` naming the key, the
    fill index, ``r``, ``n`` and ``X`` (the hit ts) instead of the unnamed qty mismatch
    ("leave qty 1.0, scanner closed 0.5")."""
    entry = Fill(bar(0), BUY, 1.0, 40000.0)
    stop = Fill(exec_close_ts(2), SELL, 0.5, 36000.0, order_type=STOP)
    p = make_pos("P-1", BTC, [entry, stop], closed=False)
    mk = marks([39000, 39000, 36000, 36000, 36000])
    r, x = bar(1), bar(2)
    with pytest.raises(RefinedExitConflict) as excinfo:
        isolated([p], {"BTCUSDT": mk}, scan_fill_ts={key_of(p): {1: r}})
    msg = str(excinfo.value)
    assert key_of(p) in msg
    assert "fill 1" in msg
    assert str(r) in msg and str(stop.ts_event) in msg and str(x) in msg


def test_refined_exit_at_the_hit_instant_fails_closed_with_refined_exit_conflict() -> None:
    """R24b lower boundary, ``r == X < n``: the partial SELL stop of 0.5 out of 1.0 is refined to
    ``bar(1)``, so the scanner holds the full lot only through minute 0 and the remaining half
    crosses the LP in minute 1 - the refined fill's own bar - which makes the hit ts the refined
    instant itself (``r == X``). The ledger keeps NT timestamps and its ``ts <= X`` filter drops the
    fill (``n = exec_close_ts(3) > X``), so it still holds the full lot at ``X``. The boundary IS a
    conflict and must raise the named ``RefinedExitConflict``; a strict ``r < X`` comparison skips it
    and falls through to the unnamed qty mismatch of ``_closing_event`` ("leave qty 1.0, scanner
    closed 0.5"). The unrefined control pins the geometry: really stamped at ``r``, the same exit
    liquidates exactly 0.5 at ``X == r``."""
    entry = Fill(bar(0), BUY, 1.0, 40000.0)
    stop = Fill(exec_close_ts(3), SELL, 0.5, 36000.0, order_type=STOP)
    p = make_pos("P-1", BTC, [entry, stop], closed=False)
    mk = marks([39000, 36000, 36000, 36000, 36000])
    r, n = bar(1), exec_close_ts(3)
    genuine = make_pos(
        "P-1", BTC, [entry, Fill(r, SELL, 0.5, 36000.0, order_type=STOP)], closed=False
    )
    (control,) = isolated([genuine], {"BTCUSDT": mk}).result.hits
    assert control.ts == r and control.qty == 0.5  # X == r for this exit
    with pytest.raises(RefinedExitConflict) as excinfo:
        isolated([p], {"BTCUSDT": mk}, scan_fill_ts={key_of(p): {1: r}})
    msg = str(excinfo.value)
    assert key_of(p) in msg
    assert "fill 1" in msg
    assert str(r) in msg and str(n) in msg  # X == r, so r and X share one repr


def test_refined_exit_with_nt_ts_at_the_hit_instant_liquidates_the_half() -> None:
    """R24b upper boundary, ``r < n == X``: the partial SELL stop of 0.5 out of 1.0 is NT-stamped
    exactly at the minute-2 mark bar open and refined to ``bar(1)``. The scanner sees 0.5 open from
    minute 1 and the remaining half crosses the LP in minute 2, so its hit ts is exactly ``n``. Both
    consumers then keep the fill (the scan applies it at ``r < X``, the ledger's ``ts <= X`` filter
    keeps ``n == X``), so the liquidation of quantity 0.5 is valid and must NOT raise; a comparison
    with ``X <= n`` rejects it. The same exit really stamped at ``n`` is the control: it liquidates
    the identical half, so the refined run's hits, ledgers and exit override must match it."""
    entry = Fill(bar(0), BUY, 1.0, 40000.0)
    stop = Fill(bar(2), SELL, 0.5, 36000.0, order_type=STOP)
    p = make_pos("P-1", BTC, [entry, stop], closed=False)
    genuine = make_pos(
        "P-1", BTC, [entry, Fill(bar(2), SELL, 0.5, 36000.0, order_type=STOP)], closed=False
    )
    mk = marks([39000, 39000, 36000, 36000, 36000])
    base = isolated([genuine], {"BTCUSDT": mk})
    (hit,) = base.result.hits
    assert hit.ts == stop.ts_event == bar(2) and hit.qty == 0.5  # X == n, the half liquidates
    ov = isolated([p], {"BTCUSDT": mk}, scan_fill_ts={key_of(p): {1: bar(1)}})
    assert ov.result.hits == base.result.hits
    assert ov.ledgers == base.ledgers
    assert ov.exit_override[key_of(p)] == (bar(2), hit.price)
    assert ov.ledgers is not None
    close = ov.ledgers[BTC][-1]
    assert (close.qty, close.liquidation) == (-0.5, True)


def cross_refined_case() -> tuple[list[Pos], dict[str, Marks], str]:
    """R24b cross probe: a BTC leg fully exited (refined to ``bar(1)``, NT ``exec_close_ts(6)``,
    post-bust funding +10) plus an ETH leg that busts the account at ``bar(3)``."""
    btc = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 0.1, 40000.0), Fill(exec_close_ts(6), SELL, 0.1, 40000.0)],
        closed=False,
    )
    eth = make_pos("P-2", ETH, [Fill(bar(0), BUY, 1.0, 2000.0)], closed=False)
    series = {
        "BTCUSDT": marks([40000.0] * 8, highs=[41000.0] * 8),
        "ETHUSDT": marks([2000.0] * 8, highs=[2000.0] * 8),
    }
    return [btc, eth], series, key_of(btc)


def test_cross_refined_exit_then_bust_fails_closed_with_refined_exit_conflict() -> None:
    """R24b cross probe: the BTC leg is flat in the scan from ``bar(1)`` (``r``) but its NT exit
    is ``exec_close_ts(6)`` (``n``) and it carries post-bust funding +10; the ETH leg busts the
    account at ``bar(3)`` (``X``). The ledger keeps BTC's exit and its +10, which the scan never
    had, so the account no longer wipes to 0 (the pre-rule error was "equity 10.0" from the
    residual check). The overlay must fail closed with ``RefinedExitConflict`` naming the key,
    the fill index, ``r``, ``n`` and ``X``."""
    positions, series, btc_key = cross_refined_case()
    funding = {
        btc_key: [(exec_close_ts(7), 10.0)],
        key_of(positions[1]): [(bar(3), -1300.0)],
    }
    r, x = bar(1), bar(3)
    n = exec_close_ts(6)
    with pytest.raises(RefinedExitConflict) as excinfo:
        cross(positions, series, funding_by_position=funding, scan_fill_ts={btc_key: {1: r}})
    msg = str(excinfo.value)
    assert btc_key in msg
    assert "fill 1" in msg
    assert str(r) in msg and str(n) in msg and str(x) in msg
    # control: without the refinement the same run completes, with the bust at bar(3)
    ov = cross(positions, series, funding_by_position=funding)
    assert ov.result.bust_ts == x and {h.key for h in ov.result.hits} == set(funding)


def test_refined_scan_timestamp_reaches_only_the_scanner_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R24 view-only override: the PositionView the scanner gets carries the refined instant,
    while the ledgers keep NT timestamps object for object (the hit is on the other instrument,
    so the refined position's ledger is exactly the unrefined one)."""
    import vibe_quant.liquidation.overlay as overlay
    from vibe_quant.liquidation.scan import scan_liquidations as real

    seen: dict[str, list[int]] = {}

    def spy(views: Any, marks_arg: Any, **kw: Any) -> Any:
        seen.update({v.key: [int(e.ts) for e in v.events] for v in views})
        return real(views, marks_arg, **kw)

    monkeypatch.setattr(overlay, "scan_liquidations", spy)

    from nautilus_trader.model.enums import OrderSide, OrderType

    btc = long_btc_position(sell_bar=8)
    eth = make_pos(
        "P-2",
        ETH,
        [
            Fill(bar(0), OrderSide.BUY, 10.0, 2000.0, order_type=OrderType.MARKET),
            Fill(exec_close_ts(6), OrderSide.SELL, 10.0, 2050.0, order_type=OrderType.MARKET),
        ],
    )
    series = {
        "BTCUSDT": marks([39000, 38000, 36200, 36100, 36000, 36000, 36000, 36000, 36000]),
        "ETHUSDT": marks([2000.0] * 8, highs=[2100.0] * 8),
    }
    r = bar(1)
    ov = isolated([btc, eth], series, scan_fill_ts={key_of(eth): {1: r}})
    assert seen[key_of(eth)][-1] == r  # the scanner saw the refined instant
    assert seen[key_of(btc)][-1] == bar(8)  # untouched positions keep NT timestamps
    assert ov.ledgers is not None
    assert ov.ledgers[ETH] == position_ledger_events([eth])[ETH]  # NT ts, untouched


def test_non_straddling_refined_scan_timestamps_keep_the_cross_bust_and_residual() -> None:
    """R24/R31: a refinement whose NT ts is also at or before the bust (``r <= n <= X``) changes
    no hit and no ledger - the cash lands at ``r`` in the scan and at ``n`` in the ledger, both
    before ``X`` - and the account still wipes to 0 there (the residual check holds)."""
    btc = make_pos(
        "P-1",
        BTC,
        [Fill(bar(0), BUY, 0.1, 40000.0), Fill(exec_close_ts(2), SELL, 0.1, 40000.0)],
        closed=False,
    )
    eth = make_pos("P-2", ETH, [Fill(bar(0), BUY, 1.0, 2000.0)], closed=False)
    btc_mk = marks([40000.0] * 8, highs=[41000.0] * 8)
    eth_mk = marks([2000.0] * 8, highs=[2000.0] * 8)
    series = {"BTCUSDT": btc_mk, "ETHUSDT": eth_mk}
    funding = {key_of(eth): [(bar(3), -1300.0)]}
    base = cross([btc, eth], series, funding_by_position=funding)
    refined = cross(
        [btc, eth],
        series,
        funding_by_position=funding,
        scan_fill_ts={key_of(btc): {1: bar(1)}},
    )
    assert base.result.bust_ts == bar(3) == refined.result.bust_ts
    assert base.result.hits == refined.result.hits
    assert base.ledgers == refined.ledgers
    paths = mark_to_market_paths(
        engine_for([btc, eth], {BTC: btc_mk, ETH: eth_mk}),
        execution_timeframe="1m",
        ledgers=refined.ledgers,
    )
    assert paths is not None
    assert 1000.0 + sum(path[1] for path in paths)[-1] == pytest.approx(0.0, abs=1e-9)


def test_refined_scan_timestamp_credits_the_cross_wallet_at_r() -> None:
    """R31: the cross wallet is credited at the refined instant. The long is fully exited at the
    refined ``bar(1)``, so it is flat through the ``bar(2)`` crash and the account never busts;
    an unrefined run whose fill is genuinely stamped ``bar(1)`` gives the same scan result, while
    the NT timestamp (``exec_close_ts(5)``) busts the account there."""
    entry = Fill(bar(0), BUY, 0.1, 40000.0)
    late_exit = Fill(exec_close_ts(5), SELL, 0.1, 42000.0)
    p = make_pos("P-1", BTC, [entry, late_exit], closed=False)
    mk = marks([39500.0, 39500.0] + [30000.0] * 6, highs=[40500.0, 40500.0] + [30000.0] * 6)
    series = {"BTCUSDT": mk}
    control = cross([p], series)
    assert not control.is_identity
    assert [h.ts for h in control.result.hits] == [bar(2)]  # open through the crash
    refined = cross([p], series, scan_fill_ts={key_of(p): {1: bar(1)}})
    genuine = cross([make_pos("P-1", BTC, [entry, Fill(bar(1), SELL, 0.1, 42000.0)], closed=False)], series)
    assert refined.is_identity and genuine.is_identity  # flat before the crash: no bust
    assert refined.result.hits == genuine.result.hits == []
    assert refined.result.bust_ts is None


def test_scan_fill_ts_rejects_unknown_keys_and_indices() -> None:
    p = long_btc_position(sell_bar=3)
    mk = {"BTCUSDT": marks([39000, 38000, 38000, 38000])}
    with pytest.raises(ValueError, match="refined scan timestamps for unknown position keys"):
        isolated([p], mk, scan_fill_ts={"nope@1": {0: bar(0)}})
    with pytest.raises(ValueError, match=r"fill index 5"):
        isolated([p], mk, scan_fill_ts={key_of(p): {5: bar(0)}})
