"""Liquidation scanner on 1m mark series (bead vibe-quant-2gyk1.1.4).

Marks are synthetic one-minute bars; expectations are hand-computed in each
docstring from the Binance one-way formula
``LP = (WB - TMM1 + UPNL1 + cum - side*q*EP) / (q*MMR - side*q)``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from vibe_quant.liquidation.scan import (
    MissingMarkSeries,
    PositionView,
    scan_liquidations,
)
from vibe_quant.liquidation.tiers import Bracket, TierTable, cross_liq_price, load_tier_tables

MIN = 60_000_000_000
BASE = 28_333_333 * MIN
_BTC = load_tier_tables()["BTCUSDT"]
LP_LONG = 36144.57831325301  # (4000 - 40000) / (0.004 - 1)
LP_SHORT = 43824.701195219124  # (4000 + 40000) / (0.004 + 1)


class Marks(NamedTuple):
    ts: np.ndarray
    low: np.ndarray
    high: np.ndarray
    close: np.ndarray
    open: np.ndarray


class MarksNoOpen(NamedTuple):
    ts: np.ndarray
    low: np.ndarray
    high: np.ndarray
    close: np.ndarray


class Ev(NamedTuple):
    ts: int
    qty: float
    price: float
    cash: float = 0.0


def bar(i: int) -> int:
    return BASE + i * MIN


def marks(lows, highs=None, closes=None, opens=None, start=0) -> Marks:
    """1m mark bars from bar(start). Without ``opens`` the mark is continuous: each bar
    opens at the previous bar's close (the first bar at its own close)."""
    n = len(lows)
    low = np.asarray(lows, dtype=np.float64)
    high = np.full(n, 41000.0) if highs is None else np.asarray(highs, dtype=np.float64)
    close = low.copy() if closes is None else np.asarray(closes, dtype=np.float64)
    if opens is None:
        open_ = np.concatenate((close[:1], close[:-1]))
    else:
        open_ = np.asarray(opens, dtype=np.float64)
    ts = (np.arange(n, dtype=np.int64) + start) * MIN + BASE
    return Marks(ts, low, high, close, open_)


def at(ts_list, mk: Marks) -> Marks:
    """Re-time ``mk``'s bars to ``ts_list`` (minutes missing in between are mark gaps)."""
    return mk._replace(ts=np.asarray(ts_list, dtype=np.int64))


def flat_table(symbol: str, mmr: float, rate: float | None = None) -> TierTable:
    return TierTable(symbol, (Bracket(0.0, 1e12, mmr, 0.0, None),), rate)


def view(key="p1", symbol="BTCUSDT", events=(), funding=()) -> PositionView:
    return PositionView(key, symbol, f"{symbol}-PERP.BINANCE", list(events), list(funding))


def long_btc(exit_bar: int | None = None, **kw) -> PositionView:
    events = [Ev(bar(0), 1.0, 40000.0)]
    if exit_bar is not None:
        events.append(Ev(bar(exit_bar), -1.0, 38000.0))
    return view(events=events, **kw)


def isolated(views, mk, tables=None, leverage=10.0):
    return scan_liquidations(
        views,
        mk,
        mode="isolated",
        starting_balance=40000.0,
        leverage=leverage,
        tables={"BTCUSDT": _BTC} if tables is None else tables,
    )


def test_isolated_long_gap_fills_at_trigger():
    """10x long q=1 @ 40000: margin 4000, tier1 MMR .004 -> LP = (4000-40000)/(.004-1)
    = 36144.57831325301. Lows [39000, 36200, 36144.6, 36100]: 36144.6 > LP does not
    trigger; bar 3 (36100) does. Loss = remaining margin = 4000 (shipped table has
    no clearance rate -> forfeit)."""
    res = isolated([long_btc()], {"BTCUSDT": marks([39000, 36200, 36144.6, 36100])})
    (hit,) = res.hits
    assert hit.ts == bar(3)
    assert hit.price == LP_LONG
    assert hit.side == 1 and hit.qty == 1.0
    assert hit.mark == 36100.0
    assert hit.realized_pnl == pytest.approx(LP_LONG - 40000.0, abs=1e-9)
    assert hit.loss == pytest.approx(4000.0, abs=1e-9)
    assert hit.maint_margin == pytest.approx(LP_LONG * 0.004, abs=1e-9)
    assert res.path_divergence is True and res.bust_ts is None
    notes = res.to_notes(n_trades=4)
    assert notes["count"] == 1 and notes["pct_trades"] == 25.0
    # forfeit rule: the fee is the margin left after the price loss; price loss + fee = margin
    assert notes["total_forfeited"] == pytest.approx(4000.0 + (LP_LONG - 40000.0), abs=1e-9)
    assert notes["total_realized_pnl"] == pytest.approx(LP_LONG - 40000.0, abs=1e-9)
    assert notes["total_insurance_deficit"] == 0.0
    assert notes["mark_source"] == "1m_mark" and notes["mode"] == "isolated"


def test_isolated_gap_past_lp_fills_at_the_bar_open():
    """The bar opens at 35500, already below LP 36144.578 (a gap): the long is taken over
    at the open (min(LP, open) = 35500). A bar that opens above LP and crashes through it
    fills at LP. Loss is floored at the 4000 margin either way. Short mirror: a bar opening
    at 44500 > LP 43824.70 fills at max(LP, open) = 44500."""
    gapped = marks([39000, 35000], opens=[40000, 35500])
    res = isolated([long_btc()], {"BTCUSDT": gapped})
    (hit,) = res.hits
    assert hit.price == 35500.0 and hit.mark == 35000.0
    assert hit.loss == pytest.approx(4000.0, abs=1e-9)
    assert hit.insurance_deficit == pytest.approx(500.0, abs=1e-9)  # 4500 price loss - 4000
    assert res.to_notes(1)["total_insurance_deficit"] == pytest.approx(500.0, abs=1e-9)
    through = marks([39000, 35000], opens=[40000, 39000])
    (hit,) = isolated([long_btc()], {"BTCUSDT": through}).hits
    assert hit.price == LP_LONG
    short = view(events=[Ev(bar(0), -1.0, 40000.0)])
    short_mk = marks([39000, 39000], highs=[41000, 45000], opens=[40000, 44500])
    (hit,) = isolated([short], {"BTCUSDT": short_mk}).hits
    assert hit.price == 44500.0 and hit.side == -1


def test_isolated_short_mirror():
    """10x short: LP = (4000 + 40000)/(.004 + 1) = 43824.701195219124. Highs
    [41000, 43700, 43824.7, 43900]: 43824.7 < LP no trigger, bar 3 triggers."""
    short = view(events=[Ev(bar(0), -1.0, 40000.0)])
    mk = marks([39000] * 4, highs=[41000, 43700, 43824.7, 43900])
    (hit,) = isolated([short], {"BTCUSDT": mk}).hits
    assert hit.ts == bar(3)
    assert hit.price == LP_SHORT
    assert hit.side == -1
    assert hit.mark == 43900.0
    assert hit.loss == pytest.approx(4000.0, abs=1e-9)


def test_low_just_above_lp_does_not_trigger_but_equal_does():
    assert not isolated([long_btc()], {"BTCUSDT": marks([39000, LP_LONG + 1e-4])}).hits
    assert isolated([long_btc()], {"BTCUSDT": marks([39000, LP_LONG])}).hits


def test_isolated_hits_where_half_equity_cross_never_does():
    """Same path, same fills: q=0.5 @ 40000 with a 40000 wallet is 0.5x exposure.
    Isolated 10x: margin 2000 -> LP 36144.578 -> hit at low 36000. Cross: MB =
    40000 + .5*(36000 - 40000) = 38000 >> MM = .004*.5*36000 = 72 -> no hit."""
    v = view(events=[Ev(bar(0), 0.5, 40000.0)])
    mk = {"BTCUSDT": marks([39000, 38000, 36000])}
    tables = {"BTCUSDT": _BTC}
    iso = scan_liquidations(
        [v], mk, mode="isolated", starting_balance=40000.0, leverage=10.0, tables=tables
    )
    cross = scan_liquidations(
        [v], mk, mode="cross", starting_balance=40000.0, leverage=10.0, tables=tables
    )
    assert [h.ts for h in iso.hits] == [bar(2)]
    assert iso.hits[0].price == LP_LONG
    assert cross.is_empty and cross.bust_ts is None


def test_tier_boundary_crossed_along_the_path():
    """Tier1 [0,50000) MMR .004 cum 0; tier2 [50000,1e12) MMR .005 cum 50. Long q=1
    @ 40000 (margin 4000, LP 36144.578 in tier1), then +1 @ 40000 at bar 1: q=2,
    margin 8000. Tier1 pricing -> LP 36144.578 -> notional 72289 (tier2, inconsistent);
    tier2 pricing -> (8000 + 50 - 80000)/(2*.005 - 2) = 71950/1.99 =
    36155.77889447236 (notional 72311, tier2). The low 36150 only crosses the tier-2 LP."""
    table = TierTable(
        "BTCUSDT",
        (Bracket(0.0, 50000.0, 0.004, 0.0, None), Bracket(50000.0, 1e12, 0.005, 50.0, None)),
        None,
    )
    v = view(events=[Ev(bar(0), 1.0, 40000.0), Ev(bar(1), 1.0, 40000.0)])
    (hit,) = isolated([v], {"BTCUSDT": marks([39000, 38000, 36150])}, {"BTCUSDT": table}).hits
    assert hit.ts == bar(2)
    assert hit.price == pytest.approx(71950.0 / 1.99, rel=1e-14)
    assert hit.qty == 2.0
    assert hit.loss == pytest.approx(8000.0, abs=1e-9)
    # single-tier control: before the scale-in the tier-1 LP 36144.578 would not be hit
    v1 = view(events=[Ev(bar(0), 1.0, 40000.0)])
    assert not isolated([v1], {"BTCUSDT": marks([39000, 38000, 36150])}, {"BTCUSDT": table}).hits


def test_funding_payment_below_mm_triggers_at_next_bar_with_new_lp():
    """Margin 4000, funding -1000 at bar 2 -> margin 3000 -> LP = (3000-40000)/(.004-1)
    = 37000/.996 = 37148.59437751004. Low 37000 is above the old LP 36144.578 on bars 0-1
    (no hit) but below the new LP on bar 2, the first bar with open_ts >= the settlement."""
    v = long_btc(funding=[(bar(2), -1000.0)])
    mk = marks([37000.0] * 5, opens=[38000.0] * 5)
    res = isolated([v], {"BTCUSDT": mk})
    (hit,) = res.hits
    assert hit.ts == bar(2)
    assert hit.price == pytest.approx(37000.0 / 0.996, rel=1e-14)
    assert hit.loss == pytest.approx(3000.0, abs=1e-9)
    # without the funding payment the same path never liquidates
    assert not isolated([long_btc()], {"BTCUSDT": mk}).hits


def test_cross_eth_crash_liquidates_both_legs_and_drops_later_position():
    """Wallet 1000, BTC long .01 @ 40000 (flat mark), ETH long 2 @ 2000 (MMR .005,
    BTC .004). MB = 1000 + 2*(p - 2000) = 2p - 3000; MM = .004*400 + .005*2p =
    1.6 + .01p. ETH lows 1900, 1600, 1500: MB-MM = 1000+2(p-2000) - 1.6 - .01p;
    bar0 800-20.6 >0, bar1 200-17.6 >0, bar2 0-16.6 <= 0 -> bust at bar 2.
    ETH is the largest adverse uPnL: cross LP = (1000 - 1.6 + 0 - 4000)/(.01 - 2) =
    3001.6/1.99 = 1508.3417085427136 (inside the bar's [1500, 1600]). BTC closes at
    its mark close 40000. BTC position #1's NT exit (bar 6) is ignored; the BTC
    position opened at bar 8 is dropped."""
    n = 10
    btc = marks([40000.0] * n, highs=[40000.0] * n)
    eth_low = [1900, 1600, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
    eth = marks(
        eth_low, highs=[2000, 1900, 1600] + [1600] * 7, closes=[1950, 1700, 1520] + [1520] * 7
    )
    views = [
        view("btc1", "BTCUSDT", [Ev(bar(0), 0.01, 40000.0), Ev(bar(6), -0.01, 40000.0)]),
        view("eth1", "ETHUSDT", [Ev(bar(0), 2.0, 2000.0)]),
        view("btc2", "BTCUSDT", [Ev(bar(8), 0.01, 40000.0)]),
    ]
    tables = {"BTCUSDT": flat_table("BTCUSDT", 0.004), "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    res = scan_liquidations(
        views,
        {"BTCUSDT": btc, "ETHUSDT": eth},
        mode="cross",
        starting_balance=1000.0,
        leverage=10.0,
        tables=tables,
    )
    by_key = {h.key: h for h in res.hits}
    assert set(by_key) == {"btc1", "eth1"}
    assert res.bust_ts == bar(2)
    assert res.dropped_keys == {"btc2"}
    assert res.path_divergence is False
    assert by_key["eth1"].price == pytest.approx(3001.6 / 1.99, rel=1e-12)
    assert by_key["eth1"].mark == 1500.0  # adverse extreme at the trigger bar
    assert by_key["btc1"].price == 40000.0
    assert by_key["btc1"].realized_pnl == 0.0
    assert by_key["eth1"].margin_balance == pytest.approx(0.0, abs=1e-9)
    assert by_key["eth1"].maint_margin == pytest.approx(16.6, rel=1e-12)
    # forfeit rule: everything the account held at the bust is the loss
    assert res.total_loss == pytest.approx(1000.0, abs=1e-9)
    n3 = res.to_notes(3)
    assert n3["total_forfeited"] - n3["total_realized_pnl"] == pytest.approx(1000.0, abs=1e-9)
    assert res.to_notes(3)["bust_ts"] == bar(2)
    assert res.to_notes(3)["dropped"] == 1
    assert res.to_notes(3)["fill_rule"] == "lp_or_open_if_gapped"


def test_cross_uses_adverse_extreme_not_close():
    """Same ETH crash but every close stays at 2000: only the 1500 low busts the account.
    A close-based check (MB = 1000 > MM) would miss it."""
    eth = marks([1900, 1500], highs=[2000, 2000], closes=[2000, 2000])
    v = view("eth1", "ETHUSDT", [Ev(bar(0), 2.0, 2000.0)])
    res = scan_liquidations(
        [v],
        {"ETHUSDT": eth},
        mode="cross",
        starting_balance=1000.0,
        leverage=10.0,
        tables={"ETHUSDT": flat_table("ETHUSDT", 0.005)},
    )
    assert res.bust_ts == bar(1)
    assert res.hits[0].price == pytest.approx((1000.0 - 4000.0) / (2 * 0.005 - 2), rel=1e-12)


def test_funding_leaves_a_profitable_position_solvent():
    """Long: margin 4000 - 3900 funding = 100 is below MM at ENTRY (160) but the mark is far
    above: LP = (100 - 40000)/(.004 - 1) = 40060.24 < the 50000 mark -> no liquidation.
    Short mirror: margin 100, LP = (100 + 40000)/1.004 = 39940.24 > the 30000 mark."""
    v = long_btc(funding=[(bar(2), -3900.0)])
    assert not isolated([v], {"BTCUSDT": marks([40000, 50000, 50000, 50000])}).hits
    short = view(events=[Ev(bar(0), -1.0, 40000.0)], funding=[(bar(2), -3900.0)])
    short_mk = marks([30000] * 4, highs=[30000] * 4)
    assert not isolated([short], {"BTCUSDT": short_mk}).hits


def test_funding_that_crosses_lp_liquidates_at_the_open_of_the_first_bar():
    """Long, margin 100 after funding: LP 40060.24; the mark sits at 40000 <= LP, so the
    first bar from the settlement (bar 2) liquidates, filled at its open (rule
    min(LP, open) = 40000). pnl 0, margin balance 100 forfeited -> loss 100. Short mirror:
    LP 39940.24 < the 40000 mark -> bar 2 fills at max(LP, open) = 40000."""
    v = long_btc(funding=[(bar(2), -3900.0)])
    mk = marks([40000.0] * 4, opens=[40000.0] * 4)
    (hit,) = isolated([v], {"BTCUSDT": mk}).hits
    assert (hit.ts, hit.price, hit.insolvent) == (bar(2), 40000.0, False)
    assert hit.loss == pytest.approx(100.0, abs=1e-9)
    short = view(events=[Ev(bar(0), -1.0, 40000.0)], funding=[(bar(2), -3900.0)])
    short_mk = marks([39000.0] * 4, highs=[40000.0] * 4, opens=[40000.0] * 4)
    (hit,) = isolated([short], {"BTCUSDT": short_mk}).hits
    assert (hit.ts, hit.price) == (bar(2), 40000.0)
    assert hit.loss == pytest.approx(100.0, abs=1e-9)


def test_funding_beyond_margin_loses_only_the_margin():
    """Funding -5000 on 4000 margin: margin floors at 0, nothing left to forfeit."""
    v = long_btc(funding=[(bar(1), -5000.0)])
    (hit,) = isolated([v], {"BTCUSDT": marks([39000, 39000, 39000])}).hits
    assert hit.ts == bar(1)
    assert hit.loss == 0.0


def test_one_x_long_never_liquidates():
    """Margin = full notional -> LP <= 0 -> never, even when the mark collapses to 1."""
    assert not isolated([long_btc()], {"BTCUSDT": marks([39000, 1.0])}, leverage=1.0).hits


def test_cross_short_with_large_other_leg_loss_liquidates_at_the_bar_open():
    """Wallet 5000; BTC short 1 @ 40000 (MMR .004), ETH long 40 @ 2000 (MMR .005). Bar 1:
    BTC high 90000 (uPnL -50000, the largest adverse leg), ETH low 800 (-48000): MB < 0
    -> bust. For the BTC short the other leg holds uPnL -48000 and MM .005*40*800 = 160:
    numerator = 5000 - 160 - 48000 + 40000 = -3160 <= 0, so its cross LP is <= 0
    (tiers.InsolventPosition: insolvent at every price). It liquidates at the first
    scanned bar under the fill rule max(LP, open) = the bar open (bar 1 opens at 40000); the
    ETH leg closes at its mark close 850."""
    btc = marks([40000.0, 40000.0], highs=[40000.0, 90000.0], closes=[40000.0, 85000.0])
    eth = marks([2000.0, 800.0], highs=[2000.0, 2000.0], closes=[2000.0, 850.0])
    views = [
        view("btc", "BTCUSDT", [Ev(bar(0), -1.0, 40000.0)]),
        view("eth", "ETHUSDT", [Ev(bar(0), 40.0, 2000.0)]),
    ]
    tables = {"BTCUSDT": flat_table("BTCUSDT", 0.004), "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    res = scan_liquidations(
        views,
        {"BTCUSDT": btc, "ETHUSDT": eth},
        mode="cross",
        starting_balance=5000.0,
        leverage=10.0,
        tables=tables,
    )
    by_key = {h.key: h for h in res.hits}
    assert res.bust_ts == bar(1)
    assert by_key["btc"].price == 40000.0 and by_key["btc"].insolvent is True
    assert by_key["btc"].mark == 90000.0
    assert by_key["eth"].price == 850.0 and by_key["eth"].insolvent is False


def test_nt_exit_at_the_trigger_bar_open_wins():
    """Exit fill at bar 2's open ts: bar 2 (low 36100 < LP) is not scanned -> no hit.
    The same exit one bar later does not save the position."""
    mk = {"BTCUSDT": marks([39000, 38000, 36100, 36100])}
    assert not isolated([long_btc(exit_bar=2)], mk).hits
    assert [h.ts for h in isolated([long_btc(exit_bar=3)], mk).hits] == [bar(2)]


def test_entry_minute_is_scanned_but_the_minute_before_is_not():
    """Exposure window (entry-60s, exit): a fill at 00:00:30 is exposed to bar 0 (its open
    00:00:00 > entry - 60s), not to bar -1 (open 23:59:00)."""
    half = 30_000_000_000
    v = view(events=[Ev(bar(0) + half, 1.0, 40000.0)])
    assert isolated([v], {"BTCUSDT": marks([36000, 39000], start=0)}).hits
    assert not isolated([v], {"BTCUSDT": marks([36000, 39000, 39000], start=-1)}).hits
    # a fill exactly at a bar open: that bar is in, the previous bar is out
    exact = view(events=[Ev(bar(0), 1.0, 40000.0)])
    assert not isolated([exact], {"BTCUSDT": marks([36000, 39000, 39000], start=-1)}).hits
    assert isolated([exact], {"BTCUSDT": marks([39000, 36000], start=0)}).hits


def test_mid_minute_exit_still_scans_the_bar_opened_before_the_exit():
    """Exit at 00:01:30 (bar 1 opened at 00:01:00 < exit): bar 1's low counts (conservative);
    an exit exactly at bar 1's open excludes bar 1."""
    mk = {"BTCUSDT": marks([39000, 36000, 39000])}
    mid = view(events=[Ev(bar(0), 1.0, 40000.0), Ev(bar(1) + 30_000_000_000, -1.0, 38000.0)])
    assert [h.ts for h in isolated([mid], mk).hits] == [bar(1)]
    at_open = view(events=[Ev(bar(0), 1.0, 40000.0), Ev(bar(1), -1.0, 38000.0)])
    assert not isolated([at_open], mk).hits


def test_clearance_fee_min_of_rate_notional_and_remaining_margin():
    """MB at LP = 4000 + (LP - 40000) = 144.57831325301. Rate .01: .01*LP = 361.4 > MB ->
    fee = MB, loss 4000. Rate .001: .001*LP = 36.14457831325301 < MB -> fee 36.14..,
    loss = 3855.42168674699 + 36.14457831325301 = 3891.566265060241 (rest returned).
    Rate None: forfeit all remaining margin -> fee = MB."""
    mk = {"BTCUSDT": marks([39000, 36000])}
    mb = 4000.0 + (LP_LONG - 40000.0)
    big = isolated([long_btc()], mk, {"BTCUSDT": flat_table("BTCUSDT", 0.004, rate=0.01)})
    assert big.hits[0].clearance_fee == pytest.approx(mb, abs=1e-9)
    assert big.hits[0].loss == pytest.approx(4000.0, abs=1e-9)
    small = isolated([long_btc()], mk, {"BTCUSDT": flat_table("BTCUSDT", 0.004, rate=0.001)})
    assert small.hits[0].clearance_fee == pytest.approx(0.001 * LP_LONG, rel=1e-12)
    assert small.hits[0].loss == pytest.approx(3891.566265060241, abs=1e-8)
    assert small.to_notes(1)["clearance_rule"] == "min(rate*notional, max(margin_balance,0))"
    forfeit = isolated([long_btc()], mk, {"BTCUSDT": flat_table("BTCUSDT", 0.004, rate=None)})
    assert forfeit.hits[0].clearance_fee == pytest.approx(mb, abs=1e-9)
    assert forfeit.to_notes(1)["clearance_rule"] == "forfeit_remaining_margin"


def test_empty_inputs_and_no_hits_are_empty_with_zero_count_notes():
    res = isolated([], {})
    assert res.is_empty
    notes = res.to_notes(n_trades=0)
    assert notes["count"] == 0 and notes["pct_trades"] == 0.0
    assert (notes["total_forfeited"], notes["total_realized_pnl"]) == (0.0, 0.0)
    assert notes["events"] == [] and notes["path_divergence"] is False
    assert notes["dropped"] == 0 and notes["fill_rule"] == "lp_or_open_if_gapped"
    assert notes["gap_rule"] == "carry_forward_last_close_max_60m"
    assert "cross_fill_rule" not in notes
    quiet = isolated([long_btc()], {"BTCUSDT": marks([39000, 38000, 37000])})
    assert quiet.is_empty and quiet.to_notes(1)["count"] == 0


def test_missing_mark_series_for_a_traded_symbol_raises():
    with pytest.raises(MissingMarkSeries):
        isolated([long_btc()], {})
    late = Marks(*(a[1:] for a in marks([39000, 38000, 37000])))  # starts after the first fill
    with pytest.raises(MissingMarkSeries):
        isolated([long_btc()], {"BTCUSDT": late})
    with pytest.raises(MissingMarkSeries):
        scan_liquidations(
            [long_btc()],
            {},
            mode="cross",
            starting_balance=40000.0,
            leverage=10.0,
            tables={"BTCUSDT": _BTC},
        )


def test_unknown_symbol_raises():
    with pytest.raises(KeyError):
        isolated([long_btc()], {"BTCUSDT": marks([39000])}, tables={})


def test_isolated_scan_scales_to_a_year_of_minutes():
    """100 sequential positions over 525,600 one-minute bars scan without a per-bar loop
    (correctness: the single crossing bar is found)."""
    n = 525_600
    low = np.full(n, 39000.0)
    low[400_000] = 30000.0
    mk = Marks(
        np.arange(n, dtype=np.int64) * MIN + BASE, low, np.full(n, 41000.0), low.copy(), low.copy()
    )
    views = []
    step = n // 100
    for i in range(100):
        views.append(
            view(
                f"p{i}",
                events=[
                    Ev(bar(i * step), 1.0, 40000.0),
                    Ev(bar(i * step + step - 1), -1.0, 40000.0),
                ],
            )
        )
    res = isolated(views, {"BTCUSDT": mk})
    assert [h.ts for h in res.hits] == [bar(400_000)]
    assert res.hits[0].key == "p76"


def test_partial_reduction_releases_margin_pro_rata():
    """2 @ 100 (margin 20), sell 1 -> margin 10, q=1: LP = (10 - 100)/(.004 - 1) =
    90.36144578313252. Low 89 crosses it; without the pro-rata release the margin stays
    20, LP 80.3 and the hit is missed. Loss = the 10 remaining margin."""
    v = view(events=[Ev(bar(0), 2.0, 100.0), Ev(bar(1), -1.0, 100.0)])
    mk = {"BTCUSDT": marks([100.0, 100.0, 89.0])}
    (hit,) = isolated([v], mk, {"BTCUSDT": flat_table("BTCUSDT", 0.004)}).hits
    assert hit.ts == bar(2) and hit.qty == 1.0
    assert hit.price == pytest.approx(90.36144578313252, rel=1e-14)
    assert hit.loss == pytest.approx(10.0, abs=1e-9)


def test_scale_in_at_a_different_price_averages_the_entry():
    """1 @ 100 then 1 @ 120: avg 110, margin 22, q=2: LP = (22 - 220)/(.008 - 2) =
    99.39759036144578. Using the latest fill price instead would give 109.44 and hit one
    bar early (low 105)."""
    v = view(events=[Ev(bar(0), 1.0, 100.0), Ev(bar(1), 1.0, 120.0)])
    mk = {"BTCUSDT": marks([100.0, 120.0, 105.0, 99.0])}
    (hit,) = isolated([v], mk, {"BTCUSDT": flat_table("BTCUSDT", 0.004)}).hits
    assert hit.ts == bar(3) and hit.qty == 2.0
    assert hit.price == pytest.approx(99.39759036144578, rel=1e-14)
    assert hit.loss == pytest.approx(22.0, abs=1e-9)


def _cross(views, mk, tables, balance):
    return scan_liquidations(
        views, mk, mode="cross", starting_balance=balance, leverage=10.0, tables=tables
    )


def test_cross_wallet_includes_entry_cash():
    """Wallet 4000, entry cash -1000 (commission/other): wallet 3000. Long 1 @ 40000, low
    37000: MB = 3000 - 3000 = 0 <= MM 148 -> bust. Ignoring the cash leaves MB 1000 > 148."""
    v = view(events=[Ev(bar(0), 1.0, 40000.0, cash=-1000.0)])
    res = _cross([v], {"BTCUSDT": marks([39000.0, 37000.0])}, {"BTCUSDT": _BTC}, 4000.0)
    assert res.bust_ts == bar(1)


def test_cross_wallet_includes_realized_pnl_of_reductions():
    """4000 wallet; buy 1 @ 40000, sell 0.5 @ 38000 -> realized -1000, wallet 3000, q=.5.
    MB = 3000 + .5*(p - 40000) <= .004*.5*p  <=>  p <= 17000/.498 = 34136.5: low 34000
    busts. Without the realized loss the threshold is 32129 and 34000 survives."""
    v = view(events=[Ev(bar(0), 1.0, 40000.0), Ev(bar(1), -0.5, 38000.0)])
    res = _cross([v], {"BTCUSDT": marks([39000.0, 38500.0, 34000.0])}, {"BTCUSDT": _BTC}, 4000.0)
    assert res.bust_ts == bar(2)
    (hit,) = res.hits
    assert hit.qty == 0.5


def test_cross_short_off_bar_lp_fills_at_the_bar_open():
    """Short 1 @ 100, wallet 10, rate .001: MB = 10 + (100 - p) <= .004p  <=>  p >= 109.56
    (LP 109.5617...). Bar 1 opens at 110 (already past the LP): fill max(LP, 110) = 110,
    not the LP. Loss = 10 price loss, fee min(.001*110, max(0,0)) = 0 -> 10.0."""
    v = view(events=[Ev(bar(0), -1.0, 100.0)])
    mk = marks([100.0, 110.0], highs=[100.0, 120.0], closes=[100.0, 115.0], opens=[100.0, 110.0])
    res = _cross([v], {"BTCUSDT": mk}, {"BTCUSDT": flat_table("BTCUSDT", 0.004, 0.001)}, 10.0)
    (hit,) = res.hits
    assert hit.price == 110.0
    assert hit.loss == pytest.approx(10.0, abs=1e-12)


def test_cross_long_off_bar_lp_fills_at_the_bar_open():
    """Long 1 @ 100, wallet 10: LP = 90.36; bar 1 opens at 85 (below the LP) -> fill 85.
    Price loss 15 > wallet 10: the account stops at 0, the venue absorbs 5."""
    v = view(events=[Ev(bar(0), 1.0, 100.0)])
    mk = marks([100.0, 80.0], highs=[100.0, 90.0], closes=[100.0, 82.0], opens=[100.0, 85.0])
    res = _cross([v], {"BTCUSDT": mk}, {"BTCUSDT": flat_table("BTCUSDT", 0.004, 0.001)}, 10.0)
    (hit,) = res.hits
    assert hit.price == 85.0
    assert hit.realized_pnl == pytest.approx(-10.0, abs=1e-12)
    assert hit.insurance_deficit == pytest.approx(5.0, abs=1e-12)


def test_cross_lp_inside_the_bar_fills_at_the_lp():
    """Wallet 10, long 1 @ 100, bar opens 100 and trades down to 80: fill at LP 90.36..."""
    v = view(events=[Ev(bar(0), 1.0, 100.0)])
    mk = marks([100.0, 80.0], highs=[100.0, 100.0], closes=[100.0, 82.0], opens=[100.0, 100.0])
    res = _cross([v], {"BTCUSDT": mk}, {"BTCUSDT": flat_table("BTCUSDT", 0.004)}, 10.0)
    assert res.hits[0].price == pytest.approx(90.0 / 0.996, rel=1e-14)


def test_cross_carries_the_prior_close_across_a_missing_minute():
    """ETH has no bar at minute 1 (BTC does). At minute 1 ETH is valued at its bar-0 close
    2000 (carry-forward), not at the minute-2 low 1500: the bust is at minute 2."""
    btc = marks([40000.0] * 3, highs=[40000.0] * 3)
    eth_full = marks([2000.0, 1500.0], highs=[2000.0, 2000.0], closes=[2000.0, 1520.0])
    eth = type(eth_full)(
        *(
            a[[0, 1]] if name != "ts" else np.array([bar(0), bar(2)], dtype=np.int64)
            for name, a in zip(eth_full._fields, eth_full, strict=True)
        )
    )
    views = [
        view("btc", "BTCUSDT", [Ev(bar(0), 0.01, 40000.0)]),
        view("eth", "ETHUSDT", [Ev(bar(0), 2.0, 2000.0)]),
    ]
    tables = {"BTCUSDT": flat_table("BTCUSDT", 0.004), "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    res = _cross(views, {"BTCUSDT": btc, "ETHUSDT": eth}, tables, 1000.0)
    assert res.bust_ts == bar(2)


def test_mark_gap_beyond_60_minutes_raises_but_short_gaps_are_accepted():
    ts = np.array([bar(0), bar(1), bar(62)], dtype=np.int64)
    low = np.array([39000.0, 39000.0, 39000.0])
    wide = Marks(ts, low, np.full(3, 41000.0), low.copy(), low.copy())
    with pytest.raises(MissingMarkSeries, match="gap"):
        isolated([long_btc(exit_bar=62)], {"BTCUSDT": wide})
    ok_ts = np.array([bar(0), bar(1), bar(60)], dtype=np.int64)
    ok = Marks(ok_ts, low, np.full(3, 41000.0), low.copy(), low.copy())
    assert isolated([long_btc(exit_bar=60)], {"BTCUSDT": ok}).is_empty


def test_nonfinite_adverse_marks_raise_instead_of_returning_no_hit():
    """A NaN (or inf, same ``isfinite`` guard) in the extreme the trigger reads (low for
    longs, high for shorts) used to compare False and silently report no liquidation."""
    low = marks([39000.0, float("nan"), 39000.0])
    with pytest.raises(ValueError, match="BTCUSDT"):
        isolated([long_btc()], {"BTCUSDT": low})
    short = view(events=[Ev(bar(0), -1.0, 40000.0)])
    high = marks([39000.0] * 3, highs=[41000.0, float("inf"), 41000.0])
    with pytest.raises(ValueError, match="BTCUSDT"):
        isolated([short], {"BTCUSDT": high})


@pytest.mark.parametrize("field", ["high", "close"])
def test_cross_rejects_any_nonfinite_mark_field(field):
    """One row per check site: the open/high/low tuple loop and the carried-close call."""
    mk = marks([39000.0, 39000.0, 39000.0])
    mk = mk._replace(**{field: np.array([np.nan, 39000.0, 39000.0])})
    with pytest.raises(ValueError, match="BTCUSDT"):
        _cross([long_btc()], {"BTCUSDT": mk}, {"BTCUSDT": _BTC}, 40000.0)


def test_nonfinite_close_of_a_non_triggering_leg_raises_instead_of_poisoning_the_loss():
    """ETH leg's close is inf while BTC busts the account: the hit price/fee/loss would be
    inf (invalid strict JSON). The scan rejects the series instead."""
    btc = marks([40000.0, 1000.0], highs=[40000.0, 40000.0])
    eth = marks([2000.0, 2000.0], highs=[2000.0, 2000.0], closes=[2000.0, np.inf])
    views = [
        view("btc", "BTCUSDT", [Ev(bar(0), 1.0, 40000.0)]),
        view("eth", "ETHUSDT", [Ev(bar(0), 0.1, 2000.0)]),
    ]
    tables = {"BTCUSDT": _BTC, "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    with pytest.raises(ValueError, match="ETHUSDT"):
        _cross(views, {"BTCUSDT": btc, "ETHUSDT": eth}, tables, 4000.0)


@pytest.mark.parametrize(
    "bad_view",
    [
        view(events=[Ev(bar(0), 1.0, 40000.0, cash=float("nan"))]),
        view(events=[Ev(bar(0), 1.0, float("nan"))]),
        view(events=[Ev(bar(0), float("inf"), 40000.0)]),
        view(events=[Ev(bar(0), 1.0, 40000.0)], funding=[(bar(1), float("nan"))]),
    ],
)
def test_nonfinite_events_and_funding_raise(bad_view):
    """One row per check in ``_validate_view`` (fill cash, fill price, fill qty, funding);
    it runs before the mode dispatch, so one mode covers both."""
    with pytest.raises(ValueError, match="BTCUSDT"):
        scan_liquidations(
            [bad_view],
            {"BTCUSDT": marks([39000.0] * 3)},
            mode="isolated",
            starting_balance=40000.0,
            leverage=10.0,
            tables={"BTCUSDT": _BTC},
        )


def test_series_without_a_bar_open_raises():
    """The fill rule needs the real bar open; a series without one fails closed instead of
    guessing it from the previous close (which understated a gap-through loss 50 -> 10).
    ``_require_marks`` runs before the mode dispatch, so one mode covers both."""
    mk = marks([39000.0, 36000.0])
    no_open = MarksNoOpen(mk.ts, mk.low, mk.high, mk.close)
    with pytest.raises(MissingMarkSeries, match="open"):
        scan_liquidations(
            [long_btc()],
            {"BTCUSDT": no_open},
            mode="isolated",
            starting_balance=40000.0,
            leverage=10.0,
            tables={"BTCUSDT": _BTC},
        )


# Events inside an allowed mark gap (no bar in the event's minute) are evaluated AT the
# event ts against the carried-forward last close; fill = min(LP, close) long /
# max(LP, close) short. Bars at minutes 0 and 2 only; minute 1 is the gap.


def _gap_marks(close0: float, close2: float) -> Marks:
    mk = marks(
        [min(close0, 39000.0), close2],
        highs=[max(close0, 41000.0), close2],
        closes=[close0, close2],
        opens=[close0, close2],
    )
    return at([bar(0), bar(2)], mk)


@pytest.mark.parametrize("mode", ["isolated", "cross"])
def test_funding_inside_a_mark_gap_liquidates_at_the_event_on_the_carried_close(mode):
    """Reviewer case: long 1 @ 40000, margin (isolated) / wallet (cross) 4000, funding
    -3900 inside the gap -> 100 left, LP = (100 - 40000)/(.004 - 1) = 40060.24. The carried
    close 40000 is already past it; the next bar (50000) is not. Liquidated at the event
    ts, fill min(LP, 40000) = 40000, pnl 0, forfeit 100 -> loss 100. Control: a carried
    close 50000 (above the LP) is solvent."""
    v = long_btc(funding=[(bar(1), -3900.0)])
    res = scan_liquidations(
        [v],
        {"BTCUSDT": _gap_marks(40000.0, 50000.0)},
        mode=mode,
        starting_balance=4000.0,
        leverage=10.0,
        tables={"BTCUSDT": _BTC},
    )
    (hit,) = res.hits
    assert (hit.ts, hit.price, hit.mark) == (bar(1), 40000.0, 40000.0)
    assert hit.realized_pnl == 0.0
    assert hit.loss == pytest.approx(100.0, abs=1e-9)
    if mode == "cross":
        assert res.bust_ts == bar(1)
    assert not scan_liquidations(
        [long_btc(funding=[(bar(1), -3900.0)])],
        {"BTCUSDT": _gap_marks(50000.0, 50000.0)},
        mode=mode,
        starting_balance=4000.0,
        leverage=10.0,
        tables={"BTCUSDT": _BTC},
    ).hits


def test_short_funding_inside_a_mark_gap_fills_at_the_carried_close():
    """Short mirror: margin 100 after funding, LP = (100 + 40000)/1.004 = 39940.24 <= the
    carried close 40000 (next bar 30000 is far below): fill max(LP, 40000) = 40000."""
    v = view(events=[Ev(bar(0), -1.0, 40000.0)], funding=[(bar(1), -3900.0)])
    mk = at(
        [bar(0), bar(2)],
        marks(
            [39000.0, 30000.0],
            highs=[40000.0, 30000.0],
            opens=[40000.0, 30000.0],
            closes=[40000.0, 30000.0],
        ),
    )
    (hit,) = isolated([v], {"BTCUSDT": mk}).hits
    assert (hit.ts, hit.price, hit.side) == (bar(1), 40000.0, -1)
    assert hit.loss == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("mode", ["isolated", "cross"])
def test_entry_inside_a_mark_gap_liquidates_at_the_entry_on_the_carried_close(mode):
    """Reviewer case: entry long 1 @ 100 inside the gap; carried close 90, next bar 100.
    Margin/wallet 10: LP = (10 - 100)/(.004 - 1) = 90.36144578313252 >= 90 -> liquidated
    at the entry ts, fill min(LP, 90) = 90, loss 10 (isolated floor = margin; cross MB 0)."""
    v = view(events=[Ev(bar(1), 1.0, 100.0)])
    mk = at(
        [bar(0), bar(2)],
        marks([89.0, 100.0], highs=[91.0, 101.0], closes=[90.0, 100.0], opens=[90.0, 100.0]),
    )
    res = scan_liquidations(
        [v],
        {"BTCUSDT": mk},
        mode=mode,
        starting_balance=10.0,
        leverage=10.0,
        tables={"BTCUSDT": _BTC},
    )
    (hit,) = res.hits
    assert (hit.ts, hit.price) == (bar(1), 90.0)
    assert hit.loss == pytest.approx(10.0, abs=1e-12)


def test_cross_event_in_one_legs_gap_values_the_other_leg_at_its_bar():
    """Two legs, ETH has no bar at minute 1, BTC does (so minute 1 is a grid minute): the
    funding at minute 1 is evaluated there with ETH at its carried close 2000 -> bust at
    minute 1 even though ETH's next bar recovers to 3000."""
    btc = marks([40000.0] * 3, highs=[40000.0] * 3)
    eth = at(
        [bar(0), bar(2)], marks([2000.0, 3000.0], highs=[2000.0, 3000.0], closes=[2000.0, 3000.0])
    )
    views = [
        view("btc", "BTCUSDT", [Ev(bar(0), 0.01, 40000.0)]),
        view("eth", "ETHUSDT", [Ev(bar(0), 2.0, 2000.0)], funding=[(bar(1), -990.0)]),
    ]
    tables = {"BTCUSDT": flat_table("BTCUSDT", 0.004), "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    res = _cross(views, {"BTCUSDT": btc, "ETHUSDT": eth}, tables, 1000.0)
    assert res.bust_ts == bar(1)


def test_long_mark_gap_split_by_events_still_raises():
    """Bars at minutes 0 and 100 only (a 100-minute hole). Funding at 30/59/89 splits it into
    segments shorter than 60 minutes each; the hole must still fail closed. ``_check_gaps``
    is shared by both modes."""
    v = view(
        events=[Ev(bar(0), 0.1, 40000.0), Ev(bar(100), -0.1, 40000.0)],
        funding=[(bar(30), -1.0), (bar(59), -1.0), (bar(89), -1.0)],
    )
    mk = at([bar(0), bar(100)], marks([39000.0, 39000.0]))
    with pytest.raises(MissingMarkSeries, match="gap"):
        scan_liquidations(
            [v],
            {"BTCUSDT": mk},
            mode="isolated",
            starting_balance=40000.0,
            leverage=10.0,
            tables={"BTCUSDT": _BTC},
        )


def test_cross_gap_through_the_lp_fills_at_the_real_open_and_floors_the_account():
    """Reviewer open cases, wallet 10, rate .001. Long 1 @ 100 (LP 90.36), bar 1 OPENS at 50:
    fill 50 (price loss 50). Short 1 @ 100 (LP 109.56), bar 1 opens at 120: fill 120 (price
    loss 20). The account cannot go below 0 (chief): loss 10 = the wallet, the rest is the
    venue's insurance deficit (40 / 10). The removed previous-close stand-in (100) filled
    both at the LP and hid the gap."""
    table = {"BTCUSDT": flat_table("BTCUSDT", 0.004, 0.001)}
    long_mk = marks([100.0, 45.0], highs=[100.0, 60.0], closes=[100.0, 55.0], opens=[100.0, 50.0])
    res = _cross([view(events=[Ev(bar(0), 1.0, 100.0)])], {"BTCUSDT": long_mk}, table, 10.0)
    (hit,) = res.hits
    assert (hit.price, hit.realized_pnl, hit.insurance_deficit) == (50.0, -10.0, 40.0)
    assert hit.loss == 10.0 and hit.clearance_fee == 0.0
    notes = res.to_notes(1)
    assert (notes["total_forfeited"], notes["total_realized_pnl"]) == (0.0, -10.0)
    assert notes["total_insurance_deficit"] == 40.0
    assert notes["events"][0]["insurance_deficit"] == 40.0
    short_mk = marks(
        [100.0, 115.0], highs=[100.0, 125.0], closes=[100.0, 121.0], opens=[100.0, 120.0]
    )
    (hit,) = _cross(
        [view(events=[Ev(bar(0), -1.0, 100.0)])], {"BTCUSDT": short_mk}, table, 10.0
    ).hits
    assert (hit.price, hit.realized_pnl, hit.insurance_deficit) == (120.0, -10.0, 10.0)
    assert hit.loss == 10.0


def test_cross_shortfall_is_shared_by_the_losing_legs_pro_rata():
    """Wallet 10; BTC long 1 @ 100 (MMR .004) and ETH long 1 @ 100 (MMR .005) both gap
    down at bar 1 (BTC open 80 low 75 close 78; ETH open 70 low 65 close 66). ETH is the
    trigger leg: LP = (10 - .3 - 25 - 100)/(.005 - 1) = 115.88 -> fill min(LP, 70) = 70;
    BTC closes at 78. Price losses 30 + 22 = 52 leave the account at -42: the venue absorbs
    42 split 30:22, so the account loss is exactly the 10 wallet."""
    btc = marks([100.0, 75.0], highs=[100.0, 80.0], closes=[100.0, 78.0], opens=[100.0, 80.0])
    eth = marks([100.0, 65.0], highs=[100.0, 70.0], closes=[100.0, 66.0], opens=[100.0, 70.0])
    views = [
        view("btc", "BTCUSDT", [Ev(bar(0), 1.0, 100.0)]),
        view("eth", "ETHUSDT", [Ev(bar(0), 1.0, 100.0)]),
    ]
    tables = {"BTCUSDT": flat_table("BTCUSDT", 0.004), "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    res = _cross(views, {"BTCUSDT": btc, "ETHUSDT": eth}, tables, 10.0)
    by_key = {h.key: h for h in res.hits}
    assert (by_key["eth"].price, by_key["btc"].price) == (70.0, 78.0)
    assert by_key["eth"].insurance_deficit == pytest.approx(42.0 * 30.0 / 52.0, rel=1e-14)
    assert by_key["btc"].insurance_deficit == pytest.approx(42.0 * 22.0 / 52.0, rel=1e-14)
    assert by_key["eth"].realized_pnl == pytest.approx(-30.0 * 10.0 / 52.0, rel=1e-14)
    assert res.total_loss == pytest.approx(10.0, abs=1e-12)
    assert res.to_notes(2)["total_insurance_deficit"] == pytest.approx(42.0, abs=1e-12)


# A mid-minute event inside an EXISTING bar applies its post-event state to that bar
# (chief rule F2): the containing bar is scanned under the post-event LP and a hit is
# timestamped at the event, filled per the open rule.
MID_BAR1 = bar(1) + 30_000_000_000  # 1700000070000000000


@pytest.mark.parametrize("mode", ["isolated", "cross"])
def test_mid_minute_funding_inside_an_existing_bar_liquidates_at_the_event(mode):
    """Reviewer case: long 1 @ 40000, margin/wallet 4000; bars 0/1 flat 40000, bar 2 flat
    50000; funding -3900 at minute 1 + 30 s leaves 100: LP = (100 - 40000)/(.004 - 1) =
    40060.24. Bar 1 (low 40000) contains the event and crosses the new LP: hit at the
    event ts, fill min(LP, open 40000) = 40000, pnl 0, forfeit 100 -> loss 100."""
    assert MID_BAR1 == 1700000070000000000
    flat = [40000.0, 40000.0, 50000.0]
    mk = marks(flat, highs=flat, closes=flat, opens=flat)
    res = scan_liquidations(
        [long_btc(funding=[(MID_BAR1, -3900.0)])],
        {"BTCUSDT": mk},
        mode=mode,
        starting_balance=4000.0,
        leverage=10.0,
        tables={"BTCUSDT": _BTC},
    )
    (hit,) = res.hits
    assert (hit.ts, hit.price) == (MID_BAR1, 40000.0)
    assert hit.realized_pnl == 0.0
    assert hit.loss == pytest.approx(100.0, abs=1e-9)
    if mode == "cross":
        assert res.bust_ts == MID_BAR1


def test_mid_minute_entry_hit_is_stamped_at_the_entry_not_before_it():
    """An entry at 00:00:30 is exposed to bar 0 (rule A); a hit there is stamped at the
    entry ts (the overlay keeps fills with ts <= hit ts), never at the bar open before it.
    The cross grid carries the same instant (``_eval_cross_segment``'s event ts); the cross
    in-bar value is pinned by the mid-minute event test above."""
    entry = bar(0) + 30_000_000_000
    res = scan_liquidations(
        [view(events=[Ev(entry, 1.0, 40000.0)])],
        {"BTCUSDT": marks([36000.0, 39000.0], opens=[40000.0, 39000.0])},
        mode="isolated",
        starting_balance=4000.0,
        leverage=10.0,
        tables={"BTCUSDT": _BTC},
    )
    assert [h.ts for h in res.hits] == [entry]


def _signed_wallet_case(open2: float, low2: float):
    """Long 1 @ 100, wallet 10, funding -20 at bar 1 while the mark is 150: wallet -10,
    margin balance -10 + 50 = 40 > MM .6 (solvent). Bar 2 opens ``open2``, low ``low2``."""
    v = view(events=[Ev(bar(0), 1.0, 100.0)], funding=[(bar(1), -20.0)])
    mk = marks(
        [100.0, 150.0, low2],
        highs=[100.0, 150.0, open2],
        closes=[100.0, 150.0, low2],
        opens=[100.0, 150.0, open2],
    )
    return _cross([v], {"BTCUSDT": mk}, {"BTCUSDT": flat_table("BTCUSDT", 0.004)}, 10.0)


def test_cross_negative_wallet_lp_uses_the_signed_wallet():
    """Reviewer B2: LP with the ACTUAL wallet -10 = (-10 - 100)/(.004 - 1) =
    110.4417670682731 (clamping the wallet to 0 gave 100.40160642570281 and left the account
    at -9.6). Bar 2 opens 115, low 109: fill at the LP, realized 10.441767068273094, forfeit
    fee 0.4417670682730943, equity -10 + realized - fee = 0, no insurance deficit."""
    res = _signed_wallet_case(115.0, 109.0)
    (hit,) = res.hits
    assert res.bust_ts == bar(2)
    assert hit.price == 110.4417670682731
    assert hit.realized_pnl == 10.441767068273094
    assert hit.clearance_fee == 0.4417670682730943
    assert hit.insurance_deficit == 0.0
    assert -10.0 + hit.realized_pnl - hit.clearance_fee == pytest.approx(0.0, abs=1e-12)


def test_cross_lp_accepts_a_negative_wallet():
    """tiers.cross_liq_price takes the signed wallet: (-10 - 100)/(.004 - 1)."""
    lp = cross_liq_price(1, 1.0, 100.0, -10.0, 0.0, 0.0, flat_table("BTCUSDT", 0.004))
    assert lp == 110.4417670682731


def test_cross_negative_wallet_gap_charges_the_account_level_deficit():
    """Reviewer B2 gap: bar 2 opens 109 (below the LP 110.44): fill 109, price PnL +9, the
    account is left at -10 + 9 = -1. Every leg is profitable, so a per-losing-leg cap
    absorbed nothing; the account-level deficit is 1 (equity floored at 0)."""
    res = _signed_wallet_case(109.0, 109.0)
    (hit,) = res.hits
    assert hit.price == 109.0
    assert hit.insurance_deficit == pytest.approx(1.0, abs=1e-12)
    assert hit.realized_pnl == pytest.approx(10.0, abs=1e-12)
    assert hit.clearance_fee == 0.0
    assert -10.0 + hit.realized_pnl - hit.clearance_fee == pytest.approx(0.0, abs=1e-12)
    assert res.to_notes(1)["total_insurance_deficit"] == pytest.approx(1.0, abs=1e-12)


def test_cross_leg_without_marks_for_more_than_60_minutes_raises():
    """Reviewer B3: BTC has only bar 0, ETH flat bars 0..100, both open with no later
    event. Carrying BTC's close for 100 minutes is a hole > 60 min: fail closed. With ETH
    bars 0..60 the carry is exactly 60 minutes (allowed)."""
    btc = marks([40000.0], highs=[40000.0])
    views = [
        view("btc", "BTCUSDT", [Ev(bar(0), 0.01, 40000.0)]),
        view("eth", "ETHUSDT", [Ev(bar(0), 1.0, 2000.0)]),
    ]
    tables = {"BTCUSDT": flat_table("BTCUSDT", 0.004), "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    eth = marks([2000.0] * 101, highs=[2000.0] * 101)
    with pytest.raises(MissingMarkSeries, match="BTCUSDT"):
        _cross(views, {"BTCUSDT": btc, "ETHUSDT": eth}, tables, 1000.0)
    eth60 = marks([2000.0] * 61, highs=[2000.0] * 61)
    assert _cross(views, {"BTCUSDT": btc, "ETHUSDT": eth60}, tables, 1000.0).is_empty


def test_nan_open_in_a_scanned_window_without_a_hit_raises():
    """The cross read validates every bar of the segment; the isolated field loop is pinned
    by ``test_isolated_scan_rejects_nonfinite_in_any_ohlc_field_of_a_scanned_bar``."""
    mk = marks([39000.0] * 3, opens=[40000.0, np.nan, 39000.0])
    with pytest.raises(ValueError, match="BTCUSDT"):
        scan_liquidations(
            [long_btc()],
            {"BTCUSDT": mk},
            mode="cross",
            starting_balance=40000.0,
            leverage=10.0,
            tables={"BTCUSDT": _BTC},
        )


def test_isolated_one_x_long_still_checks_mark_coverage():
    """LP <= 0 (never liquidates) must not skip the gap check: bars 0/100, exit at 100."""
    mk = at([bar(0), bar(100)], marks([39000.0, 39000.0]))
    with pytest.raises(MissingMarkSeries, match="gap"):
        isolated([long_btc(exit_bar=100)], {"BTCUSDT": mk}, leverage=1.0)


def test_cross_mid_minute_event_values_an_in_bar_leg_at_its_open_not_its_future_close():
    """Mixed gap at minute 1.5: BTC bar 1 opens 100 (low 100) and closes 150; ETH has no
    bar at minute 1 (carried 100). Wallet 10, funding -9.5 at 1.5 leaves 0.5 <= MM .4 + .5:
    bust at the event, loss 0.5, both legs fill at 100. Both uPnLs are 0, so the first leg
    (BTC) triggers: BTC fills min(LP 100.40, open 100) = 100 (fill-rule open), ETH at its
    carried 100. Valuing BTC at its future close 150 moves the fill."""
    btc = marks(
        [100.0, 100.0, 150.0],
        highs=[100.0, 150.0, 150.0],
        closes=[100.0, 150.0, 150.0],
        opens=[100.0, 100.0, 150.0],
    )
    eth = at([bar(0), bar(2)], marks([100.0, 100.0], highs=[100.0, 100.0]))
    views = [
        view("btc", "BTCUSDT", [Ev(bar(0), 1.0, 100.0)], funding=[(MID_BAR1, -9.5)]),
        view("eth", "ETHUSDT", [Ev(bar(0), 1.0, 100.0)]),
    ]
    tables = {"BTCUSDT": flat_table("BTCUSDT", 0.004), "ETHUSDT": flat_table("ETHUSDT", 0.005)}
    res = _cross(views, {"BTCUSDT": btc, "ETHUSDT": eth}, tables, 10.0)
    by_key = {h.key: h for h in res.hits}
    assert res.bust_ts == MID_BAR1
    assert by_key["btc"].clearance_fee == pytest.approx(0.5, abs=1e-12)
    assert (by_key["btc"].price, by_key["eth"].price) == (100.0, 100.0)
    assert res.total_loss == pytest.approx(0.5, abs=1e-12)


def test_notes_report_forfeited_realized_and_deficit_sums_not_a_total_loss():
    """Reviewer SF1 (chief ruling): notes carry three explicit sums instead of the
    ambiguous ``total_loss`` (which read -10 for an account bust that ends at equity 0).
    total_forfeited = sum(clearance_fee) >= 0; total_realized_pnl = sum(realized_pnl),
    signed (price PnL at the fills plus the deficit share); total_insurance_deficit. The
    signed ``hit.loss`` the overlay books is unchanged. Each event repeats the three."""
    res = _signed_wallet_case(115.0, 109.0)
    notes = res.to_notes(1)
    assert "total_loss" not in notes
    assert notes["total_forfeited"] == 0.4417670682730943
    assert notes["total_realized_pnl"] == 10.441767068273094
    assert notes["total_insurance_deficit"] == 0.0
    (ev,) = notes["events"]
    assert (ev["forfeited"], ev["realized_pnl"], ev["insurance_deficit"]) == (
        0.4417670682730943,
        10.441767068273094,
        0.0,
    )
    assert res.hits[0].loss == pytest.approx(0.4417670682730943 - 10.441767068273094, abs=1e-12)
    assert res.total_loss == res.hits[0].loss
    # post-bust equity: wallet -10 + realized - fee = 0
    assert -10.0 + notes["total_realized_pnl"] - notes["total_forfeited"] == pytest.approx(
        0.0, abs=1e-12
    )

    gap = _signed_wallet_case(109.0, 109.0).to_notes(1)
    assert "total_loss" not in gap
    assert gap["total_forfeited"] == 0.0
    assert gap["total_realized_pnl"] == pytest.approx(10.0, abs=1e-12)
    assert gap["total_insurance_deficit"] == pytest.approx(1.0, abs=1e-12)
    (gev,) = gap["events"]
    assert gev["forfeited"] == 0.0
    assert gev["realized_pnl"] == pytest.approx(10.0, abs=1e-12)
    assert gev["insurance_deficit"] == pytest.approx(1.0, abs=1e-12)


def _flat100(n=3, **override):
    flat = [100.0] * n
    mk = marks(flat, highs=flat, closes=flat, opens=flat)
    return mk._replace(**{k: np.array(v, dtype=np.float64) for k, v in override.items()})


@pytest.mark.parametrize("field", ["open", "low", "high", "close"])
def test_isolated_scan_rejects_nonfinite_in_any_ohlc_field_of_a_scanned_bar(field):
    """Reviewer SF2: q=1 EP100 10x on flat-100 bars; the isolated trigger reads only the
    adverse extreme and open, so a NaN high/close (long) or low/close (short) in bar 1
    returned zero hits. Every OHLC field of a scanned bar must be finite: one row per field
    (a dropped guard for one field is otherwise unobserved), one value (nan and inf are the
    same ``isfinite`` check), one side (the validation is shared)."""
    col = [100.0, float("nan"), 100.0]
    v = view(events=[Ev(bar(0), 1.0, 100.0)])
    with pytest.raises(ValueError, match="BTCUSDT"):
        isolated(
            [v], {"BTCUSDT": _flat100(**{field: col})}, {"BTCUSDT": flat_table("BTCUSDT", 0.004)}
        )


@pytest.mark.parametrize("field", ["open", "close"])
def test_cross_rejects_nonfinite_field_of_the_bar_containing_a_mid_minute_entry_and_exit(
    field,
):
    """Reviewer SF2: entry at minute 0 + 30 s, exit at + 45 s, both inside bar 0. The leg's
    exit price is its observed OPEN (the close is still in the future), so a NaN close of
    the containing bar was replaced by the open before validation and accepted. The two
    inputs of that substitution: the raw close and the open it is replaced with."""
    v = view(
        events=[
            Ev(bar(0) + 30_000_000_000, 1.0, 100.0),
            Ev(bar(0) + 45_000_000_000, -1.0, 100.0),
        ]
    )
    mk = _flat100(**{field: [np.nan, 100.0, 100.0]})
    with pytest.raises(ValueError, match="BTCUSDT"):
        _cross([v], {"BTCUSDT": mk}, {"BTCUSDT": flat_table("BTCUSDT", 0.004)}, 10.0)
