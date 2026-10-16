"""Zero-tolerance tests for Binance USDT-M liquidation math (bead vibe-quant-2gyk1.1.3).

Vectors are hand-computed by the chief; floats are asserted exactly (==) where
the literal round-trips, otherwise to a stated tolerance.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from vibe_quant.liquidation.tiers import (
    Bracket,
    InsolventPosition,
    TierTable,
    TierTableError,
    TierTableExceeded,
    UnknownSymbol,
    cross_liq_price,
    isolated_liq_price,
    load_tier_tables,
)

if TYPE_CHECKING:
    from pathlib import Path

_TABLES = load_tier_tables()
_BTC = _TABLES["BTCUSDT"]


def _table(symbol: str, brackets: list[tuple[float, float, float, float, int | None]]) -> TierTable:
    return TierTable(
        symbol=symbol,
        brackets=tuple(Bracket(*bracket) for bracket in brackets),
        clearance_fee_rate=None,
    )


def _bracket(floor: float, cap: float, mmr: float, cum: float, lev: int | None = 125) -> dict:
    return {"floor": floor, "cap": cap, "mmr": mmr, "cum": cum, "max_leverage": lev}


def _symbol(brackets: list[dict], fee: float | None = None) -> dict:
    return {
        "brackets": brackets,
        "clearance_fee_rate": fee,
        "clearance_fee_source": "test",
    }


def _snapshot(tmp_path: Path, symbols: dict, version: str = "2026-10-12") -> Path:
    payload = {
        "version": version,
        "fetched_at": "2026-10-12T00:00:00+00:00",
        "source": "test",
        "note": "test",
        "symbols": symbols,
    }
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestIsolatedVectors:
    def test_long_exact(self):
        lp = isolated_liq_price(1, 1.0, 40000.0, 4000.0, _BTC)
        assert lp == 36144.57831325301

    def test_short_exact(self):
        # 44000 / 1.004 = 11000000/251 = 43824.7011952191235...; the correctly
        # rounded float64 is 43824.701195219124 (0x1.566167030f4c8p+15). The
        # brief's hand literal 43824.70119521912 (0x1.566167030f4c7p+15) was one
        # ulp low (chief confirmed the typo 2026-10-12), so the exact vector is
        # the correctly-rounded double.
        lp = isolated_liq_price(-1, 1.0, 40000.0, 4000.0, _BTC)
        assert lp == 43824.701195219124

    def test_margin_balance_equals_maintenance_at_lp(self):
        lp = isolated_liq_price(1, 1.0, 40000.0, 4000.0, _BTC)
        margin_balance = 4000.0 + (lp - 40000.0) * 1.0
        maintenance = _BTC.maintenance_margin(lp * 1.0)
        assert abs(margin_balance - maintenance) <= 1e-9

    def test_lp_independent_of_qty_when_margin_scales(self):
        big = isolated_liq_price(1, 1.0, 40000.0, 4000.0, _BTC)
        small = isolated_liq_price(1, 0.0125, 40000.0, 50.0, _BTC)
        assert small == 36144.57831325301
        assert small == big

    def test_one_x_long_is_never_liquidated(self):
        assert isolated_liq_price(1, 1.0, 40000.0, 40000.0, _BTC) <= 0.0


class TestCrossVector:
    def test_binance_published_example(self):
        table = _table("BINANCE_EXAMPLE", [(0.0, 1e12, 0.025, 16300.0, None)])
        lp = cross_liq_price(1, 109.488, 32481.98, 1535443.01, 356512.508, -448192.89, table)
        assert abs(lp - 26316.89) <= 0.01


class TestTierBoundary:
    def test_lp_notional_selects_tier_two(self):
        """Fixture: tier1 [0,1000) mmr .01 cum 0; tier2 [1000,100000) mmr .02 cum 10
        (continuity: 10 = 0 + 1000*(.02 - .01)).

        Long qty 1, entry 2000, isolated margin 500:
          tier1 pricing: (500 + 0 - 2000)/(.01 - 1) = 1515.1515151515152 -> notional
                         1515 lands in tier2, so tier1 is self-inconsistent;
          tier2 pricing: (500 + 10 - 2000)/(.02 - 1) = -1490/-0.98 = 1520.4081632653063
                         -> notional 1520 lands in [1000,100000), self-consistent.
        """
        table = _table(
            "TB",
            [(0.0, 1000.0, 0.01, 0.0, None), (1000.0, 100000.0, 0.02, 10.0, None)],
        )
        lp = isolated_liq_price(1, 1.0, 2000.0, 500.0, table)
        assert lp == 1520.4081632653063
        assert lp != 1515.1515151515152
        assert table.bracket_for(1.0 * lp).mmr == 0.02

    def test_entry_notional_does_not_select_bracket(self):
        """Same fixture. Long qty 1, entry 2000 (ENTRY notional 2000 -> tier2),
        isolated margin 1500. The bracket follows the LP notional, not the entry
        notional:
          tier1 pricing: (1500 + 0 - 2000)/(.01 - 1) = -500/-0.99 = 505.0505050505051
                         -> notional 505 in tier1, self-consistent;
          tier2 pricing: (1500 + 10 - 2000)/(.02 - 1) = -490/-0.98 = 500.0
                         -> notional 500 in tier1, so tier2 is rejected.
        Selecting by entry notional would instead return 500.0.
        """
        table = _table(
            "TB",
            [(0.0, 1000.0, 0.01, 0.0, None), (1000.0, 100000.0, 0.02, 10.0, None)],
        )
        lp = isolated_liq_price(1, 1.0, 2000.0, 1500.0, table)
        assert lp == 505.0505050505051
        assert lp != 500.0


class TestMaintenanceMargin:
    def test_tier_one(self):
        assert _BTC.maintenance_margin(100000.0) == 100000.0 * 0.004

    def test_tier_two_floor_is_inclusive(self):
        bracket = _BTC.bracket_for(300000.0)
        assert bracket.mmr == 0.005
        assert _BTC.maintenance_margin(300000.0) == 300000.0 * 0.005 - 300.0

    def test_notional_at_or_above_last_cap_raises(self):
        with pytest.raises(TierTableExceeded):
            _BTC.bracket_for(_BTC.brackets[-1].cap)


class TestLoaderValidation:
    def test_cum_discontinuity_raises(self, tmp_path):
        path = _snapshot(
            tmp_path,
            {
                "X": _symbol(
                    [_bracket(0.0, 1000.0, 0.01, 0.0), _bracket(1000.0, 10000.0, 0.02, 999.0)]
                )
            },
        )
        with pytest.raises(TierTableError):
            load_tier_tables(path)

    def test_contiguous_brackets_are_accepted(self, tmp_path):
        path = _snapshot(
            tmp_path,
            {
                "X": _symbol(
                    [_bracket(0.0, 1000.0, 0.01, 0.0), _bracket(1000.0, 10000.0, 0.02, 10.0)]
                )
            },
        )
        tables = load_tier_tables(path)
        assert tables["X"].brackets[1].cum == 10.0

    def test_overlapping_brackets_raise(self, tmp_path):
        path = _snapshot(
            tmp_path,
            {
                "X": _symbol(
                    [_bracket(0.0, 1000.0, 0.01, 0.0), _bracket(900.0, 10000.0, 0.02, 10.0)]
                )
            },
        )
        with pytest.raises(TierTableError):
            load_tier_tables(path)

    def test_missing_field_raises(self, tmp_path):
        path = _snapshot(
            tmp_path,
            {"X": _symbol([{"cap": 1000.0, "mmr": 0.01, "cum": 0.0, "max_leverage": 125}])},
        )
        with pytest.raises(TierTableError):
            load_tier_tables(path)

    def test_unknown_symbol_raises(self):
        assert issubclass(UnknownSymbol, KeyError)
        with pytest.raises(UnknownSymbol):
            _TABLES["FAKEUSDT"]


class TestShippedSnapshot:
    def test_symbol_set(self):
        assert set(_TABLES) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}

    def test_metadata(self):
        assert isinstance(_TABLES.version, str) and _TABLES.version
        assert isinstance(_TABLES.fetched_at, str) and _TABLES.fetched_at

    def test_shipped_brackets_pass_min_width(self):
        tables = load_tier_tables()
        assert set(tables) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
        for symbol, table in tables.items():
            assert min(bracket.cap - bracket.floor for bracket in table.brackets) >= 10000, symbol

    def test_btc_tier_one(self):
        bracket = _BTC.brackets[0]
        assert (bracket.floor, bracket.cap, bracket.mmr, bracket.cum) == (
            0.0,
            300000.0,
            0.004,
            0.0,
        )

    @pytest.mark.parametrize(
        ("symbol", "tier_one"),
        [
            ("ETHUSDT", (0.0, 300000.0, 0.004, 0.0)),
            ("SOLUSDT", (0.0, 50000.0, 0.005, 0.0)),
            ("BNBUSDT", (0.0, 10000.0, 0.005, 0.0)),
        ],
    )
    def test_other_symbols_tier_one(self, symbol, tier_one):
        bracket = _TABLES[symbol].brackets[0]
        assert (bracket.floor, bracket.cap, bracket.mmr, bracket.cum) == tier_one


class TestInputValidation:
    @pytest.mark.parametrize(
        ("side", "qty", "entry_price"),
        [
            (0, 1.0, 40000.0),
            (2, 1.0, 40000.0),
            (1, 0.0, 40000.0),
            (1, -1.0, 40000.0),
            (1, 1.0, 0.0),
            (1, 1.0, -40000.0),
            (1, math.inf, 40000.0),
            (1, math.nan, 40000.0),
            (1, 1.0, math.inf),
        ],
    )
    def test_invalid_common_inputs_raise(self, side, qty, entry_price):
        with pytest.raises(ValueError):
            isolated_liq_price(side, qty, entry_price, 4000.0, _BTC)

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError):
            isolated_liq_price(1, 1.0, 40000.0, -1.0, _BTC)


def _exact_lp(side, qty, entry, wallet, other_mm, other_upnl, table):
    """Independent exact-arithmetic oracle: Fraction math, brute-force bracket pick."""
    s, q, ep = Fraction(side), Fraction(qty), Fraction(entry)
    for bracket in table.brackets:
        mmr, cum = Fraction(bracket.mmr), Fraction(bracket.cum)
        lp = (Fraction(wallet) - Fraction(other_mm) + Fraction(other_upnl) + cum - s * q * ep) / (
            q * mmr - s * q
        )
        if Fraction(bracket.floor) <= q * lp < Fraction(bracket.cap):
            return lp
    raise AssertionError("oracle found no consistent bracket")


def _margin_balance(side, qty, entry, wallet, other_mm, other_upnl, price):
    return wallet - other_mm + other_upnl + side * qty * (price - entry)


class TestTierTwoShortVectors:
    """Tier-2 SOL shorts: nonzero cum (75), TMM1 and UPNL1 all matter."""

    _SOL = _TABLES["SOLUSDT"]

    def test_isolated_short_uses_cum(self):
        lp = isolated_liq_price(-1, 1000.0, 100.0, 30000.0, self._SOL)
        assert lp == pytest.approx(
            float(_exact_lp(-1, 1000.0, 100.0, 30000.0, 0.0, 0.0, self._SOL)), rel=1e-12
        )
        assert lp == pytest.approx(129.23497267759564, rel=1e-12)
        assert self._SOL.bracket_for(1000.0 * lp).cum == 75.0
        mb = _margin_balance(-1, 1000.0, 100.0, 30000.0, 0.0, 0.0, lp)
        assert mb == pytest.approx(self._SOL.maintenance_margin(1000.0 * lp), rel=1e-12)

    def test_cross_short_uses_other_positions(self):
        lp = cross_liq_price(-1, 1000.0, 100.0, 30000.0, 500.0, -200.0, self._SOL)
        assert lp == pytest.approx(
            float(_exact_lp(-1, 1000.0, 100.0, 30000.0, 500.0, -200.0, self._SOL)), rel=1e-12
        )
        assert lp == pytest.approx(128.53949329359165, rel=1e-12)
        mb = _margin_balance(-1, 1000.0, 100.0, 30000.0, 500.0, -200.0, lp)
        assert mb == pytest.approx(self._SOL.maintenance_margin(1000.0 * lp), rel=1e-12)
        # TMM1 and UPNL1 each move the answer
        no_others = cross_liq_price(-1, 1000.0, 100.0, 30000.0, 0.0, 0.0, self._SOL)
        assert lp < no_others
        only_upnl = cross_liq_price(-1, 1000.0, 100.0, 30000.0, 0.0, -200.0, self._SOL)
        assert lp < only_upnl < no_others

    def test_cross_long_tier_two_with_others(self):
        lp = cross_liq_price(1, 1000.0, 100.0, 30000.0, 500.0, 200.0, self._SOL)
        assert lp == pytest.approx(
            float(_exact_lp(1, 1000.0, 100.0, 30000.0, 500.0, 200.0, self._SOL)), rel=1e-12
        )
        mb = _margin_balance(1, 1000.0, 100.0, 30000.0, 500.0, 200.0, lp)
        assert mb == pytest.approx(self._SOL.maintenance_margin(1000.0 * lp), rel=1e-12)


class TestInsolventPosition:
    """InsolventPosition only for a short whose LP <= 0 (insolvent at every price)."""

    def test_insolvent_cross_short_raises(self):
        # MB = 4000 - 100000 = -96000: short deficit grows with price, LP would be -55776.89
        with pytest.raises(InsolventPosition, match=r"-96000"):
            cross_liq_price(-1, 1.0, 40000.0, 4000.0, 0.0, -100000.0, _BTC)

    def test_insolvent_is_a_value_error(self):
        assert issubclass(InsolventPosition, ValueError)

    def test_long_with_non_positive_lp_is_never_not_insolvent(self):
        assert isolated_liq_price(1, 1.0, 40000.0, 40000.0, _BTC) <= 0.0
        assert cross_liq_price(1, 1.0, 40000.0, 50000.0, 0.0, 0.0, _BTC) <= 0.0
        # long on the same negative balance: LP is positive and above entry, returned as a number
        lp = cross_liq_price(1, 1.0, 40000.0, 4000.0, 0.0, -100000.0, _BTC)
        assert lp > 40000.0

    def test_short_with_positive_lp_is_returned_as_a_number(self):
        # MB = 5000 - 4000 = 1000 > 0: LP positive, no raise
        lp = cross_liq_price(-1, 1.0, 40000.0, 5000.0, 0.0, -4000.0, _BTC)
        assert lp == pytest.approx(float(_exact_lp(-1, 1.0, 40000.0, 5000.0, 0.0, -4000.0, _BTC)))
        assert lp > 0.0


class TestLpIsNotAnEntrySolvencyCheck:
    """Callers compare LP with the CURRENT mark; the LP function never judges the entry."""

    def test_long_after_funding_returns_lp_above_entry(self):
        # q1 EP40000 margin 100: (100 - 40000)/(0.004 - 1) = 40060.24...; the position is
        # solvent at a mark of 50000 (MB 10100 > MM 200), so no raise.
        lp = isolated_liq_price(1, 1.0, 40000.0, 100.0, _BTC)
        assert lp == pytest.approx(39900.0 / 0.996, rel=1e-12)
        assert 40060.0 < lp < 40061.0 < 50000.0

    def test_short_after_funding_returns_lp_below_entry(self):
        # (100 + 40000)/(0.004 + 1) = 39940.239...; solvent at a mark of 30000 (MB 10100 > MM 120)
        lp = isolated_liq_price(-1, 1.0, 40000.0, 100.0, _BTC)
        assert lp == pytest.approx(40100.0 / 1.004, rel=1e-12)
        assert 30000.0 < lp < 40000.0

    def test_zero_margin_returns_lp_beyond_entry(self):
        assert isolated_liq_price(1, 1.0, 40000.0, 0.0, _BTC) == pytest.approx(40000.0 / 0.996)
        assert isolated_liq_price(-1, 1.0, 40000.0, 0.0, _BTC) == pytest.approx(40000.0 / 1.004)


class TestBoundaryRobustness:
    # tier1 [0,1000) mmr .25 cum 0; tier2 [1000,1e6) mmr .5 cum 250 (= 1000*(.5-.25)); dyadic => exact floats
    _T = _table("DY", [(0.0, 1000.0, 0.25, 0.0, None), (1000.0, 1e6, 0.5, 250.0, None)])

    def test_lp_exactly_on_shared_boundary_owned_by_upper_bracket(self):
        # tier1: (1250-2000)/(.25-1) = 1000.0; tier2: (1250+250-2000)/(.5-1) = 1000.0
        lp = isolated_liq_price(1, 1.0, 2000.0, 1250.0, self._T)
        assert lp == 1000.0

    def test_lp_just_below_boundary_prices_with_lower_bracket(self):
        below = 999.9375  # dyadic: tier-1 LP = (WB - 2000)/(.25 - 1) exactly
        lp = isolated_liq_price(1, 1.0, 2000.0, 2000.0 - 0.75 * below, self._T)
        assert lp == below

    def test_lp_just_above_boundary_prices_with_upper_bracket(self):
        above = 1000.0625  # tier-2 LP = (WB + 250 - 2000)/(.5 - 1) exactly
        lp = isolated_liq_price(1, 1.0, 2000.0, 2000.0 - 250.0 - 0.5 * above, self._T)
        assert lp == above

    def test_rounding_at_btc_boundary_does_not_raise(self):
        # reviewer case: adjacent candidate notionals 800000.0 / 799999.9999999999
        lp = isolated_liq_price(1, 3.0, 300000.0, 103699.99999999994, _BTC)
        assert lp == pytest.approx(800000.0 / 3.0, rel=1e-12)
        short = isolated_liq_price(-1, 3.0, 300000.0, 103699.99999999994, _BTC)
        assert short > 300000.0

    @pytest.mark.parametrize("margin", [51.5, 51.50000001, 60.0])
    def test_short_lp_at_or_above_last_cap_raises_exceeded(self, margin):
        # 51.5 -> LP notional exactly 150.0 == cap; slack must not leak the cap through
        top = _table("TOP", [(0.0, 150.0, 0.01, 0.0, None)])
        with pytest.raises(TierTableExceeded):
            isolated_liq_price(-1, 1.0, 100.0, margin, top)

    def test_shipped_btc_short_lp_beyond_last_cap_raises_exceeded(self):
        with pytest.raises(TierTableExceeded):
            isolated_liq_price(-1, 50000.0, 40000.0, 1e9, _BTC)

    def test_interior_boundary_slack_still_applies(self):
        # reviewer rounding case sits on the 800000 interior boundary and must keep returning
        lp = isolated_liq_price(1, 3.0, 300000.0, 103699.99999999994, _BTC)
        assert lp == pytest.approx(800000.0 / 3.0, rel=1e-12)

    def test_short_lp_above_last_cap_raises_exceeded(self):
        with pytest.raises(TierTableExceeded):
            isolated_liq_price(-1, 1.0, 100.0, 60.0, _table("TOP", [(0.0, 150.0, 0.01, 0.0, None)]))


class TestThinLastTierCapGuard:
    """A valid THIN final tier must not leak the last cap through interior slack.

    Reviewer r2 case: the interior (non-last) bracket's boundary slack widens its
    cap enough to accept a notional that is at or beyond the last cap, so the
    last-cap guard has to run before the slack lookup (not after it).
    """

    _THIN = (
        Bracket(0.0, 149.99999999, 0.01, 0.0, None),
        Bracket(149.99999999, 150.0, 0.02, 1.50000000000003, None),
    )

    def _thin(self) -> TierTable:
        """The 1e-8-wide table built directly: the loader now rejects it.

        Direct construction keeps the guard-order regression independent of the
        loader's min-bracket-width rule (see ``TestMinBracketWidth``).
        """
        return TierTable(symbol="THIN", brackets=self._THIN, clearance_fee_rate=None)

    def test_thin_last_tier_cap_raises_exceeded_isolated(self):
        """Isolated short, q=1, EP=100, margin 51.5000000001.

          tier1 LP = (51.5000000001 + 0 + 100)/(0.01 + 1) = 151.5000000001/1.01
                   = 150.000000000099 -> interior slack would accept it;
          tier2 LP = (51.5000000001 + 1.50000000000003 + 100)/(0.02 + 1)
                   = 153.0000000001/1.02 = 150.000000000098 -> at/beyond the last cap.
        """
        with pytest.raises(TierTableExceeded):
            isolated_liq_price(-1, 1.0, 100.0, 51.5000000001, self._thin())

    def test_thin_last_tier_cap_raises_exceeded_cross(self):
        """Same table as the isolated case: cross short with TMM1 = UPNL1 = 0."""
        with pytest.raises(TierTableExceeded):
            cross_liq_price(-1, 1.0, 100.0, 51.5000000001, 0.0, 0.0, self._thin())

    def test_guard_runs_before_near_isolated(self):
        """Exact-cap boundary: the last-cap guard must run before the slack lookup.

          isolated short, q=1, EP=100, wallet 51.49999999999997:
          tier1 LP = 151.49999999999997/1.01 = 149.99999999999997 -> interior slack
          would accept it;
          tier2 LP = (51.49999999999997 + 1.50000000000003 + 100)/(0.02 + 1)
                   = 153/1.02 = 150.0 == the last cap exactly.
        The exact message proves the last-cap guard raised, not the slack lookup.
        """
        with pytest.raises(TierTableExceeded) as excinfo:
            isolated_liq_price(-1, 1.0, 100.0, 51.49999999999997, self._thin())
        assert str(excinfo.value) == "THIN: liquidation notional 150.0 >= last cap 150.0"

    def test_guard_runs_before_near_cross(self):
        """Same table as the isolated case: cross short with TMM1 = UPNL1 = 0."""
        with pytest.raises(TierTableExceeded) as excinfo:
            cross_liq_price(-1, 1.0, 100.0, 51.49999999999997, 0.0, 0.0, self._thin())
        assert str(excinfo.value) == "THIN: liquidation notional 150.0 >= last cap 150.0"


class TestMinBracketWidth:
    """The loader rejects degenerate (sub-1-USDT-wide) brackets.

    A 1e-8-wide last bracket makes the interior-slack class reachable; the
    direct-construction guard-order regression above pins the solver, while
    these tests pin the loader rule that keeps such snapshots out.
    """

    def test_loader_rejects_thin_bracket(self, tmp_path):
        path = _snapshot(
            tmp_path,
            {
                "THIN": _symbol(
                    [
                        _bracket(0.0, 149.99999999, 0.01, 0.0, None),
                        _bracket(149.99999999, 150.0, 0.02, 1.4999999999, None),
                    ]
                )
            },
        )
        with pytest.raises(TierTableError, match=r"bracket 1 width .* < 1\.0 USDT"):
            load_tier_tables(path)

    def test_min_width_boundary(self, tmp_path):
        accepted = _snapshot(
            tmp_path,
            {
                "X": _symbol(
                    [
                        _bracket(0.0, 1000.0, 0.01, 0.0, None),
                        _bracket(1000.0, 1001.0, 0.02, 10.0, None),
                    ]
                )
            },
        )
        tables = load_tier_tables(accepted)
        second = tables["X"].brackets[1]
        assert second.cap - second.floor == 1.0
        rejected = _snapshot(
            tmp_path,
            {
                "X": _symbol(
                    [
                        _bracket(0.0, 1000.0, 0.01, 0.0, None),
                        _bracket(1000.0, 1000.999999, 0.02, 10.0, None),
                    ]
                )
            },
        )
        with pytest.raises(TierTableError, match=r"bracket 1 width .* < 1\.0 USDT"):
            load_tier_tables(rejected)


class TestExactZeroLiquidationPrice:
    """``candidates[0] <= 0`` (not ``< 0``): exact-zero LP is insolvent for a short."""

    _ZERO = _table("Z", [(0.0, 1000.0, 0.01, 0.0, None)])

    def test_short_lp_exactly_zero_is_insolvent(self):
        """Cross short q=1, EP=100, wallet 0, TMM1 0, UPNL1 -100.

          numerator = 0 - 0 + (-100) + 0 + 100 = 0; LP = 0/(0.01 + 1) = 0.0.
        """
        with pytest.raises(InsolventPosition):
            cross_liq_price(-1, 1.0, 100.0, 0.0, 0.0, -100.0, self._ZERO)

    def test_long_lp_exactly_zero_never_liquidates(self):
        """Isolated long q=1, EP=100, margin 100.

          numerator = 100 - 0 + 0 + 0 - 100 = 0; LP = 0/(0.01 - 1) = -0.0 -> "never".
        """
        assert isolated_liq_price(1, 1.0, 100.0, 100.0, self._ZERO) == 0.0


class TestInvalidTables:
    @pytest.mark.parametrize(
        "mutate",
        [
            {"mmr": 1.5},
            {"mmr": 1.0},
            {"mmr": -0.01},
            {"mmr": float("nan")},
            {"cum": float("nan")},
            {"cum": -5.0},
            {"cum": float("inf")},
            {"cap": float("inf")},
            {"floor": float("nan")},
            {"max_leverage": 0},
        ],
    )
    def test_invalid_numeric_domain_raises(self, tmp_path, mutate):
        bracket = {**_bracket(0.0, 1000.0, 0.01, 0.0), **mutate}
        path = _snapshot(tmp_path, {"X": _symbol([bracket])})
        with pytest.raises(TierTableError):
            load_tier_tables(path)

    @pytest.mark.parametrize("fee", [float("nan"), -0.1, 1.5])
    def test_invalid_clearance_fee_raises(self, tmp_path, fee):
        path = _snapshot(tmp_path, {"X": _symbol([_bracket(0.0, 1000.0, 0.01, 0.0)], fee=fee)})
        with pytest.raises(TierTableError):
            load_tier_tables(path)

    def test_bracket_direct_construction_validates(self):
        with pytest.raises(TierTableError):
            Bracket(0.0, 1000.0, 1.5, 0.0, None)

    def test_zero_width_keeps_cap_message(self, tmp_path):
        # cap == floor is checked before the min-width rule, so the message is
        # the pre-existing "cap ... <= floor" one, not the degenerate-width one.
        path = _snapshot(tmp_path, {"X": _symbol([_bracket(0.0, 0.0, 0.01, 0.0, None)])})
        with pytest.raises(TierTableError, match=r"cap 0\.0 <= floor 0\.0"):
            load_tier_tables(path)

    def test_gap_between_brackets_raises(self, tmp_path):
        # cap 300000 then floor 300000.1: continuity of cum holds (0.1 * 0.005 is tiny)
        brackets = [
            _bracket(0.0, 300000.0, 0.004, 0.0),
            _bracket(300000.1, 800000.0, 0.005, 300.0005),
        ]
        path = _snapshot(tmp_path, {"X": _symbol(brackets)})
        with pytest.raises(TierTableError, match="gap|contiguous|previous cap"):
            load_tier_tables(path)


class TestInputTypes:
    @pytest.mark.parametrize(
        "args",
        [
            (1, None, 40000.0, 4000.0),
            (1, 1.0, None, 4000.0),
            (1, 1.0, 40000.0, None),
            (None, 1.0, 40000.0, 4000.0),
            (1, "1", 40000.0, 4000.0),
            (True, 1.0, 40000.0, 4000.0),
            (1, True, 40000.0, 4000.0),
            (1, 1.0, 40000.0, math.nan),
        ],
    )
    def test_isolated_bad_types_raise_value_error(self, args):
        with pytest.raises(ValueError):
            isolated_liq_price(*args, _BTC)

    @pytest.mark.parametrize(
        "args",
        [
            (1, 1.0, 40000.0, None, 0.0, 0.0),
            (1, 1.0, 40000.0, 4000.0, None, 0.0),
            (1, 1.0, 40000.0, 4000.0, 0.0, None),
            (1, 1.0, 40000.0, 4000.0, 0.0, math.nan),
            (1, 1.0, 40000.0, 4000.0, -1.0, 0.0),
        ],
    )
    def test_cross_bad_types_raise_value_error(self, args):
        with pytest.raises(ValueError):
            cross_liq_price(*args, _BTC)
