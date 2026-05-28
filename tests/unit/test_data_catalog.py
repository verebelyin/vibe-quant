"""Tests for data catalog module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from nautilus_trader.model.data import Bar, BarType

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    aggregate_bars,
    get_bar_type,
    klines_to_bars,
)


@pytest.fixture
def btc_bar_type_1m() -> BarType:
    """Create 1m bar type for BTCUSDT."""
    return get_bar_type("BTCUSDT", "1m")


@pytest.fixture
def btc_bar_type_5m() -> BarType:
    """Create 5m bar type for BTCUSDT."""
    return get_bar_type("BTCUSDT", "5m")


@pytest.fixture
def btc_instrument_id() -> InstrumentId:
    """Create instrument ID for BTCUSDT."""
    return InstrumentId(Symbol("BTCUSDT-PERP"), Venue("BINANCE"))


def make_bar(
    bar_type: BarType,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    ts_event_ms: int,
    ts_init_ms: int,
) -> Bar:
    """Create a Bar for testing."""
    return Bar(
        bar_type=bar_type,
        open=Price.from_str(str(open_price)),
        high=Price.from_str(str(high)),
        low=Price.from_str(str(low)),
        close=Price.from_str(str(close)),
        volume=Quantity.from_str(str(volume)),
        ts_event=ts_event_ms * 1_000_000,  # Convert ms to ns
        ts_init=ts_init_ms * 1_000_000,
    )


class TestAggregateBars:
    """Tests for aggregate_bars function."""

    def test_aggregate_bars_5m(self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType) -> None:
        """5 1m bars aggregate to 1 5m bar with correct OHLCV."""
        # Create 5 consecutive 1m bars starting at minute 0
        bars_1m = [
            make_bar(btc_bar_type_1m, 100.0, 105.0, 95.0, 102.0, 10.0, 0, 59999),
            make_bar(btc_bar_type_1m, 102.0, 108.0, 100.0, 106.0, 15.0, 60000, 119999),
            make_bar(btc_bar_type_1m, 106.0, 110.0, 104.0, 108.0, 12.0, 120000, 179999),
            make_bar(btc_bar_type_1m, 108.0, 112.0, 106.0, 109.0, 8.0, 180000, 239999),
            make_bar(btc_bar_type_1m, 109.0, 115.0, 107.0, 113.0, 20.0, 240000, 299999),
        ]

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)

        assert len(result) == 1
        agg_bar = result[0]

        # Check OHLC
        assert float(agg_bar.open) == 100.0  # First bar's open
        assert float(agg_bar.high) == 115.0  # Max high across all bars
        assert float(agg_bar.low) == 95.0  # Min low across all bars
        assert float(agg_bar.close) == 113.0  # Last bar's close

        # Check volume (sum of all)
        assert float(agg_bar.volume) == pytest.approx(65.0)

        # Check timestamps
        assert agg_bar.ts_event == 0  # First bar's event time
        assert agg_bar.ts_init == 299999 * 1_000_000  # Last bar's init time

    def test_aggregate_bars_multiple_periods(
        self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType
    ) -> None:
        """10 1m bars aggregate to 2 5m bars."""
        # Create 10 consecutive 1m bars (2 x 5m periods)
        bars_1m = []
        for i in range(10):
            ts = i * 60000
            bars_1m.append(
                make_bar(
                    btc_bar_type_1m,
                    100.0 + i,
                    105.0 + i,
                    95.0 + i,
                    102.0 + i,
                    10.0,
                    ts,
                    ts + 59999,
                )
            )

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)

        assert len(result) == 2

        # First 5m bar (minutes 0-4)
        assert float(result[0].open) == 100.0
        assert float(result[0].close) == 106.0

        # Second 5m bar (minutes 5-9)
        assert float(result[1].open) == 105.0
        assert float(result[1].close) == 111.0

    def test_aggregate_bars_empty(self, btc_bar_type_5m: BarType) -> None:
        """Empty input returns empty list."""
        result = aggregate_bars([], btc_bar_type_5m, 5)
        assert result == []

    def test_aggregate_bars_partial_period(
        self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType
    ) -> None:
        """Partial period still gets aggregated."""
        # Only 3 bars (partial 5m period)
        bars_1m = [
            make_bar(btc_bar_type_1m, 100.0, 105.0, 95.0, 102.0, 10.0, 0, 59999),
            make_bar(btc_bar_type_1m, 102.0, 108.0, 100.0, 106.0, 15.0, 60000, 119999),
            make_bar(btc_bar_type_1m, 106.0, 110.0, 104.0, 108.0, 12.0, 120000, 179999),
        ]

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)

        assert len(result) == 1
        assert float(result[0].open) == 100.0
        assert float(result[0].high) == 110.0
        assert float(result[0].low) == 95.0
        assert float(result[0].close) == 108.0

    def test_aggregate_bars_single_bar(
        self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType
    ) -> None:
        """Single bar produces single aggregated bar."""
        bars_1m = [
            make_bar(btc_bar_type_1m, 100.0, 105.0, 95.0, 102.0, 10.0, 0, 59999),
        ]

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)

        assert len(result) == 1
        assert float(result[0].open) == 100.0


class TestAggregateBarsAlignment:
    """No-leakage guards: aggregation must never pull a value across a window
    edge. Locks the alignment contract behind the Reddit-audit resample class
    (resample/merge_asof pulling a value from the wrong side of its timestamp).
    """

    @staticmethod
    def _minute_bar(bar_type: BarType, minute: int, o: float, h: float, low: float, c: float, v: float) -> Bar:
        """1m bar at an absolute minute offset (ts_event=open edge, ts_init=close edge)."""
        ts = minute * 60_000
        return make_bar(bar_type, o, h, low, c, v, ts, ts + 59_999)

    @staticmethod
    def _expect(window: list[Bar]) -> tuple[float, float, float, float, float, int, int]:
        """Expected (open, high, low, close, volume, ts_event, ts_init) for exactly
        these bars -- nothing from neighbouring windows."""
        return (
            float(window[0].open),
            max(float(b.high) for b in window),
            min(float(b.low) for b in window),
            float(window[-1].close),
            sum(float(b.volume) for b in window),
            window[0].ts_event,
            window[-1].ts_init,
        )

    def test_window_reductions_have_no_cross_edge_contamination(
        self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType
    ) -> None:
        """3 full 5m windows. Window 1 carries an extreme high+low spike; window
        0's high/low must exclude them (no backward leak), window 2 stays clean
        (no forward leak). Each window's OHLCV/timestamps come only from its own
        1m bars."""
        bars_1m = []
        for m in range(15):
            high = 9999.0 if m == 5 else 100.0 + m + 1
            low = -9999.0 if m == 6 else 100.0 + m - 1
            bars_1m.append(self._minute_bar(btc_bar_type_1m, m, 100.0 + m, high, low, 100.0 + m + 0.5, m + 1.0))

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)
        assert len(result) == 3

        for w, agg in enumerate(result):
            exp_o, exp_h, exp_l, exp_c, exp_v, exp_te, exp_ti = self._expect(bars_1m[w * 5 : w * 5 + 5])
            assert float(agg.open) == exp_o
            assert float(agg.high) == exp_h
            assert float(agg.low) == exp_l
            assert float(agg.close) == exp_c
            assert float(agg.volume) == pytest.approx(exp_v)
            assert agg.ts_event == exp_te  # open edge = first bar
            assert agg.ts_init == exp_ti  # close edge = last bar

        # The spikes live only in window 1, never bleed into neighbours.
        assert float(result[0].high) < 9999.0
        assert float(result[0].low) > -9999.0
        assert float(result[1].high) == 9999.0
        assert float(result[1].low) == -9999.0
        assert float(result[2].high) < 9999.0
        assert float(result[2].low) > -9999.0

    def test_binning_is_absolute_not_positional(
        self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType
    ) -> None:
        """Bars start at minute 2 (mid-window). Windows are epoch-aligned
        (floor(minute/5)), NOT chunked every-5-from-the-first-bar. So minutes
        2-4 form a short first window and minute 5 opens the next -- proving a
        minute-5 value can't be pulled into the [0,5) window."""
        bars_1m = [
            self._minute_bar(btc_bar_type_1m, m, 100.0 + m, 105.0 + m, 95.0 + m, 102.0 + m, 10.0)
            for m in range(2, 9)  # minutes 2..8
        ]

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)
        assert len(result) == 2

        # Window [0,5): only minutes 2,3,4 present.
        assert result[0].ts_event == 2 * 60_000 * 1_000_000
        assert result[0].ts_init == (4 * 60_000 + 59_999) * 1_000_000
        assert float(result[0].open) == 102.0  # minute 2 open
        assert float(result[0].close) == 106.0  # minute 4 close
        # Window [5,10): minutes 5..8 — minute 5 did NOT leak into window 0.
        assert result[1].ts_event == 5 * 60_000 * 1_000_000
        assert float(result[1].open) == 105.0  # minute 5 open

    def test_intra_window_gap_stays_in_its_window(
        self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType
    ) -> None:
        """A missing minute inside a window must not shift later bars across the
        edge: minutes 0,1,3,4 still all land in window [0,5); minute 5 opens the
        next window regardless of the gap."""
        minutes = [0, 1, 3, 4, 5, 6]
        bars_1m = [
            self._minute_bar(btc_bar_type_1m, m, 100.0 + m, 105.0 + m, 95.0 + m, 102.0 + m, 10.0)
            for m in minutes
        ]

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)
        assert len(result) == 2
        assert result[0].ts_init == (4 * 60_000 + 59_999) * 1_000_000  # last bar in [0,5)
        assert float(result[0].close) == 106.0  # minute 4, not minute 5
        assert result[1].ts_event == 5 * 60_000 * 1_000_000

    def test_partial_trailing_window_is_bounded_not_extrapolated(
        self, btc_bar_type_1m: BarType, btc_bar_type_5m: BarType
    ) -> None:
        """5 + 2 bars -> one full window then a 2-bar trailing window. The
        partial window is bounded to its present bars (ts_init = its last 1m
        bar's close edge), never mislabeled as a full-window close edge."""
        bars_1m = [
            self._minute_bar(btc_bar_type_1m, m, 100.0 + m, 105.0 + m, 95.0 + m, 102.0 + m, 10.0)
            for m in range(7)
        ]

        result = aggregate_bars(bars_1m, btc_bar_type_5m, 5)
        assert len(result) == 2

        tail = result[1]
        exp_o, exp_h, exp_l, exp_c, exp_v, exp_te, exp_ti = self._expect(bars_1m[5:7])
        assert float(tail.open) == exp_o
        assert float(tail.high) == exp_h
        assert float(tail.low) == exp_l
        assert float(tail.close) == exp_c
        assert float(tail.volume) == pytest.approx(exp_v)
        assert tail.ts_event == 5 * 60_000 * 1_000_000
        # Bounded to minute 6, NOT extrapolated to the full-window edge (minute 9).
        assert tail.ts_init == (6 * 60_000 + 59_999) * 1_000_000

    def test_15m_window_timestamps_use_open_and_close_edges(
        self, btc_bar_type_1m: BarType
    ) -> None:
        """Second target_minutes (15m): ts_event = first 1m open edge, ts_init =
        last 1m close edge; OHLCV reduced over exactly the 15 constituent bars."""
        bar_type_15m = get_bar_type("BTCUSDT", "15m")
        bars_1m = [
            self._minute_bar(btc_bar_type_1m, m, 100.0 + m, 105.0 + m, 95.0 + m, 102.0 + m, 2.0)
            for m in range(15)
        ]

        result = aggregate_bars(bars_1m, bar_type_15m, 15)
        assert len(result) == 1
        agg = result[0]
        assert agg.ts_event == 0
        assert agg.ts_init == (14 * 60_000 + 59_999) * 1_000_000
        assert float(agg.open) == 100.0
        assert float(agg.high) == 119.0  # 105 + 14
        assert float(agg.low) == 95.0
        assert float(agg.close) == 116.0  # 102 + 14
        assert float(agg.volume) == pytest.approx(30.0)


class TestKlinesToBars:
    """Tests for klines_to_bars function."""

    @pytest.fixture
    def archive(self, tmp_path: Path) -> Generator[RawDataArchive]:
        """Create archive with temp database."""
        db_path = tmp_path / "test_catalog.db"
        arc = RawDataArchive(db_path)
        yield arc
        arc.close()

    def test_klines_to_bars(
        self,
        archive: RawDataArchive,
        btc_instrument_id: InstrumentId,
        btc_bar_type_1m: BarType,
    ) -> None:
        """Conversion produces valid Bar objects."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
            (1704067260000, 42300.0, 42600.0, 42200.0, 42500.0, 150.0, 1704067319999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")
        rows = archive.get_klines("BTCUSDT", "1m")

        bars = klines_to_bars(rows, btc_instrument_id, btc_bar_type_1m)

        assert len(bars) == 2

        # Check first bar
        bar0 = bars[0]
        assert float(bar0.open) == 42000.0
        assert float(bar0.high) == 42500.0
        assert float(bar0.low) == 41800.0
        assert float(bar0.close) == 42300.0
        assert float(bar0.volume) == 100.0
        assert bar0.ts_event == 1704067200000 * 1_000_000  # ms to ns
        assert bar0.ts_init == 1704067259999 * 1_000_000

        # Check second bar
        bar1 = bars[1]
        assert float(bar1.open) == 42300.0
        assert float(bar1.close) == 42500.0

    def test_klines_to_bars_empty(
        self,
        btc_instrument_id: InstrumentId,
        btc_bar_type_1m: BarType,
    ) -> None:
        """Empty klines returns empty bars list."""
        bars = klines_to_bars([], btc_instrument_id, btc_bar_type_1m)
        assert bars == []

    def test_klines_to_bars_preserves_precision(
        self,
        archive: RawDataArchive,
        btc_instrument_id: InstrumentId,
        btc_bar_type_1m: BarType,
    ) -> None:
        """Conversion preserves decimal precision."""
        klines = [
            (1704067200000, 42000.12, 42500.34, 41800.56, 42300.78, 100.123, 1704067259999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")
        rows = archive.get_klines("BTCUSDT", "1m")

        bars = klines_to_bars(rows, btc_instrument_id, btc_bar_type_1m)

        assert len(bars) == 1
        # NautilusTrader Price preserves precision from string conversion
        assert "42000.12" in str(bars[0].open)
        assert "42500.34" in str(bars[0].high)


class TestGetBarType:
    """Tests for get_bar_type function."""

    def test_get_bar_type_1m(self) -> None:
        """1m bar type has correct spec."""
        bar_type = get_bar_type("BTCUSDT", "1m")
        assert bar_type.spec.step == 1
        assert bar_type.spec.aggregation == BarAggregation.MINUTE
        assert bar_type.spec.price_type == PriceType.LAST

    def test_get_bar_type_5m(self) -> None:
        """5m bar type has correct spec."""
        bar_type = get_bar_type("BTCUSDT", "5m")
        assert bar_type.spec.step == 5
        assert bar_type.spec.aggregation == BarAggregation.MINUTE

    def test_get_bar_type_1h(self) -> None:
        """1h bar type has correct spec."""
        bar_type = get_bar_type("BTCUSDT", "1h")
        assert bar_type.spec.step == 1
        assert bar_type.spec.aggregation == BarAggregation.HOUR

    def test_get_bar_type_4h(self) -> None:
        """4h bar type has correct spec."""
        bar_type = get_bar_type("BTCUSDT", "4h")
        assert bar_type.spec.step == 4
        assert bar_type.spec.aggregation == BarAggregation.HOUR

    def test_get_bar_type_1s(self) -> None:
        """1s bar type has correct spec."""
        bar_type = get_bar_type("BTCUSDT", "1s")
        assert bar_type.spec.step == 1
        assert bar_type.spec.aggregation == BarAggregation.SECOND

    def test_get_bar_type_5s(self) -> None:
        """5s bar type has correct spec."""
        bar_type = get_bar_type("BTCUSDT", "5s")
        assert bar_type.spec.step == 5
        assert bar_type.spec.aggregation == BarAggregation.SECOND

    def test_get_bar_type_instrument_id(self) -> None:
        """Bar type has correct instrument ID."""
        bar_type = get_bar_type("ETHUSDT", "5m")
        assert str(bar_type.instrument_id.symbol) == "ETHUSDT-PERP"
        assert str(bar_type.instrument_id.venue) == "BINANCE"
