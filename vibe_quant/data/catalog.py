"""NautilusTrader ParquetDataCatalog management."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

# Default catalog path
DEFAULT_CATALOG_PATH = Path("data/catalog")

# Venue identifier
BINANCE_VENUE = Venue("BINANCE")

# Instrument configurations for supported symbols
INSTRUMENT_CONFIGS = {
    "BTCUSDT": {
        "base": "BTC",
        "quote": "USDT",
        "price_precision": 1,
        "size_precision": 3,
        "price_increment": "0.1",
        "size_increment": "0.001",
        "max_leverage": Decimal("125"),
        "margin_init": Decimal("0.008"),
        "margin_maint": Decimal("0.004"),
        "maker_fee": Decimal("0.0002"),
        "taker_fee": Decimal("0.0004"),
    },
    "ETHUSDT": {
        "base": "ETH",
        "quote": "USDT",
        "price_precision": 2,
        "size_precision": 3,
        "price_increment": "0.01",
        "size_increment": "0.001",
        "max_leverage": Decimal("100"),
        "margin_init": Decimal("0.01"),
        "margin_maint": Decimal("0.005"),
        "maker_fee": Decimal("0.0002"),
        "taker_fee": Decimal("0.0004"),
    },
    "SOLUSDT": {
        "base": "SOL",
        "quote": "USDT",
        "price_precision": 3,
        "size_precision": 0,
        "price_increment": "0.001",
        "size_increment": "1",
        "max_leverage": Decimal("50"),
        "margin_init": Decimal("0.02"),
        "margin_maint": Decimal("0.01"),
        "maker_fee": Decimal("0.0002"),
        "taker_fee": Decimal("0.0004"),
    },
}

# Bar aggregation mapping
INTERVAL_TO_AGGREGATION = {
    "1m": (1, BarAggregation.MINUTE),
    "5m": (5, BarAggregation.MINUTE),
    "15m": (15, BarAggregation.MINUTE),
    "1h": (1, BarAggregation.HOUR),
    "4h": (4, BarAggregation.HOUR),
    "1d": (1, BarAggregation.DAY),
}


def create_instrument(symbol: str) -> CryptoPerpetual:
    """Create NautilusTrader instrument for a symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').

    Returns:
        CryptoPerpetual instrument.
    """
    config = INSTRUMENT_CONFIGS[symbol]
    quote = Currency.from_str(config["quote"])

    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(f"{symbol}-PERP"), BINANCE_VENUE),
        raw_symbol=Symbol(symbol),
        base_currency=Currency.from_str(config["base"]),
        quote_currency=quote,
        settlement_currency=quote,
        is_inverse=False,
        price_precision=config["price_precision"],
        size_precision=config["size_precision"],
        price_increment=Price.from_str(config["price_increment"]),
        size_increment=Quantity.from_str(config["size_increment"]),
        max_quantity=None,
        min_quantity=None,
        max_notional=None,
        min_notional=Money(5, quote),  # Binance minimum
        max_price=None,
        min_price=None,
        margin_init=config["margin_init"],
        margin_maint=config["margin_maint"],
        maker_fee=config["maker_fee"],
        taker_fee=config["taker_fee"],
        ts_event=0,
        ts_init=0,
    )


def klines_to_bars(
    klines: Sequence[sqlite3.Row],
    instrument_id: InstrumentId,
    bar_type: BarType,
    size_precision: int = 8,
    price_precision: int = 2,
) -> list[Bar]:
    """Convert raw klines to NautilusTrader Bar objects.

    Args:
        klines: Sequence of kline rows from archive.
        instrument_id: NautilusTrader instrument ID.
        bar_type: NautilusTrader bar type.
        size_precision: Decimal places for volume (must match instrument).
            Default 8 kept for API compatibility; callers should pass
            instrument.size_precision explicitly for production use.
        price_precision: Decimal places for prices (must match instrument).
            Default 2; callers should pass instrument.price_precision.

    Returns:
        List of Bar objects.
    """
    bars = []
    for k in klines:
        # Convert ms timestamp to ns
        ts_event = int(k["open_time"]) * 1_000_000
        ts_init = int(k["close_time"]) * 1_000_000

        # Round volume to instrument size_precision to avoid NT mismatch
        vol = round(float(k["volume"]), size_precision)
        vol_str = f"{vol:.{size_precision}f}"

        # Format prices to exact instrument precision to avoid NT mismatch
        # (e.g. "3500.1" -> "3500.10" for price_precision=2)
        pp = price_precision

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{float(k['open']):.{pp}f}"),
            high=Price.from_str(f"{float(k['high']):.{pp}f}"),
            low=Price.from_str(f"{float(k['low']):.{pp}f}"),
            close=Price.from_str(f"{float(k['close']):.{pp}f}"),
            volume=Quantity.from_str(vol_str),
            ts_event=ts_event,
            ts_init=ts_init,
        )
        bars.append(bar)

    return bars


def aggregate_bars(
    bars_1m: list[Bar],
    target_bar_type: BarType,
    target_minutes: int,
    size_precision: int = 8,
) -> list[Bar]:
    """Aggregate 1-minute bars to higher timeframe.

    Pre-computes a combined divisor (ns -> target-minute alignment)
    to replace the two-step division per bar with a single integer
    division. Tracks group OHLCV inline to avoid building intermediate
    lists where possible.

    Args:
        bars_1m: List of 1-minute bars (sorted by time).
        target_bar_type: Target bar type for aggregated bars.
        target_minutes: Target bar duration in minutes.
        size_precision: Decimal places for volume (must match instrument).

    Returns:
        List of aggregated bars.
    """
    if not bars_1m:
        return []

    # Pre-compute combined divisor: ns -> ms -> minutes -> target-aligned
    # bar_ts_ns // 1_000_000 // 60_000 // target_minutes
    # = bar_ts_ns // (1_000_000 * 60_000 * target_minutes)
    ns_per_target_period = 1_000_000 * 60_000 * target_minutes

    aggregated: list[Bar] = []
    current_group: list[Bar] = []
    group_key: int | None = None

    for bar in bars_1m:
        bar_key = bar.ts_event // ns_per_target_period

        if group_key is None:
            group_key = bar_key

        if bar_key != group_key:
            # Aggregate current group
            if current_group:
                aggregated.append(
                    _aggregate_group(current_group, target_bar_type, size_precision)
                )

            # Start new group
            current_group = [bar]
            group_key = bar_key
        else:
            current_group.append(bar)

    # Don't forget the last group
    if current_group:
        aggregated.append(
            _aggregate_group(current_group, target_bar_type, size_precision)
        )

    return aggregated


def _aggregate_group(bars: list[Bar], bar_type: BarType, size_precision: int = 8) -> Bar:
    """Aggregate a group of bars into a single bar.

    Uses a single loop to track high/low/volume instead of
    separate max/min/sum generator expressions.

    Args:
        bars: List of bars to aggregate.
        bar_type: Target bar type.
        size_precision: Decimal places for volume (must match instrument).

    Returns:
        Aggregated bar.
    """
    first = bars[0]
    if len(bars) == 1:
        # Fast path: single bar, no aggregation needed
        return Bar(
            bar_type=bar_type,
            open=first.open,
            high=first.high,
            low=first.low,
            close=first.close,
            volume=first.volume,
            ts_event=first.ts_event,
            ts_init=first.ts_init,
        )

    # Single-pass OHLCV computation
    high_price = first.high
    low_price = first.low
    total_volume = float(first.volume)

    for i in range(1, len(bars)):
        b = bars[i]
        h = b.high
        if h > high_price:
            high_price = h
        lo = b.low
        if lo < low_price:
            low_price = lo
        total_volume += float(b.volume)

    last = bars[-1]
    vol = round(total_volume, size_precision)
    vol_str = f"{vol:.{size_precision}f}"
    return Bar(
        bar_type=bar_type,
        open=first.open,
        high=high_price,
        low=low_price,
        close=last.close,
        volume=Quantity.from_str(vol_str),
        ts_event=first.ts_event,
        ts_init=last.ts_init,
    )


def get_bar_type(symbol: str, interval: str) -> BarType:
    """Get NautilusTrader BarType for a symbol and interval.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (e.g., '1m', '5m', '1h').

    Returns:
        NautilusTrader BarType.
    """
    instrument_id = InstrumentId(Symbol(f"{symbol}-PERP"), BINANCE_VENUE)
    step, aggregation = INTERVAL_TO_AGGREGATION[interval]

    bar_spec = BarSpecification(step, aggregation, PriceType.LAST)
    return BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
    )


class CatalogManager:
    """Manager for NautilusTrader ParquetDataCatalog."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        """Initialize catalog manager.

        Args:
            catalog_path: Path to catalog directory. Uses default if not specified.
        """
        self._catalog_path = catalog_path or DEFAULT_CATALOG_PATH
        self._catalog: ParquetDataCatalog | None = None

    @property
    def catalog(self) -> ParquetDataCatalog:
        """Get or create catalog."""
        if self._catalog is None:
            self._catalog_path.mkdir(parents=True, exist_ok=True)
            self._catalog = ParquetDataCatalog(str(self._catalog_path))
        return self._catalog

    def write_instrument(self, instrument: CryptoPerpetual) -> None:
        """Write instrument to catalog.

        Args:
            instrument: Instrument to write.
        """
        self.catalog.write_data([instrument])

    def clear_bar_data(self, symbol: str, interval: str) -> None:
        """Remove existing parquet files for a bar type.

        Used before re-writing bars from archive to avoid disjoint interval errors.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
        """
        import shutil

        bar_type = get_bar_type(symbol, interval)
        # NautilusTrader catalog stores bars under data/bar/<bar_type_str>/
        bar_dir = self._catalog_path / "data" / "bar" / str(bar_type)
        if bar_dir.exists():
            shutil.rmtree(bar_dir)
        # Reset catalog cache so it re-reads from disk
        self._catalog = None

    def write_bars(self, bars: list[Bar]) -> None:
        """Write bars to catalog.

        Validates that bars are non-empty and have valid (non-zero) timestamps
        before writing to avoid corrupt 4-byte parquet files.

        Args:
            bars: List of bars to write.
        """
        if not bars:
            return

        # Filter out bars with epoch-zero timestamps (corrupt data)
        valid_bars = [b for b in bars if b.ts_event > 0 and b.ts_init > 0]
        if not valid_bars:
            return

        self.catalog.write_data(valid_bars)

    def get_instruments(self) -> list[CryptoPerpetual]:
        """Get all instruments from catalog.

        Returns:
            List of instruments.
        """
        return self.catalog.instruments()

    def get_bar_count(self, symbol: str, interval: str) -> int:
        """Get count of bars for a symbol and interval.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.

        Returns:
            Number of bars.
        """
        bar_type = get_bar_type(symbol, interval)
        bars = self.catalog.bars(bar_types=[bar_type])
        return len(bars)

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        """Get bars for a symbol and interval with optional date filtering.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            List of Bar objects.
        """
        bar_type = get_bar_type(symbol, interval)
        return self.catalog.bars(
            bar_types=[bar_type], start=start, end=end,  # type: ignore[arg-type]
        )

    def get_bar_date_range(
        self, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        """Get date range of bars for a symbol and interval.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.

        Returns:
            (start_datetime, end_datetime) or None if no data.
        """
        bar_type = get_bar_type(symbol, interval)
        bars = self.catalog.bars(bar_types=[bar_type])

        if not bars:
            return None

        # Convert nanoseconds to datetime
        start_ns = bars[0].ts_event
        end_ns = bars[-1].ts_event
        start_dt = datetime.fromtimestamp(start_ns / 1e9, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ns / 1e9, tz=UTC)

        return (start_dt, end_dt)
