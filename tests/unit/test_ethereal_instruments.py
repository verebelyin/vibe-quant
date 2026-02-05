"""Unit tests for Ethereal instrument definitions."""

from decimal import Decimal

import pytest
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual

from vibe_quant.ethereal.instruments import (
    ETHEREAL_FUNDING_INTERVAL_SECS,
    ETHEREAL_INSTRUMENT_CONFIGS,
    ETHEREAL_VENUE,
    create_ethereal_instrument,
    get_all_ethereal_instruments,
    get_ethereal_instrument,
    get_ethereal_symbols,
    get_max_leverage,
)


class TestEtherealInstrumentConfigs:
    """Tests for instrument configuration data."""

    def test_btcusd_max_leverage(self) -> None:
        """BTCUSD has 20x max leverage."""
        assert ETHEREAL_INSTRUMENT_CONFIGS["BTCUSD"].max_leverage == 20

    def test_ethusd_max_leverage(self) -> None:
        """ETHUSD has 20x max leverage."""
        assert ETHEREAL_INSTRUMENT_CONFIGS["ETHUSD"].max_leverage == 20

    def test_solusd_max_leverage(self) -> None:
        """SOLUSD has 10x max leverage."""
        assert ETHEREAL_INSTRUMENT_CONFIGS["SOLUSD"].max_leverage == 10

    def test_maker_fee_is_zero(self) -> None:
        """All instruments have 0% maker fee."""
        for config in ETHEREAL_INSTRUMENT_CONFIGS.values():
            assert config.maker_fee == Decimal("0.0000")

    def test_taker_fee_is_003_percent(self) -> None:
        """All instruments have 0.03% taker fee."""
        for config in ETHEREAL_INSTRUMENT_CONFIGS.values():
            assert config.taker_fee == Decimal("0.0003")

    def test_all_symbols_present(self) -> None:
        """All expected symbols are configured."""
        expected = {"BTCUSD", "ETHUSD", "SOLUSD"}
        assert set(ETHEREAL_INSTRUMENT_CONFIGS.keys()) == expected


class TestFundingInterval:
    """Tests for funding interval configuration."""

    def test_funding_interval_is_1_hour(self) -> None:
        """Funding interval is 1 hour (3600 seconds)."""
        assert ETHEREAL_FUNDING_INTERVAL_SECS == 3600

    def test_funding_interval_is_hourly_not_8_hourly(self) -> None:
        """Funding interval is NOT 8 hours (Binance default)."""
        binance_funding_interval = 8 * 3600
        assert binance_funding_interval != ETHEREAL_FUNDING_INTERVAL_SECS
        assert binance_funding_interval > ETHEREAL_FUNDING_INTERVAL_SECS


class TestEtherealVenue:
    """Tests for venue identifier."""

    def test_venue_is_ethereal(self) -> None:
        """Venue identifier is ETHEREAL."""
        assert str(ETHEREAL_VENUE) == "ETHEREAL"
        assert isinstance(ETHEREAL_VENUE, Venue)


class TestCreateEtherealInstrument:
    """Tests for create_ethereal_instrument function."""

    def test_creates_crypto_perpetual(self) -> None:
        """Creates CryptoPerpetual instance."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert isinstance(instrument, CryptoPerpetual)

    def test_btcusd_instrument_id(self) -> None:
        """BTCUSD has correct instrument ID."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert str(instrument.id.symbol) == "BTCUSD-PERP"
        assert str(instrument.id.venue) == "ETHEREAL"

    def test_ethusd_instrument_id(self) -> None:
        """ETHUSD has correct instrument ID."""
        instrument = create_ethereal_instrument("ETHUSD")
        assert str(instrument.id.symbol) == "ETHUSD-PERP"

    def test_solusd_instrument_id(self) -> None:
        """SOLUSD has correct instrument ID."""
        instrument = create_ethereal_instrument("SOLUSD")
        assert str(instrument.id.symbol) == "SOLUSD-PERP"

    def test_btcusd_leverage_margin(self) -> None:
        """BTCUSD margin_init reflects 20x max leverage."""
        instrument = create_ethereal_instrument("BTCUSD")
        # margin_init = 1/20 = 0.05
        assert instrument.margin_init == Decimal("0.05")

    def test_ethusd_leverage_margin(self) -> None:
        """ETHUSD margin_init reflects 20x max leverage."""
        instrument = create_ethereal_instrument("ETHUSD")
        assert instrument.margin_init == Decimal("0.05")

    def test_solusd_leverage_margin(self) -> None:
        """SOLUSD margin_init reflects 10x max leverage."""
        instrument = create_ethereal_instrument("SOLUSD")
        # margin_init = 1/10 = 0.1
        assert instrument.margin_init == Decimal("0.1")

    def test_maker_fee_applied(self) -> None:
        """Maker fee is 0% on instrument."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert instrument.maker_fee == Decimal("0.0000")

    def test_taker_fee_applied(self) -> None:
        """Taker fee is 0.03% on instrument."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert instrument.taker_fee == Decimal("0.0003")

    def test_settlement_in_usde(self) -> None:
        """Settlement currency is USDe."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert str(instrument.settlement_currency) == "USDE"

    def test_is_not_inverse(self) -> None:
        """Instrument is not inverse (linear)."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert instrument.is_inverse is False

    def test_unknown_symbol_raises(self) -> None:
        """Unknown symbol raises KeyError."""
        with pytest.raises(KeyError):
            create_ethereal_instrument("XYZUSD")


class TestGetEtherealInstrument:
    """Tests for get_ethereal_instrument helper."""

    def test_returns_instrument(self) -> None:
        """Returns CryptoPerpetual for valid symbol."""
        instrument = get_ethereal_instrument("BTCUSD")
        assert isinstance(instrument, CryptoPerpetual)

    def test_same_as_create(self) -> None:
        """Same result as create_ethereal_instrument."""
        created = create_ethereal_instrument("ETHUSD")
        gotten = get_ethereal_instrument("ETHUSD")
        assert created.id == gotten.id


class TestGetAllEtherealInstruments:
    """Tests for get_all_ethereal_instruments function."""

    def test_returns_all_three(self) -> None:
        """Returns all three instruments."""
        instruments = get_all_ethereal_instruments()
        assert len(instruments) == 3
        assert set(instruments.keys()) == {"BTCUSD", "ETHUSD", "SOLUSD"}

    def test_values_are_instruments(self) -> None:
        """All values are CryptoPerpetual instances."""
        instruments = get_all_ethereal_instruments()
        for instrument in instruments.values():
            assert isinstance(instrument, CryptoPerpetual)


class TestGetMaxLeverage:
    """Tests for get_max_leverage function."""

    def test_btcusd_leverage(self) -> None:
        """BTCUSD max leverage is 20."""
        assert get_max_leverage("BTCUSD") == 20

    def test_ethusd_leverage(self) -> None:
        """ETHUSD max leverage is 20."""
        assert get_max_leverage("ETHUSD") == 20

    def test_solusd_leverage(self) -> None:
        """SOLUSD max leverage is 10."""
        assert get_max_leverage("SOLUSD") == 10

    def test_unknown_symbol_raises(self) -> None:
        """Unknown symbol raises KeyError."""
        with pytest.raises(KeyError):
            get_max_leverage("UNKNOWN")


class TestGetEtherealSymbols:
    """Tests for get_ethereal_symbols function."""

    def test_returns_list(self) -> None:
        """Returns list of symbols."""
        symbols = get_ethereal_symbols()
        assert isinstance(symbols, list)

    def test_contains_all_symbols(self) -> None:
        """Contains all expected symbols."""
        symbols = get_ethereal_symbols()
        assert "BTCUSD" in symbols
        assert "ETHUSD" in symbols
        assert "SOLUSD" in symbols

    def test_count_is_three(self) -> None:
        """Exactly 3 symbols."""
        assert len(get_ethereal_symbols()) == 3


class TestInstrumentPrecision:
    """Tests for instrument price and size precision."""

    def test_btcusd_price_precision(self) -> None:
        """BTCUSD has 1 decimal price precision."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert instrument.price_precision == 1

    def test_ethusd_price_precision(self) -> None:
        """ETHUSD has 2 decimal price precision."""
        instrument = create_ethereal_instrument("ETHUSD")
        assert instrument.price_precision == 2

    def test_solusd_price_precision(self) -> None:
        """SOLUSD has 3 decimal price precision."""
        instrument = create_ethereal_instrument("SOLUSD")
        assert instrument.price_precision == 3

    def test_btcusd_size_precision(self) -> None:
        """BTCUSD has 4 decimal size precision."""
        instrument = create_ethereal_instrument("BTCUSD")
        assert instrument.size_precision == 4

    def test_solusd_size_precision(self) -> None:
        """SOLUSD has 2 decimal size precision."""
        instrument = create_ethereal_instrument("SOLUSD")
        assert instrument.size_precision == 2
