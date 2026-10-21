"""Trade-ledger goldens: one fake run per scenario through BOTH tiers.

Owned by rabmo.2 L1 (mode-off G1-G13) and L2 (liquidation G8-G12): the shared fake
engine (``tests/unit/fake_engine.py``) and the base goldens that rabmo.2's ledger
refactors must leave bit-identical.

Each scenario runs the SAME positions/bars through
``NTScreeningRunner._extract_metrics`` and through validation's
``extract_results`` (twice: post-fill slippage active, and engine-modeled
slippage so the extraction charges none). The clock is frozen: screening
``time.time`` returns 1000.0 and ``start_time=1000.0``, validation's
``BacktestResult.elapsed_time`` is 0.0, so ``execution_time_seconds`` is 0.0 in
every captured result.

Capture (two IDENTICAL captures are required before rabmo.2 merges):

    VIBE_QUANT_CAPTURE_GOLDENS=1 .venv/bin/python -m pytest \
        tests/unit/test_tier_ledger_goldens.py -p no:randomly

Serialization (``_golden``): dataclass -> ``{field: _golden(value)}`` over
``dataclasses.fields()``; float -> ``repr``; tuple/list -> list; numpy array ->
``[repr(v)]``; ``DailyReturns`` -> ``{"start_day": int, "values": [repr]}``;
dict -> sorted keys. The file holds ``{"base": "a557a612", "cases": {...}}``.

Liquidation scenarios (G8-G12, rabmo.2 L2) run with ``liquidation_mode`` on and the
runner's/venue's catalog pointing at a real temp catalog of MARK bars, built by the
MARK-bar helpers of ``test_screening_liquidation`` (imported, not copied). Their
LAST execution bars are stamped at the minute close (as catalog bars are), so a
survivor has the valuation horizon both tiers mark it at.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide

from tests.unit.fake_engine import (
    FakeFill,
    FakePosition,
    fake_bt_result,
    fake_engine,
    fake_venue,
)
from tests.unit.test_screening_liquidation import CALM, CRASH, Bars, write_marks
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.metrics import DailyReturns
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.extraction import extract_results
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

BASE = "a557a612"
GOLDEN = Path(__file__).parent / "goldens" / "tier_ledger.json"
CAPTURE_ENV = "VIBE_QUANT_CAPTURE_GOLDENS"

MIN = 60_000_000_000
START_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
START_NS = START_MS * 1_000_000
DAY_MIN = 1440
N_BARS = 3 * DAY_MIN  # window 2025-01-01..2025-01-04
WINDOW_START, WINDOW_END = "2025-01-01", "2025-01-04"

BTC = "BTCUSDT-PERP.BINANCE"
ETH = "ETHUSDT-PERP.BINANCE"
BALANCE = 1000.0
FUNDING_RATE = 0.0003  # 3x the fallback, so G3's archived rate is visible
MODE_OFF_SCENARIOS = ["G1", "G2", "G3", "G4", "G5", "G5b", "G6", "G7", "G13"]
#: rabmo.2 L2: the liquidation goldens, each pinned with the mode ON in both tiers.
#: Every intended hit: liquidation_count >= the value here (0 = identity overlay).
LIQUIDATION_HITS = {"G8": 1, "G9": 1, "G10": 0, "G11": 1, "G12": 1}
SCENARIOS = [*MODE_OFF_SCENARIOS, *LIQUIDATION_HITS]


@pytest.fixture(autouse=True)
def _fresh_funding_cache() -> None:
    clear_rate_cache()


def at(minute: int) -> int:
    return START_NS + minute * MIN


@dataclass(frozen=True)
class _Inputs:
    positions: list[FakePosition]
    bars_by_type: dict[str, list[Any]] | None
    bt_result: Any
    runner: NTScreeningRunner
    funding_archive: Path
    balance: float = BALANCE
    liquidation_mode: str = "off"
    leverage: int = 10
    catalog_path: Path | None = None


def _bar_type(instrument_id: str) -> str:
    return f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"


def _flat_bars(instrument_id: str, closes: list[float] | None = None) -> list[Any]:
    """One 1m bar per minute of the window, close-only (open/high/low = close)."""
    bar_type = BarType.from_str(_bar_type(instrument_id))
    prices = closes if closes is not None else [40000.0] * N_BARS
    return [
        _Bar(bar_type, at(i), price) for i, price in enumerate(prices)
    ]


@dataclass(frozen=True)
class _Bar:
    bar_type: BarType
    ts_init: int
    price: float

    @property
    def open(self) -> float:
        return self.price

    @property
    def high(self) -> float:
        return self.price

    @property
    def low(self) -> float:
        return self.price

    @property
    def close(self) -> float:
        return self.price

    @property
    def volume(self) -> float:
        return 1000.0


@dataclass(frozen=True)
class _OHLCBar:
    """A LAST execution bar with real OHLC, stamped at the minute close like a catalog bar."""

    bar_type: BarType
    ts_init: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 1000.0


def _empty_archive(tmp_path: Path, name: str = "funding.db") -> Path:
    path = tmp_path / name
    archive = RawDataArchive(path)
    _ = archive.conn  # create the schema
    archive.close()
    return path


def _runner(
    symbols: list[str],
    funding_archive: Path,
    balance: int = int(BALANCE),
    *,
    liquidation_mode: str = "off",
    leverage: int = 10,
    catalog_path: Path | None = None,
) -> NTScreeningRunner:
    runner = NTScreeningRunner(
        dsl_dict={},
        symbols=symbols,
        start_date=WINDOW_START,
        end_date=WINDOW_END,
        catalog_path=str(catalog_path) if catalog_path is not None else None,
        funding_archive_path=str(funding_archive),
        liquidation_mode=liquidation_mode,  # type: ignore[arg-type]
        leverage=leverage,
        starting_balance=balance,
    )
    # Set during __call__ by a real run; the extractor reads them directly.
    runner._all_timeframes = {"1m"}
    runner._resolved_catalog_path = catalog_path
    return runner


def _inputs(
    positions: list[FakePosition],
    bars_by_type: dict[str, list[Any]] | None,
    tmp_path: Path,
    *,
    symbols: list[str],
    funding_archive: Path | None = None,
    balance: float = BALANCE,
    liquidation_mode: str = "off",
    leverage: int = 10,
    catalog_path: Path | None = None,
) -> _Inputs:
    archive = _empty_archive(tmp_path) if funding_archive is None else funding_archive
    return _Inputs(
        positions=positions,
        bars_by_type=bars_by_type,
        bt_result=fake_bt_result(total_positions=len(positions)),
        runner=_runner(
            symbols,
            archive,
            int(balance),
            liquidation_mode=liquidation_mode,
            leverage=leverage,
            catalog_path=catalog_path,
        ),
        funding_archive=archive,
        balance=balance,
        liquidation_mode=liquidation_mode,
        leverage=leverage,
        catalog_path=catalog_path,
    )


def _flat_trade(pid: str, open_min: int, close_min: int, realized: float) -> FakePosition:
    """A flat-price round trip carrying an explicit net PnL (PF/order witnesses)."""
    ts_opened, ts_closed = at(open_min), at(close_min)
    return FakePosition(
        id=pid,
        instrument_id=BTC,
        ts_opened=ts_opened,
        ts_closed=ts_closed,
        events=[
            FakeFill(ts_opened, OrderSide.BUY, 1.0, 1000.0),
            FakeFill(ts_closed, OrderSide.SELL, 1.0, 1000.0),
        ],
        realized_pnl=realized,
        entry=OrderSide.BUY,
        avg_px_open=1000.0,
        avg_px_close=1000.0,
        peak_qty=1.0,
        commission_list=[0.0, 0.0],
    )


def _liquidation_bars(bars: Bars) -> dict[str, list[Any]]:
    """LAST 1m bars mirroring the MARK bars, stamped at the minute close (catalog-style)."""
    bar_type = BarType.from_str(_bar_type(BTC))
    return {
        _bar_type(BTC): [
            _OHLCBar(bar_type, at(i) + MIN - 1_000_000, o, h, low, c)
            for i, (o, h, low, c) in enumerate(
                zip(bars.open, bars.high, bars.low, bars.close, strict=True)
            )
        ]
    }


def _liquidated_long(
    pid: str, open_min: int, close_min: int, fees: tuple[float, float] = (0.0, 0.0)
) -> FakePosition:
    """The 1 BTC long @40000 of the MARK scenarios: NT sells it at 38000 after the arena did."""
    return FakePosition.closed(
        pid,
        BTC,
        entry_ns=at(open_min),
        exit_ns=at(close_min),
        entry_price=40000.0,
        exit_price=38000.0,
        entry_fee=fees[0],
        exit_fee=fees[1],
    )


def _winner(pid: str, open_min: int, close_min: int, exit_price: float) -> FakePosition:
    return FakePosition.closed(
        pid,
        BTC,
        entry_ns=at(open_min),
        exit_ns=at(close_min),
        entry_price=40000.0,
        exit_price=exit_price,
    )


def _liquidation_scenario(name: str, tmp_path: Path, mode: str | None) -> _Inputs:
    """G8-G12 on the MARK bars of ``test_screening_liquidation`` (CRASH = 30000 at 07:30)."""
    if name == "G8":
        # Isolated hit: A is liquidated at 07:30, its NT close at 09:00 never happens.
        positions = [
            _liquidated_long("A", 420, 540, (20.0, 19.0)),
            _winner("B1", DAY_MIN + 600, DAY_MIN + 700, 40500.0),
            _winner("B2", DAY_MIN + 720, DAY_MIN + 780, 40200.0),
            _winner("B3", 2 * DAY_MIN + 600, 2 * DAY_MIN + 660, 39900.0),
        ]
        return _inputs(
            positions,
            _liquidation_bars(CRASH),
            tmp_path,
            symbols=["BTCUSDT"],
            balance=40000.0,
            liquidation_mode=mode or "isolated",
            catalog_path=write_marks(tmp_path, CRASH),
        )

    if name == "G9":
        # Cross bust: the 4000 wallet is wiped at 07:30 and equity never goes below 0.
        # (G12 adds a position that the bust drops; here the hit is the whole account.)
        positions = [_liquidated_long("A", 420, 540, (20.0, 19.0))]
        return _inputs(
            positions,
            _liquidation_bars(CRASH),
            tmp_path,
            symbols=["BTCUSDT"],
            balance=4000.0,
            liquidation_mode=mode or "cross",
            catalog_path=write_marks(tmp_path, CRASH),
        )

    if name == "G10":
        # Identity overlay: calm market, nothing truncated; metrics == mode off.
        positions = [
            _liquidated_long("A", 420, 540, (20.0, 19.0)),
            _winner("B1", DAY_MIN + 600, DAY_MIN + 700, 40500.0),
            _winner("B2", DAY_MIN + 720, DAY_MIN + 780, 40200.0),
            _winner("B3", 2 * DAY_MIN + 600, 2 * DAY_MIN + 660, 39900.0),
        ]
        return _inputs(
            positions,
            _liquidation_bars(CALM),
            tmp_path,
            symbols=["BTCUSDT"],
            balance=40000.0,
            liquidation_mode=mode or "cross",
            catalog_path=write_marks(tmp_path, CALM),
        )

    if name == "G11":
        # Survivor marked at the run end (40000) with its modeled exit fee.
        positions = [
            _liquidated_long("A", 420, 540, (20.0, 19.0)),
            FakePosition.open(
                "S", BTC, entry_ns=at(2 * DAY_MIN + 600), entry_price=40000.0, entry_fee=20.0
            ),
        ]
        return _inputs(
            positions,
            _liquidation_bars(CRASH),
            tmp_path,
            symbols=["BTCUSDT"],
            balance=40000.0,
            liquidation_mode=mode or "isolated",
            catalog_path=write_marks(tmp_path, CRASH),
        )

    if name == "G12":
        # Dropped open position: A is still open when the 4000 wallet busts at 07:30 and
        # B1 opens at 10:00; neither is ever marked (and B1 leaves no trade).
        positions = [
            FakePosition.open("A", BTC, entry_ns=at(420), entry_price=40000.0, entry_fee=20.0),
            FakePosition.open("B1", BTC, entry_ns=at(600), entry_price=40000.0),
        ]
        return _inputs(
            positions,
            _liquidation_bars(CRASH),
            tmp_path,
            symbols=["BTCUSDT"],
            balance=4000.0,
            liquidation_mode=mode or "cross",
            catalog_path=write_marks(tmp_path, CRASH),
        )

    raise AssertionError(f"unknown liquidation scenario {name}")


def _scenario(name: str, tmp_path: Path, mode: str | None = None) -> _Inputs:
    """The scenario fixtures; every trade stays inside one 8h funding block
    unless the scenario is G3 (which reads an archived settlement)."""
    if name in LIQUIDATION_HITS:
        return _liquidation_scenario(name, tmp_path, mode)
    one_btc = {_bar_type(BTC): _flat_bars(BTC)}

    if name == "G1":
        # Closed mix; P1 carries a rebate leg: screening sums |c|, validation
        # sums c then takes |total| -> 0.24000000000000002 vs 0.16.
        positions = [
            FakePosition.closed(
                "P1",
                BTC,
                entry_ns=at(600),
                exit_ns=at(630),
                entry_price=40000.0,
                exit_price=40500.0,
                entry_fee=0.2,
                exit_fee=-0.04,
            ),
            FakePosition.closed(
                "P2",
                BTC,
                entry_ns=at(720),
                exit_ns=at(760),
                entry_price=40500.0,
                exit_price=40700.0,
                qty=0.5,
                entry_side=OrderSide.SELL,
            ),
        ]
        return _inputs(positions, one_btc, tmp_path, symbols=["BTCUSDT"])

    if name == "G2":
        # Open mark: a=-0.26 (entry commission), u=-0.46, f=0.0455 (VIP0 taker).
        position = FakePosition.open(
            "P1",
            BTC,
            entry_ns=at(2 * DAY_MIN + 1000),
            entry_price=9146.0,
            qty=0.01,
            entry_fee=0.26,
        )
        closes = [9146.0] * (N_BARS - 1) + [9100.0]
        bars = {_bar_type(BTC): _flat_bars(BTC, closes)}
        return _inputs([position], bars, tmp_path, symbols=["BTCUSDT"])

    if name == "G3":
        # One archived settlement (16:00) inside the hold, on the close day.
        positions = [
            FakePosition.closed(
                "P1",
                BTC,
                entry_ns=at(600),
                exit_ns=at(1080),
                entry_price=40000.0,
                exit_price=40100.0,
            )
        ]
        archive = tmp_path / "funding_rates.db"
        data = RawDataArchive(archive)
        data.insert_funding_rates(
            "BTCUSDT", [(START_MS + 16 * 3_600_000, FUNDING_RATE, 40000.0)], "golden"
        )
        data.close()
        return _inputs(
            positions, one_btc, tmp_path, symbols=["BTCUSDT"], funding_archive=archive
        )

    if name == "G4":
        return _inputs([], None, tmp_path, symbols=["BTCUSDT"])

    if name == "G5":
        positions = [
            FakePosition.closed(
                "P1",
                BTC,
                entry_ns=at(600),
                exit_ns=at(630),
                entry_price=40000.0,
                exit_price=40050.0,
            ),
            FakePosition.closed(
                "P2",
                BTC,
                entry_ns=at(660),
                exit_ns=at(680),
                entry_price=40000.0,
                exit_price=39970.0,
            ),
            FakePosition.closed(
                "P3",
                BTC,
                entry_ns=at(720),
                exit_ns=at(730),
                entry_price=40000.0,
                exit_price=40020.0,
            ),
        ]
        return _inputs(positions, one_btc, tmp_path, symbols=["BTCUSDT"])

    if name == "G5b":
        # Four closed, one zero-notional: 3 usable returns of exactly 0.1.
        tenths = [
            FakePosition.closed(
                f"P{i}",
                BTC,
                entry_ns=at(600 + 40 * i),
                exit_ns=at(630 + 40 * i),
                entry_price=1000.0,
                exit_price=1100.0,
            )
            for i in range(3)
        ]
        zero = FakePosition(
            id="P4",
            instrument_id=BTC,
            ts_opened=at(760),
            ts_closed=at(790),
            events=[
                FakeFill(at(760), OrderSide.BUY, 0.0, 0.0),
                FakeFill(at(790), OrderSide.SELL, 0.0, 0.0),
            ],
            realized_pnl=0.0,
            entry=OrderSide.BUY,
            avg_px_open=0.0,
            avg_px_close=0.0,
            peak_qty=0.0,
            commission_list=[0.0, 0.0],
        )
        bars = {_bar_type(BTC): _flat_bars(BTC, [1000.0] * N_BARS)}
        return _inputs([*tenths, zero], bars, tmp_path, symbols=["BTCUSDT"])

    if name == "G6":
        # PF-order witness: cache order 0.1,0.2,0.3,-1.0; validation sorts
        # trades by entry time -> -1.0,0.3,0.2,0.1.
        positions = [
            _flat_trade("P1", 900, 930, 0.1),
            _flat_trade("P2", 800, 830, 0.2),
            _flat_trade("P3", 700, 730, 0.3),
            _flat_trade("P4", 600, 630, -1.0),
        ]
        bars = {_bar_type(BTC): _flat_bars(BTC, [1000.0] * N_BARS)}
        return _inputs(positions, bars, tmp_path, symbols=["BTCUSDT"])

    if name == "G7":
        # Two instruments: one BTC winner, one ETH short winner.
        positions = [
            FakePosition.closed(
                "P1",
                BTC,
                entry_ns=at(600),
                exit_ns=at(630),
                entry_price=40000.0,
                exit_price=40500.0,
            ),
            FakePosition.closed(
                "P2",
                ETH,
                entry_ns=at(660),
                exit_ns=at(690),
                entry_price=2500.0,
                exit_price=2450.0,
                qty=2.0,
                entry_side=OrderSide.SELL,
            ),
        ]
        bars = {
            _bar_type(BTC): _flat_bars(BTC),
            _bar_type(ETH): _flat_bars(ETH, [2500.0] * N_BARS),
        }
        return _inputs(positions, bars, tmp_path, symbols=["BTCUSDT", "ETHUSDT"])

    if name == "G13":
        # No bars of the execution timeframe: both DD fallbacks + their WARNINGs.
        positions = [
            FakePosition.closed(
                "P1",
                BTC,
                entry_ns=at(600),
                exit_ns=at(630),
                entry_price=40000.0,
                exit_price=40040.0,
            ),
            FakePosition.closed(
                "P2",
                BTC,
                entry_ns=at(660),
                exit_ns=at(680),
                entry_price=40000.0,
                exit_price=39975.0,
            ),
        ]
        return _inputs(positions, None, tmp_path, symbols=["BTCUSDT"])

    raise AssertionError(f"unknown scenario {name}")


def _golden(obj: Any) -> Any:
    """Deterministic JSON value of an extractor result (see the module docstring)."""
    if isinstance(obj, DailyReturns):
        return {"start_day": int(obj.first_day), "values": [repr(v) for v in obj.values]}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: _golden(getattr(obj, field.name)) for field in fields(obj)}
    if isinstance(obj, np.ndarray):
        return [repr(v) for v in obj.tolist()]
    if isinstance(obj, (float, np.floating)):  # np.float64 reprs as "np.float64(...)"
        return repr(float(obj))
    if isinstance(obj, (list, tuple)):
        return [_golden(value) for value in obj]
    if isinstance(obj, dict):
        return {str(key): _golden(obj[key]) for key in sorted(obj, key=str)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _observed(inputs: _Inputs, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr("vibe_quant.screening.nt_runner.time.time", lambda: 1000.0)
    screening = inputs.runner._extract_metrics(  # noqa: SLF001
        params={},
        bt_result=inputs.bt_result,
        engine=fake_engine(inputs.positions, inputs.bars_by_type),
        start_time=1000.0,
        starting_balance=inputs.balance,
    )
    cases: dict[str, Any] = {"screening": _golden(screening)}
    for label, slippage in (
        ("validation_post_fill", 0.0),
        ("validation_engine_slippage", 0.1),
    ):
        result = extract_results(
            run_id=1,
            strategy_name="golden",
            bt_result=inputs.bt_result,
            engine=fake_engine(inputs.positions, inputs.bars_by_type),
            venue_config=fake_venue(slippage, inputs.balance, inputs.leverage),
            primary_timeframe="1m",
            funding_calculator=FundingCalculator(inputs.funding_archive),
            run_start_date=WINDOW_START,
            run_end_date=WINDOW_END,
            execution_timeframe="1m",
            catalog_path=inputs.catalog_path,
            liquidation_mode=inputs.liquidation_mode,
        )
        cases[label] = _golden(result)
    return cases


def _capture(scenario: str, observed: dict[str, Any]) -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    if GOLDEN.exists():
        data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    else:
        data = {"base": BASE, "cases": {}}
    data["cases"][scenario] = observed
    GOLDEN.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _check_golden(scenario: str, observed: dict[str, Any]) -> None:
    if os.environ.get(CAPTURE_ENV) == "1":
        _capture(scenario, observed)
        return
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["base"] == BASE
    assert observed == golden["cases"][scenario]


def _without(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in keys}


@pytest.mark.parametrize("scenario", MODE_OFF_SCENARIOS)
def test_mode_off_goldens_match(
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inputs = _scenario(scenario, tmp_path)
    with caplog.at_level(logging.WARNING):
        observed = _observed(inputs, monkeypatch)

    if scenario == "G1":
        # R5: a rebate leg is summed per leg by screening, net by validation.
        assert observed["screening"]["total_fees"] == "0.24000000000000002"
        assert observed["validation_post_fill"]["total_fees"] == "0.16"
        assert observed["validation_engine_slippage"]["total_fees"] == "0.16"

    if scenario == "G13":
        # R13: both closed-trade fallbacks, warning texts verbatim.
        messages = [record.getMessage() for record in caplog.records]
        assert (
            "Mark-to-market drawdown unavailable for params {} (timeframe=1m) — "
            "using closed-trade drawdown, which ignores open-trade losses"
        ) in messages
        assert (
            "Mark-to-market drawdown unavailable (timeframe=1m) — "
            "using closed-trade drawdown, which ignores open-trade losses"
        ) in messages

    _check_golden(scenario, observed)


@pytest.mark.parametrize("scenario", list(LIQUIDATION_HITS))
def test_liquidation_goldens_match(
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R14/R17: the liquidation rebuilds of both tiers, pinned on the same fake run."""
    inputs = _scenario(scenario, tmp_path)
    with caplog.at_level(logging.WARNING):
        observed = _observed(inputs, monkeypatch)

    expected = LIQUIDATION_HITS[scenario]
    assert observed["screening"]["liquidation_count"] >= expected
    for label in ("validation_post_fill", "validation_engine_slippage"):
        assert observed[label]["notes"]["liquidation"]["count"] >= expected

    if scenario == "G10":
        # Identity overlay: the mode-on run keeps every mode-off number (the note and
        # the mode-on exit-timing diagnostic are the only additions).
        off = _observed(_scenario("G10", tmp_path, mode="off"), monkeypatch)
        assert observed["screening"]["liquidation_count"] == 0
        assert _without(observed["screening"], "exit_timing") == _without(
            off["screening"], "exit_timing"
        )
        for label in ("validation_post_fill", "validation_engine_slippage"):
            assert _without(observed[label], "notes", "liquidation_events") == _without(
                off[label], "notes", "liquidation_events"
            )

    _check_golden(scenario, observed)
