"""Funding avoidance follows each symbol's own settlement interval (vibe-quant-neuvu.5, T3).

Rule R15 under test (``aux_data.is_funding_settlement``): hour ``s`` is a
settlement iff ``s % (H(s) * 1h) == 0`` where ``H(s)`` is the archived interval
of the first row whose minute-rounded time is ``>= s`` (past the last row: the
last row's; no rows / no archive: 8 with ONE WARNING per (archive, symbol)).
R15b (causal gap rule): when that next row lies farther than its own interval
from ``s`` -- or ``s`` precedes every row -- the previous row's interval decides
(before the first row: 8h and the same WARNING); normal operation keeps R15.
The generated ``_is_near_funding_time`` is integer-ns and, on an all-8h symbol,
makes the same decision as the old hard-coded 00/08/16 ``datetime`` function
(R16, ``_legacy`` below is that function copied verbatim).
"""

from __future__ import annotations

import logging
import types
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from tests.unit.test_audit_dsl_medium import _filter_dsl, _Recorder
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError, is_funding_settlement
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.validation.funding import clear_rate_cache

if TYPE_CHECKING:
    from pathlib import Path

HOUR_NS = 3_600_000_000_000
MIN_NS = 60_000_000_000
HYPE = "HYPEUSDT-PERP.BINANCE"
BTC = "BTCUSDT-PERP.BINANCE"
SOL = "SOLUSDT-PERP.BINANCE"
LOGGER = "vibe_quant.dsl.aux_data"


def _ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp()) * 1_000_000_000


def _ms(iso: str) -> int:
    return _ns(iso) // 1_000_000


def _day_rows(day: str, step_h: int, days: int = 1, jitter_ms: int = 4) -> list[int]:
    """Archived funding times (ms, with exchange jitter) every ``step_h`` hours."""
    start = _ms(f"{day}T00:00:00")
    return [
        start + i * step_h * 3_600_000 + jitter_ms for i in range(days * 24 // step_h)
    ]


def _archive(path: Path, rows: dict[str, list[tuple[int, int]]]) -> Path:
    """tmp archive: ``{symbol: [(funding_time_ms, interval_hours), ...]}`` (rates + Vision)."""
    archive = RawDataArchive(path)
    for symbol, entries in rows.items():
        archive.insert_funding_rates(symbol, [(t, 0.0001) for t, _ in entries], "test")
        archive.insert_funding_intervals(symbol, entries and [(t, h) for t, h in entries], "test")
    archive.close()
    return path


def _configure(path: Path) -> None:
    clear_rate_cache()
    aux_data.configure(path)


@pytest.fixture(autouse=True)
def _fresh() -> Any:
    clear_rate_cache()
    yield
    clear_rate_cache()
    aux_data.configure(None)


def _hype_btc_archive(tmp_path: Path) -> Path:
    days = ("2025-05-31", "2025-06-01", "2025-06-02")
    hype = [(t, 4) for d in days for t in _day_rows(d, 4)]
    btc = [(t, 8) for d in days for t in _day_rows(d, 8)]
    return _archive(tmp_path / "a.db", {"HYPEUSDT": hype, "BTCUSDT": btc})


def _compile(before: int, after: int) -> Any:
    dsl = _filter_dsl(
        "fa_probe",
        avoid_around_funding={"enabled": True, "minutes_before": before, "minutes_after": after},
    )
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(dsl))
    return getattr(module, f"{_to_class_name('fa_probe')}Strategy")


def _method(before: int, after: int, instrument_id: str) -> Any:
    """The generated ``_is_near_funding_time`` bound to a stub of ``instrument_id``."""
    stub = types.SimpleNamespace(config=types.SimpleNamespace(instrument_id=instrument_id))
    return types.MethodType(_compile(before, after)._is_near_funding_time, stub)


def _legacy(ts_ns: int, before: int, after: int) -> bool:
    """The OLD generated ``_is_near_funding_time`` body (00/08/16 UTC), verbatim."""
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=UTC)
    hour = dt.hour
    minute = dt.minute
    funding_hours = [0, 8, 16]
    for fh in funding_hours:
        minutes_before = before
        minutes_after = after
        if hour == fh and minute < minutes_after:
            return True
        prev_hour = (fh - 1) % 24
        if hour == prev_hour and minute >= (60 - minutes_before):
            return True
    return False


def test_hype_4h_settlement_blocks_entry_at_0403(tmp_path: Path) -> None:
    """R17: HYPE settles every 4h, so 04:03 is blocked; BTC (8h) at the same instant is not."""
    _configure(_hype_btc_archive(tmp_path))
    hype, btc = _method(5, 5, HYPE), _method(5, 5, BTC)
    ts = _ns("2025-06-01T04:03:00")

    assert hype(ts) is True  # legacy 00/08/16 allowed it
    assert btc(ts) is False
    # half-open edges around the 04:00 settlement, 5 min each side
    assert hype(_ns("2025-06-01T04:05:00")) is False
    assert hype(_ns("2025-06-01T04:04:00")) is True
    assert hype(_ns("2025-06-01T03:55:00")) is True
    assert hype(_ns("2025-06-01T03:54:00")) is False
    # nanosecond edges of both windows (window start inclusive, end exclusive)
    s4 = _ns("2025-06-01T04:00:00")
    assert hype(s4 - 5 * MIN_NS - 1) is False
    assert hype(s4 - 5 * MIN_NS) is True
    assert hype(s4 + 5 * MIN_NS - 1) is True
    assert hype(s4 + 5 * MIN_NS) is False
    # a clearly off-grid minute is open for both
    assert hype(_ns("2025-06-01T05:30:00")) is False


def test_settlement_rule_rows(tmp_path: Path) -> None:
    """R15: H(s) is the interval of the first minute-rounded row >= s; edges both ways."""
    d = "2025-06-01"
    path = _archive(
        tmp_path / "a.db",
        {
            "SOLUSDT": [
                (_ms(f"{d}T00:00:00") + 4, 4),
                (_ms(f"{d}T04:00:00") + 4, 4),
                (_ms(f"{d}T07:59:59") + 996, 4),  # 07:59:59.996 rounds to 08:00
                (_ms(f"{d}T11:59:59") + 996, 6),  # rounds onto 12:00
                (_ms(f"{d}T16:00:00") + 4, 8),
            ],
        },
    )
    _configure(path)

    def settles(hh: int, mm: int = 0) -> bool:
        return is_funding_settlement(SOL, _ns(f"{d}T{hh:02d}:{mm:02d}:00"))

    assert settles(4) is True  # row 04:00 itself: ">=" is inclusive (a ">" would pick 8h)
    assert settles(8) is True  # the 07:59:59.996 row rounds onto 08:00 and is a 4h row
    assert settles(12) is True  # the 11:59:59.996 row rounds onto 12:00 and is a 6h row
    assert settles(10) is False  # first row >= 10:00 is the 6h row; 10 % 6 != 0
    assert settles(14) is False  # first row >= 14:00 is 16:00 (8h)
    assert settles(16) is True  # 16 % 8 == 0
    assert settles(2) is False  # first row >= 02:00 is the 04:00 4h row; 2 % 4 != 0
    assert settles(0) is True
    assert settles(5) is False  # off the grid for any of the intervals
    # past the last row (16:00, 8h): the last row's interval
    assert is_funding_settlement(SOL, _ns("2025-06-03T08:00:00")) is True
    assert is_funding_settlement(SOL, _ns("2025-06-03T04:00:00")) is False
    # before the first row (00:00): no causal interval, so the 8h fallback (R15b)
    assert is_funding_settlement(SOL, _ns("2025-05-20T04:00:00")) is False
    assert is_funding_settlement(SOL, _ns("2025-05-20T08:00:00")) is True


@pytest.mark.parametrize("before_after", [(0, 0), (5, 5), (15, 0), (0, 15), (60, 60)])
def test_8h_symbol_matches_legacy_hours(tmp_path: Path, before_after: tuple[int, int]) -> None:
    """R16: BTC (all 8h), every 1m close of 2025-06-01 -> new decision == legacy decision."""
    before, after = before_after
    _configure(_hype_btc_archive(tmp_path))
    method = _method(before, after, BTC)
    day = _ns("2025-06-01T00:00:00")
    blocked = 0
    for k in range(1, 1441):
        ts = day + k * MIN_NS
        expected = _legacy(ts, before, after)
        assert method(ts) is expected, (before, after, ts)
        blocked += expected
    # not vacuous: the legacy window blocks exactly its size per settlement (24h = 3 settlements)
    assert blocked == 3 * (min(before, 60) + min(after, 60))


def test_generated_method_is_integer_ns_and_uses_lookup() -> None:
    src = StrategyCompiler().compile(
        validate_strategy_dict(_filter_dsl("fa_src", avoid_around_funding={"enabled": True}))
    )
    assert "is_funding_settlement" in src
    assert "funding_hours" not in src
    assert "fromtimestamp" not in src.split("def _is_near_funding_time")[1].split("\n    def ")[0]


def test_each_instance_uses_own_symbol_grid(tmp_path: Path) -> None:
    """R23: BTC+HYPE strategies side by side at 04:00 -> BTC allowed, HYPE blocked."""
    _configure(_hype_btc_archive(tmp_path))
    ts = _ns("2025-06-01T04:00:00")
    btc, hype = _method(5, 5, BTC), _method(5, 5, HYPE)

    assert (btc(ts), hype(ts)) == (False, True)
    # and at 08:00 both settle
    ts8 = _ns("2025-06-01T08:00:00")
    assert (btc(ts8), hype(ts8)) == (True, True)


def test_sol_interval_change(tmp_path: Path) -> None:
    """R19: SOL 2h in 2022-11, 8h in 2023-01: 06:00 blocked in 2022, 04:00 allowed in 2023."""
    sol = [(t, 2) for t in _day_rows("2022-11-12", 2, jitter_ms=3)]
    sol += [(t, 8) for t in _day_rows("2023-01-01", 8, jitter_ms=3)]
    _configure(_archive(tmp_path / "a.db", {"SOLUSDT": sol}))
    method = _method(5, 5, SOL)

    assert method(_ns("2022-11-12T06:00:00")) is True
    assert method(_ns("2022-11-12T05:57:00")) is True
    assert method(_ns("2022-11-12T05:30:00")) is False
    assert method(_ns("2023-01-01T04:00:00")) is False
    assert method(_ns("2023-01-01T08:00:00")) is True


def test_gap_uses_previous_rows_interval(tmp_path: Path) -> None:
    """R15b: inside an archive gap the previous row's interval decides (causal).

    SOL archives 2h settlements through 2022-11-12, then 8h from 2023-01-01. A
    2022-11-13 06:00 hour lies weeks before the next row (8h), so the previous
    2h interval applies: 06:00 is a settlement.
    """
    sol = [(t, 2) for t in _day_rows("2022-11-12", 2, jitter_ms=3)]
    sol += [(t, 8) for t in _day_rows("2023-01-01", 8, jitter_ms=3)]
    _configure(_archive(tmp_path / "a.db", {"SOLUSDT": sol}))
    method = _method(5, 5, SOL)

    assert is_funding_settlement(SOL, _ns("2022-11-13T06:00:00")) is True  # next row 8h would say False
    assert is_funding_settlement(SOL, _ns("2022-11-13T05:30:00")) is False
    assert method(_ns("2022-11-13T06:00:00")) is True
    assert method(_ns("2022-11-13T05:57:00")) is True  # 3 min before the 2h settlement
    assert method(_ns("2022-11-13T05:30:00")) is False


def test_gap_uses_previous_interval_when_next_is_shorter(tmp_path: Path) -> None:
    """R15b: the gap interval is the previous one even when the next row is shorter.

    8h rows through 2022-11-12, then 2h rows from 2023-01-01: 2022-11-13 02:00 is
    not a settlement on the previous 8h grid (a next-row 2h would call it one).
    """
    sol = [(t, 8) for t in _day_rows("2022-11-12", 8, jitter_ms=3)]
    sol += [(t, 2) for t in _day_rows("2023-01-01", 2, jitter_ms=3)]
    _configure(_archive(tmp_path / "a.db", {"SOLUSDT": sol}))
    method = _method(5, 5, SOL)

    assert is_funding_settlement(SOL, _ns("2022-11-13T02:00:00")) is False  # next row 2h would say True
    assert is_funding_settlement(SOL, _ns("2022-11-13T04:00:00")) is False
    assert is_funding_settlement(SOL, _ns("2022-11-13T08:00:00")) is True
    assert method(_ns("2022-11-13T04:00:00")) is False


def test_gap_boundary_uses_next_row_within_its_interval(tmp_path: Path) -> None:
    """R15b edge: ``next_row - s == H(next)`` is normal operation -> the next row's interval.

    8h rows through 2022-11-12 16:00, next row 2022-11-13 00:00 with 4h: 20:00 is
    exactly one next-interval before it and settles on the 4h grid (8h would not),
    while 19:00 is farther than that and falls back to the previous 8h row.
    """
    sol = [(t, 8) for t in _day_rows("2022-11-12", 8, jitter_ms=3)]
    sol += [(t, 4) for t in _day_rows("2022-11-13", 4, jitter_ms=3)]
    _configure(_archive(tmp_path / "a.db", {"SOLUSDT": sol}))

    assert is_funding_settlement(SOL, _ns("2022-11-12T20:00:00")) is True  # == one 4h interval
    assert is_funding_settlement(SOL, _ns("2022-11-12T19:00:00")) is False  # > 4h -> previous 8h
    assert is_funding_settlement(SOL, _ns("2022-11-12T16:00:00")) is True  # the 8h row itself


def test_before_first_row_uses_8h_and_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R15b: before the first row -> the 8h grid, not the future row's 4h, one WARNING."""
    d = "2025-06-01"
    path = _archive(
        tmp_path / "a.db",
        {"SOLUSDT": [(_ms(f"{d}T04:00:00") + 4, 4), (_ms(f"{d}T08:00:00") + 4, 4)]},
    )
    _configure(path)
    method = _method(5, 5, SOL)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        # 2025-05-20 04:00 is on the future row's 4h grid but precedes all data:
        # the 8h fallback allows it (the old lookup blocked it).
        assert is_funding_settlement(SOL, _ns("2025-05-20T04:00:00")) is False
        assert method(_ns("2025-05-20T04:00:00")) is False
        assert is_funding_settlement(SOL, _ns("2025-05-20T08:00:00")) is True
        assert method(_ns("2025-05-20T08:00:00")) is True
        assert method(_ns("2025-05-20T06:00:00")) is False
        # the first row itself keeps its own interval (exact-row inclusivity)
        assert is_funding_settlement(SOL, _ns("2025-06-01T04:00:00")) is True
    warnings = [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "SOLUSDT" in warnings[0].getMessage()
    assert "8h" in warnings[0].getMessage()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert is_funding_settlement(SOL, _ns("2025-06-01T12:00:00")) is True  # past last row: 4h, no warning
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_missing_symbol_falls_back_to_8h(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """R20: symbol absent from the archive -> 8h decisions, exactly one WARNING naming it."""
    _configure(_hype_btc_archive(tmp_path))
    method = _method(5, 5, "DOGEUSDT-PERP.BINANCE")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert method(_ns("2025-06-01T08:02:00")) is True
        assert method(_ns("2025-06-01T04:03:00")) is False
        assert method(_ns("2025-06-01T15:58:00")) is True
        assert method(_ns("2025-06-01T09:00:00")) is False
    warnings = [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "DOGEUSDT" in warnings[0].getMessage()


def test_missing_archive_falls_back_to_8h(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """R21: archive file missing -> 8h, one WARNING, no AuxDataUnavailableError, no file created."""
    missing = tmp_path / "nope.db"
    _configure(missing)
    method = _method(5, 5, HYPE)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        try:
            allowed = method(_ns("2025-06-01T04:03:00"))
            blocked = method(_ns("2025-06-01T00:01:00"))
        except AuxDataUnavailableError:  # pragma: no cover - the failure being tested
            pytest.fail("on_bar path raised AuxDataUnavailableError")
    assert (allowed, blocked) == (False, True)
    warnings = [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "HYPEUSDT" in warnings[0].getMessage()
    assert not missing.exists()


def test_new_run_warns_again_after_configure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _hype_btc_archive(tmp_path)
    ts = _ns("2025-06-01T08:00:00")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _configure(path)
        is_funding_settlement("DOGEUSDT-PERP.BINANCE", ts)
        is_funding_settlement("DOGEUSDT-PERP.BINANCE", ts)
        _configure(path)
        is_funding_settlement("DOGEUSDT-PERP.BINANCE", ts)
    assert len([r for r in caplog.records if r.name == LOGGER]) == 2


def test_lookup_error_falls_back_without_leaking_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unexpected loader failure never reaches on_bar: 8h rule, class name logged once, no text."""
    from vibe_quant.validation.funding import FundingCalculator

    _configure(_hype_btc_archive(tmp_path))

    def boom(self: object, symbol: str) -> None:
        raise RuntimeError("SECRET-MARKER-7731")

    monkeypatch.setattr(FundingCalculator, "_load_rates", boom)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert is_funding_settlement(HYPE, _ns("2025-06-01T08:00:00")) is True
        assert is_funding_settlement(HYPE, _ns("2025-06-01T04:00:00")) is False
    assert "SECRET-MARKER-7731" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert len([r for r in caplog.records if r.name == LOGGER]) == 1


def test_cache_follows_clear_rate_cache(tmp_path: Path) -> None:
    """The per-symbol grid is revalidated against the rate cache: a rewritten archive is seen."""
    path = tmp_path / "a.db"
    _archive(path, {"HYPEUSDT": [(t, 8) for t in _day_rows("2025-06-01", 8)]})
    _configure(path)
    ts = _ns("2025-06-01T04:00:00")
    assert is_funding_settlement(HYPE, ts) is False
    _archive(path, {"HYPEUSDT": [(t, 4) for t in _day_rows("2025-06-01", 4)][1:]})  # append-only
    assert is_funding_settlement(HYPE, ts) is False  # cached rate list, no stale-read surprise
    clear_rate_cache()
    assert is_funding_settlement(HYPE, ts) is True


# -- R18: real engine run, 1h bars on HYPE ------------------------------------------------


def _hype_hour_bars(count: int) -> list[Bar]:
    t0 = _ns("2024-01-01T00:00:00")
    bt = BarType.from_str(f"{HYPE}-1-HOUR-LAST-EXTERNAL")
    return [
        Bar(
            bar_type=bt,
            open=Price.from_str("100.000"),
            high=Price.from_str("100.500"),
            low=Price.from_str("99.500"),
            close=Price.from_str("100.000"),
            volume=Quantity.from_str("10.00"),
            ts_event=t0 + i * HOUR_NS,
            ts_init=t0 + (i + 1) * HOUR_NS - 1_000_000,
        )
        for i in range(count)
    ]


def _run_hype(dsl: dict[str, Any], bars: list[Bar]) -> Any:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    engine.add_instrument(create_instrument("HYPEUSDT"))
    engine.add_data(bars)
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(dsl))
    cn = _to_class_name(dsl["name"])
    base = getattr(module, f"{cn}Strategy")
    cls = type("Probe", (_Recorder, base), {})
    strat = cls(getattr(module, f"{cn}Config")(instrument_id=HYPE))
    engine.add_strategy(strat)
    engine.run()
    engine.dispose()
    return strat


def test_hype_1h_run_blocks_six_closes(tmp_path: Path) -> None:
    """R18: a HYPE 1h strategy never enters at the six 4h-grid closes (old code: only 3)."""
    hype = [(t, 4) for d in ("2023-12-31", "2024-01-01", "2024-01-02") for t in _day_rows(d, 4)]
    _configure(_archive(tmp_path / "a.db", {"HYPEUSDT": hype}))
    dsl = _filter_dsl("fa_hype", avoid_around_funding={"enabled": True})
    strat = _run_hype(dsl, _hype_hour_bars(48))

    assert {0, 4, 8, 12, 16, 20}.isdisjoint(strat.entry_hours)
    assert {1, 5, 9, 13, 17, 21} <= set(strat.entry_hours)
