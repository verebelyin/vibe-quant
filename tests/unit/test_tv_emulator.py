"""The TradingView broker emulator against the real screening engine (R60, R61) and by hand.

``scripts/research/tv_ports/tv_emulator.py`` replays the strategy's OWN signals through a broker.
In ``engine`` mode its trades must equal the screening runner's, fill for fill: the same
``(ts, side, action)`` AND quantity, price, commission, the protective levels that reached the
venue, and the costed equity after every closed trade. The fixtures run on real BTCUSDT catalog
bars; the by-hand tests use tiny synthetic catalogs whose bars are shaped to hit one rule.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.unit.test_reverse_runners import (
    _END as FLAT_END,
)
from tests.unit.test_reverse_runners import (
    _REVERSAL_NS,
    _flat_catalog,
    _funding_archive,
    reverse_dsl,
    run_screening,
)
from tests.unit.test_screening_fill_mode import _START as FLAT_START
from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.screening import nt_runner
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.sizing_settings import SizingSettings
from vibe_quant.validation.extraction import all_positions

REPO = Path(__file__).resolve().parents[2]
TV_DIR = REPO / "scripts" / "research" / "tv_ports"
CATALOG_BARS = (
    REPO / "data" / "catalog" / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
)
needs_catalog = pytest.mark.skipif(not CATALOG_BARS.is_dir(), reason="needs the BTCUSDT 1m catalog")


def _load(name: str) -> ModuleType:
    """Import ``scripts/research/tv_ports/<name>.py`` by path (vibe_quant never imports scripts/research)."""
    if str(TV_DIR) not in sys.path:
        sys.path.insert(0, str(TV_DIR))
    spec = importlib.util.spec_from_file_location(name, TV_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


te = _load("tv_emulator")

# ---------------------------------------------------------------------------------------------
# the engine side of an E-parity comparison
# ---------------------------------------------------------------------------------------------


class Engine:
    """What the real screening runner did, read from its engine cache."""

    def __init__(self) -> None:
        self.fills: list[Any] = []
        self.levels: list[tuple[str, str, str]] = []
        self.positions: list[Any] = []
        self.ledger: Any = None
        self.metrics: Any = None
        self.orders: list[Any] = []


def run_engine(
    monkeypatch: pytest.MonkeyPatch,
    dsl: dict[str, Any],
    start: str,
    end: str,
    **runner_kwargs: Any,
) -> Engine:
    """The screening runner over [start, end) with its positions, orders and cost ledger captured."""
    out = Engine()
    original = NTScreeningRunner._extract_metrics

    def spy(self: NTScreeningRunner, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        out.positions = all_positions(engine)
        out.orders = list(engine.cache.orders())
        out.ledger = k.get("cost_ledger")
        return original(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "_extract_metrics", spy)
    out.metrics = NTScreeningRunner(dsl, ["BTCUSDT"], start, end, **runner_kwargs)({})
    assert out.metrics.sharpe_ratio != float("-inf"), "the engine run failed"
    kinds = {"MARKET": "exit", "STOP_MARKET": "stop", "LIMIT": "target"}
    for pos in sorted(out.positions, key=lambda p: int(p.ts_opened)):
        for i, f in enumerate(pos.events):
            action = "entry" if i == 0 else kinds[f.order_type.name]
            out.fills.append(
                te.Fill(
                    int(f.ts_event), f.order_side.name, action, str(f.last_qty), str(f.last_px),
                    str(f.commission),
                )
            )
    protective = [
        o
        for o in out.orders
        if o.is_reduce_only
        and o.order_type.name in ("STOP_MARKET", "LIMIT")
        and o.status.name != "REJECTED"  # a reduce-only order with nothing to reduce never lives
    ]
    for o in sorted(protective, key=lambda o: (int(o.ts_init), o.order_type.name != "STOP_MARKET")):
        stop = o.order_type.name == "STOP_MARKET"
        out.levels.append((o.order_type.name, str(o.trigger_price if stop else o.price), str(o.quantity)))
    return out


def assert_e_parity(engine: Engine, emu: Any, *, min_trades: int = 3) -> None:
    """R60: same (ts, side, action) and qty, prices, levels, commissions and costed equity."""
    assert len([p for p in engine.positions if p.is_closed]) >= min_trades
    assert [(f.ts, f.side, f.action) for f in emu.fills] == [
        (f.ts, f.side, f.action) for f in engine.fills
    ]
    assert emu.fills == engine.fills  # + qty, price, commission, exactly
    assert [(lv.kind, lv.price, lv.qty) for lv in emu.levels] == engine.levels
    assert emu.costed_equity() == te.costed_equity_series(
        engine.positions, engine.ledger, emu.starting_balance
    )
    assert emu.costed_equity()  # a money fixture closes trades


def emulate(dsl: dict[str, Any], start: str, end: str, mode: str = "engine", **kw: Any) -> Any:
    runner = NTScreeningRunner(dsl, ["BTCUSDT"], start, end, **kw)
    return te.emulate(runner, "BTCUSDT", mode)


# ---------------------------------------------------------------------------------------------
# R60: engine vs emulator on real bars
# ---------------------------------------------------------------------------------------------

LEVEL_LONG = {
    "stop_loss_long": {"type": "signal_level", "indicator": "dc.lower"},
    "take_profit_long": {"type": "signal_level", "indicator": "dc.upper"},
}
LEVEL_BOTH = {
    **LEVEL_LONG,
    "stop_loss_short": {"type": "signal_level", "indicator": "dc.upper"},
    "take_profit_short": {"type": "signal_level", "indicator": "dc.lower"},
}


def money_fixtures() -> dict[str, dict[str, Any]]:
    """flat-only, reverse, signal_level and reverse + signal_level + refresh strategies."""
    wide = {
        "stop_loss": {"type": "fixed_pct", "percent": 20.0},
        "take_profit": {"type": "fixed_pct", "percent": 90.0},
    }
    dc_rsi = {"dc": {"type": "DONCHIAN", "period": 20}, "rsi": {"type": "RSI", "period": 14}}
    return {
        "plain": {
            "name": "e_plain",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi crosses_below 30"], "short": ["rsi crosses_above 70"]},
            "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 45"]},
            "stop_loss": {"type": "fixed_pct", "percent": 1.0},
            "take_profit": {"type": "fixed_pct", "percent": 2.0},
        },
        "reverse": {
            "name": "e_reverse",
            "position_mode": "reverse",
            "indicators": {"fast": {"type": "EMA", "period": 9}, "slow": {"type": "EMA", "period": 21}},
            "entry_conditions": {
                "long": ["fast crosses_above slow"],
                "short": ["fast crosses_below slow"],
            },
            "stop_loss": {"type": "fixed_pct", "percent": 1.5},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        },
        "signal_level": {
            "name": "e_level",
            "indicators": dc_rsi,
            "entry_conditions": {"long": ["rsi crosses_above 45"]},
            "exit_conditions": {"long": ["rsi > 70"]},
            **wide,
            **LEVEL_LONG,
        },
        "reverse_level_refresh": {
            "name": "e_revlevel",
            "position_mode": "reverse",
            "refresh_on_signal": True,
            "indicators": dc_rsi,
            "entry_conditions": {
                "long": ["rsi crosses_above 45"],
                "short": ["rsi crosses_below 55"],
            },
            **wide,
            **LEVEL_BOTH,
        },
    }


@needs_catalog
@pytest.mark.parametrize("name", ["plain", "reverse", "signal_level", "reverse_level_refresh"])
@pytest.mark.parametrize("timeframe", ["1h", "15m"])
def test_engine_mode_equals_screening_on_real_bars(
    name: str, timeframe: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R60 (outbox modes): a month of BTCUSDT; every fill, level and costed equity equal."""
    dsl = {**money_fixtures()[name], "timeframe": timeframe}
    engine = run_engine(monkeypatch, dsl, "2026-03-02", "2026-03-30")
    assert_e_parity(engine, emulate(dsl, "2026-03-02", "2026-03-30"))


@needs_catalog
@pytest.mark.parametrize("name", ["plain", "reverse", "reverse_level_refresh"])
def test_engine_mode_equals_screening_on_1m_defer(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """R60 (1m: every action deferred one bar, no outbox, protection live at once)."""
    dsl = {**money_fixtures()[name], "timeframe": "1m"}
    engine = run_engine(monkeypatch, dsl, "2026-03-02", "2026-03-04")
    assert_e_parity(engine, emulate(dsl, "2026-03-02", "2026-03-04"))


@needs_catalog
def test_a_protective_order_already_marketable_at_release_fills_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R60 edge: 5m signal-level protection released on a tick that is already past a level.

    A stop whose trigger the release tick has passed fills on that very tick at the tick price, and
    a limit already through its price fills as a taker at the book; both equal the engine's fills.
    """
    dsl = {**money_fixtures()["reverse_level_refresh"], "timeframe": "5m"}
    engine = run_engine(monkeypatch, dsl, "2026-03-12", "2026-03-15")
    emu = emulate(dsl, "2026-03-12", "2026-03-15")
    assert_e_parity(engine, emu)
    release_ts = {lv.ts for lv in emu.levels}
    at_release = [f for f in emu.fills if f.action == "stop" and f.ts in release_ts]
    assert at_release, "the window must contain a stop that fills on its own release tick"


# ---------------------------------------------------------------------------------------------
# synthetic hourly catalog: shape ONE bar to hit one rule
# ---------------------------------------------------------------------------------------------

_MIN_MS = 60_000
_HOUR_MS = 3_600_000
_DAY0_MS = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()) * 1000
_DAY0_NS = _DAY0_MS * 1_000_000


def hour_catalog(root: Path, hours: list[tuple[float, float, float, float, float]]) -> Path:
    """1m + 1h bars from hourly (open, high, low, close, first-minute close) tuples.

    The hour's 1m path is ``open -> first-minute close (twice), high, low, close ...``, so its
    aggregated 1h bar is exactly (open, high, low, close) whatever the first-minute close is, and
    the two fill ticks of the hour boundary (the closes of its first two minutes) are both that
    first-minute close.
    """
    manager = CatalogManager(root)
    instrument = create_instrument("BTCUSDT")
    manager.write_instrument(instrument)
    klines: list[Any] = []
    for h, (o, hi, lo, c, m0) in enumerate(hours):
        path = [m0, m0, hi, lo] + [c] * 56
        prev = o
        for m, close in enumerate(path):
            open_ms = _DAY0_MS + h * _HOUR_MS + m * _MIN_MS
            klines.append(
                {
                    "open_time": open_ms,
                    "close_time": open_ms + _MIN_MS - 1,
                    "open": prev,
                    "high": max(prev, close),
                    "low": min(prev, close),
                    "close": close,
                    "volume": 1.0,
                }
            )
            prev = close
    bars_1m = klines_to_bars(
        klines, instrument.id, get_bar_type("BTCUSDT", "1m"),
        instrument.size_precision, instrument.price_precision,
    )
    manager.write_bars(bars_1m)
    manager.write_bars(
        aggregate_bars(
            bars_1m, get_bar_type("BTCUSDT", "1h"), 60,
            instrument.size_precision, instrument.price_precision,
        )
    )
    return root


def _scenario_dsl() -> dict[str, Any]:
    """Long when the 1h close is above 100.5; 2 % stop and 3 % target; nothing else."""
    return {
        "name": "tv_scenario",
        "timeframe": "1h",
        "indicators": {},
        "entry_conditions": {"long": ["close > 100.5"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


def _flat(n: int) -> list[tuple[float, float, float, float, float]]:
    return [(100.0, 100.0, 100.0, 100.0, 100.0)] * n


def _both_touched(high: float, low: float) -> list[tuple[float, float, float, float, float]]:
    """Hour 1 closes at 101 (the signal); hour 2 opens 101 and spans ``low``..``high``."""
    return [
        (100.0, 100.0, 100.0, 100.0, 100.0),
        (100.0, 101.0, 100.0, 101.0, 101.0),
        (101.0, high, low, 100.0, 101.0),
        *_flat(21),
    ]


_RUN_START, _RUN_END = "2024-01-01", "2024-01-02"
_SIZING = LaunchSettings(sizing=SizingSettings(max_margin_pct=None))


@pytest.mark.parametrize(
    ("high", "low", "first"),
    [
        # |H-O| = 3.6 < |L-O| = 4.0: the high is reached first -> the 3 % target (104.0) fills first
        (104.6, 97.0, "target"),
        # |L-O| = 2.5 < |H-O| = 5.0: the low is reached first -> the 2 % stop (99.0) fills first
        (106.0, 98.5, "stop"),
        # a tie goes to the low
        (105.0, 97.0, "stop"),
    ],
)
@pytest.mark.parametrize("mode", ["engine", "tv"])
def test_both_levels_touched_in_one_bar_follow_the_engines_tick_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, high: float, low: float, first: str, mode: str
) -> None:
    """R61: the bar that touches stop AND target exits by the engine's tick order, in both modes."""
    catalog = hour_catalog(tmp_path / "catalog", _both_touched(high, low))
    emu = emulate(
        _scenario_dsl(), _RUN_START, _RUN_END, mode=mode,
        catalog_path=str(catalog), launch=_SIZING,
    )
    assert [f.action for f in emu.fills] == ["entry", first]
    entry, exit_fill = emu.fills
    assert (entry.side, exit_fill.side) == ("BUY", "SELL")
    assert entry.qty == exit_fill.qty == "4.950"  # 20 USDT risk / 2.02 stop, capped at 50 % of 1000
    stop_px, target_px = (lv.price for lv in emu.levels)
    assert (emu.levels[0].kind, emu.levels[1].kind) == ("STOP_MARKET", "LIMIT")
    entry_px = float(entry.price)
    assert float(stop_px) == pytest.approx(round(entry_px * 0.98, 1), abs=1e-9)
    assert float(target_px) == pytest.approx(round(entry_px * 1.03, 1), abs=1e-9)
    if mode == "engine":
        # the real runner takes the same leg first, at the same prices and costs
        engine = run_engine(
            monkeypatch, _scenario_dsl(), _RUN_START, _RUN_END,
            catalog_path=str(catalog), launch=_SIZING,
        )
        assert [f.action for f in engine.fills] == ["entry", first]
        assert_e_parity(engine, emu, min_trades=1)


@pytest.fixture
def gap_catalog(tmp_path: Path) -> Path:
    """Hour 1 closes 101 (signal); hour 2 GAPS to open 102.4 and its first minute closes 102.7."""
    hours = [
        (100.0, 100.0, 100.0, 100.0, 100.0),
        (100.0, 101.0, 100.0, 101.0, 101.0),
        (102.4, 102.8, 102.2, 102.5, 102.7),
        *_flat(21),
    ]
    return hour_catalog(tmp_path / "catalog", hours)


def test_tv_mode_fills_at_the_next_bar_open_and_engine_mode_at_the_release_tick(
    gap_catalog: Path,
) -> None:
    """TV fills the signal at the next bar's OPEN (102.4, no slippage); the engine at that hour's
    first 1m close (102.7, one tick of slippage possible) one minute and 59.999 s into the hour."""
    kw: dict[str, Any] = {"catalog_path": str(gap_catalog), "launch": _SIZING}
    tv = emulate(_scenario_dsl(), _RUN_START, _RUN_END, mode="tv", **kw)
    engine = emulate(_scenario_dsl(), _RUN_START, _RUN_END, mode="engine", **kw)

    (tv_entry, *_) = tv.fills
    (engine_entry, *_) = engine.fills
    hour2_open_ns = _DAY0_NS + 2 * _HOUR_MS * 1_000_000
    assert tv_entry.action == engine_entry.action == "entry"
    assert (tv_entry.price, tv_entry.ts) == ("102.4", hour2_open_ns)
    assert engine_entry.ts == hour2_open_ns + _MIN_MS * 1_000_000 - 1_000_000
    assert engine_entry.price in ("102.6", "102.7", "102.8")
    # TradingView sizes on the signal bar's close (101): the same 4.950 BTC in both modes
    assert tv_entry.qty == engine_entry.qty == "4.950"
    # protection comes from the actual fill: 2 % / 3 % off 102.4 -> 100.352 -> 100.4, 105.472 -> 105.5
    assert [(lv.kind, lv.price) for lv in tv.levels] == [("STOP_MARKET", "100.4"), ("LIMIT", "105.5")]
    # no slippage and taker commission in TV mode: 0.05 % of 4.950 x 102.4
    assert tv_entry.commission.startswith("0.25344")


# ---------------------------------------------------------------------------------------------
# TV: the opening of a bar is processed at the state available BEFORE the bar (no look-ahead)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("high", "low"), [(150.0, 50.0), (101.5, 100.5)])
def test_tv_atr_stop_is_set_from_the_atr_known_at_the_entry_open(
    tmp_path: Path, high: float, low: float
) -> None:
    """ATR(1) = 2 on the signal bar, entry 101 at the next open: stop 99.0 whatever the entry
    bar goes on to do (its own range would give ATR 100 or 1)."""
    hours = [
        (100.0, 101.0, 99.0, 100.0, 100.0),
        (100.0, 102.0, 100.0, 101.0, 101.0),
        (101.0, high, low, 100.0, 101.0),
        *_flat(21),
    ]
    catalog = hour_catalog(tmp_path / "catalog", hours)
    dsl = {
        **_scenario_dsl(),
        "indicators": {"atr": {"type": "ATR", "period": 1}},
        "stop_loss": {"type": "atr_fixed", "indicator": "atr", "atr_multiplier": 1.0},
        "take_profit": {"type": "fixed_pct", "percent": 90},
    }
    emu = emulate(dsl, _RUN_START, _RUN_END, mode="tv", catalog_path=str(catalog), launch=_SIZING)
    assert emu.fills[0].price == "101.0"
    assert (emu.levels[0].kind, emu.levels[0].price) == ("STOP_MARKET", "99.0")


def _gap_stop_hours() -> list[tuple[float, float, float, float, float]]:
    """Signal at hour 1's close (101); entry at hour 2's open 101 (stop 99.0); hour 3 GAPS to 90."""
    return [
        (100.0, 100.0, 100.0, 100.0, 100.0),
        (100.0, 101.0, 100.0, 101.0, 101.0),
        (101.0, 101.0, 101.0, 101.0, 101.0),
        *[(90.0, 90.0, 90.0, 90.0, 90.0)] * 21,
    ]


def test_tv_gap_through_the_stop_fills_at_the_open_instant_and_owns_no_later_funding(
    tmp_path: Path,
) -> None:
    """A stop gapped through at hour 3's open fills at that open (price 90.0, 03:00:00), not at the
    bar close, so a settlement at 03:30 -- after the exit -- is not paid by the closed position."""
    catalog = hour_catalog(tmp_path / "catalog", _gap_stop_hours())
    archive = _funding_archive(tmp_path / "f.db", _DAY0_NS + 3 * _HOUR_MS * 1_000_000 + 30 * _MIN_MS * 1_000_000)
    emu = emulate(
        _scenario_dsl(), _RUN_START, _RUN_END, mode="tv",
        catalog_path=str(catalog), launch=_SIZING, funding_archive_path=str(archive),
    )
    entry, stop = emu.fills
    assert (entry.action, entry.price, entry.ts) == ("entry", "101.0", _DAY0_NS + 2 * _HOUR_MS * 1_000_000)
    assert (stop.action, stop.price, stop.ts) == ("stop", "90.0", _DAY0_NS + 3 * _HOUR_MS * 1_000_000)
    assert sum(emu.ledger.finalized.values()) == 0.0
    # the same gap one settlement earlier (02:30, inside the position) IS paid: 0.0001 x 4.950 x 101.0
    early = _funding_archive(tmp_path / "g.db", _DAY0_NS + 2 * _HOUR_MS * 1_000_000 + 30 * _MIN_MS * 1_000_000)
    emu2 = emulate(
        _scenario_dsl(), _RUN_START, _RUN_END, mode="tv",
        catalog_path=str(catalog), launch=_SIZING, funding_archive_path=str(early),
    )
    assert sum(emu2.ledger.finalized.values()) == pytest.approx(0.0001 * 4.950 * 101.0, rel=1e-9)


def _tp_gap_hours() -> list[tuple[float, float, float, float, float]]:
    """Entry at 101 (target 104.0); hour 3 opens at 110, ABOVE the target."""
    return [
        (100.0, 100.0, 100.0, 100.0, 100.0),
        (100.0, 101.0, 100.0, 101.0, 101.0),
        (101.0, 101.0, 101.0, 101.0, 101.0),
        *[(110.0, 110.0, 110.0, 110.0, 110.0)] * 21,
    ]


def test_tv_fills_a_gap_through_the_take_profit_at_the_open(tmp_path: Path) -> None:
    catalog = hour_catalog(tmp_path / "catalog", _tp_gap_hours())
    emu = emulate(_scenario_dsl(), _RUN_START, _RUN_END, mode="tv", catalog_path=str(catalog), launch=_SIZING)
    entry, target, *_ = emu.fills  # the 110 close signals again afterwards
    assert (entry.action, entry.price) == ("entry", "101.0")
    assert (target.action, target.side, target.qty, target.price) == ("target", "SELL", "4.950", "110.0")
    assert target.ts == _DAY0_NS + 3 * _HOUR_MS * 1_000_000


def test_engine_mode_gap_through_the_take_profit_equals_the_screening_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine releases the target on its fill tick (103.9, one minute into the hour): the
    emulator's trade equals the real screening runner's, and does not refuse."""
    catalog = hour_catalog(tmp_path / "catalog", _tp_gap_hours())
    kw: dict[str, Any] = {"catalog_path": str(catalog), "launch": _SIZING}
    emu = emulate(_scenario_dsl(), _RUN_START, _RUN_END, mode="engine", **kw)
    engine = run_engine(monkeypatch, _scenario_dsl(), _RUN_START, _RUN_END, **kw)
    assert [(f.action, f.price) for f in engine.fills][:2] == [("entry", "101.0"), ("target", "103.9")]
    assert emu.fills[:2] == engine.fills[:2]


# ---------------------------------------------------------------------------------------------
# funding: window (entry, exit]
# ---------------------------------------------------------------------------------------------


def test_a_settlement_on_the_reversal_instant_is_paid_by_the_old_side_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settlement stamped exactly on the reversal instant belongs to the position that closes
    there (exit inclusive: 0.0001 x 101.0 USDT) and not to the one opened on that instant (entry
    exclusive), and the emulator charges exactly what the engine's ledger does."""
    from vibe_quant.trade_ledger import accrue_position_funding

    catalog = _flat_catalog(tmp_path / "catalog")
    archive = _funding_archive(tmp_path / "f.db", _REVERSAL_NS)
    dsl = reverse_dsl("emu_funding", long=("close < 10100.5",), short=("close >= 10470", "close < 10490"))
    launch = LaunchSettings(sizing=SizingSettings(risk_per_trade=0.0202, max_margin_pct=None))
    kw: dict[str, Any] = {
        "catalog_path": str(catalog), "funding_archive_path": str(archive), "launch": launch,
    }

    emu = te.emulate(NTScreeningRunner(dsl, ["BTCUSDT"], FLAT_START, FLAT_END, **kw), "BTCUSDT", "engine")
    ref = run_screening(
        catalog, dsl, risk_per_trade=0.0202, funding_archive=archive, end=FLAT_END
    )

    long_entry, short_entry = [f for f in emu.fills if f.action == "entry"]
    assert (long_entry.qty, long_entry.price) == ("0.010", "10100.0")
    assert short_entry.ts == _REVERSAL_NS
    (long_cost,) = emu.ledger.finalized.values()  # only the long is closed
    assert long_cost == pytest.approx(0.0101, rel=1e-12)
    assert list(emu.ledger.finalized.values()) == list(ref.ledger.finalized.values())
    short_pos = emu.positions[-1]
    assert short_pos.is_open and short_pos.ts_opened == _REVERSAL_NS
    funding = emu.ledger.model.funding
    assert accrue_position_funding(funding, short_pos, exit_ns=_REVERSAL_NS + 10 * 60 * 10**9).total == 0.0
    assert emu.ledger.paid_through(_REVERSAL_NS, [short_pos]) == pytest.approx(0.0101, rel=1e-12)


# ---------------------------------------------------------------------------------------------
# refusals and the CLI
# ---------------------------------------------------------------------------------------------


def test_the_emulator_refuses_what_it_does_not_model(tmp_path: Path) -> None:
    """An unmodeled rule raises; it never runs a different one (max_bars_in_trade time exit)."""
    catalog = hour_catalog(tmp_path / "catalog", _both_touched(104.6, 97.0))
    dsl = {**_scenario_dsl(), "max_bars_in_trade": 4}
    with pytest.raises(te.EmulatorUnsupported, match="max_bars_in_trade"):
        emulate(dsl, _RUN_START, _RUN_END, catalog_path=str(catalog), launch=_SIZING)


@pytest.mark.parametrize("mode", ["engine", "tv"])
def test_atr_trailing_stops_are_refused_before_any_fill(tmp_path: Path, mode: str) -> None:
    catalog = hour_catalog(tmp_path / "catalog", _both_touched(104.6, 97.0))
    dsl = {
        **_scenario_dsl(),
        "indicators": {"atr": {"type": "ATR", "period": 1}},
        "stop_loss": {"type": "atr_trailing", "indicator": "atr", "atr_multiplier": 1.0},
    }
    with pytest.raises(te.EmulatorUnsupported, match="atr_trailing"):
        emulate(dsl, _RUN_START, _RUN_END, mode=mode, catalog_path=str(catalog), launch=_SIZING)


def test_unknown_mode_is_refused(tmp_path: Path) -> None:
    catalog = hour_catalog(tmp_path / "catalog", _both_touched(104.6, 97.0))
    runner = NTScreeningRunner(
        _scenario_dsl(), ["BTCUSDT"], _RUN_START, _RUN_END, catalog_path=str(catalog), launch=_SIZING
    )
    with pytest.raises(ValueError, match="mode must be"):
        te.emulate(runner, "BTCUSDT", "paper")


class _StubPort:
    """The part of ``vibe_quant.discovery.port.Port`` the CLI reads."""

    def __init__(self, dsl: dict[str, Any], defaults: dict[str, Any]) -> None:
        self._dsl, self._defaults = dsl, defaults

    def defaults(self) -> dict[str, Any]:
        return dict(self._defaults)

    def render(self, values: dict[str, Any], timeframe: str | None = None) -> dict[str, Any]:
        text = json.dumps(self._dsl)
        for key, value in values.items():
            text = text.replace(f'"{{{{{key}}}}}"', json.dumps(value))
        out: dict[str, Any] = json.loads(text)
        if timeframe is not None:
            out["timeframe"] = timeframe
        return out


@pytest.fixture
def stub_port(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``vibe_quant.discovery.port`` (a sibling task) stubbed: load_port returns a templated DSL."""
    seen: list[str] = []
    module = ModuleType("vibe_quant.discovery.port")

    def load_port(path: str) -> _StubPort:
        seen.append(str(path))
        dsl = {
            "name": "stub_port",
            "timeframe": "1h",
            "indicators": {"rsi": {"type": "RSI", "period": "{{period}}"}},
            "entry_conditions": {"long": ["rsi crosses_below 30"], "short": ["rsi crosses_above 70"]},
            "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 45"]},
            "stop_loss": {"type": "fixed_pct", "percent": 1.0},
            "take_profit": {"type": "fixed_pct", "percent": 2.0},
        }
        return _StubPort(dsl, {"period": 14})

    module.load_port = load_port  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vibe_quant.discovery.port", module)
    return seen


def test_cli_render_applies_set_overrides_and_timeframe(
    stub_port: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    tv = _load("tv")
    assert tv.main(["render", "--port", "ports/03_x"]) == 0
    default = yaml.safe_load(capsys.readouterr().out)
    assert stub_port == ["ports/03_x"]
    assert (default["timeframe"], default["indicators"]["rsi"]["period"]) == ("1h", 14)

    assert tv.main(["render", "--port", "ports/09_y", "--set", "period=7", "--timeframe", "4h"]) == 0
    other = yaml.safe_load(capsys.readouterr().out)
    assert (other["timeframe"], other["indicators"]["rsi"]["period"]) == ("4h", 7)
    with pytest.raises(SystemExit):
        tv.main(["render", "--port", "ports/03_x", "--set", "period"])


@needs_catalog
def test_cli_emulate_prints_the_engine_trade_list(
    stub_port: list[str], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``emulate`` on the stub port prints one line per trade whose entry equals the engine's."""
    tv = _load("tv")
    args = ["emulate", "--port", "ports/03_x", "--symbol", "BTCUSDT",
            "--start", "2026-03-02", "--end", "2026-03-12"]
    assert tv.main(args) == 0
    out = capsys.readouterr().out
    dsl = {**money_fixtures()["plain"], "name": "stub_port", "timeframe": "1h"}
    engine = run_engine(monkeypatch, dsl, "2026-03-02", "2026-03-12")
    entries = [f for f in engine.fills if f.action == "entry"]
    trade_lines = [ln for ln in out.splitlines() if "entry " in ln]
    assert len(trade_lines) == len(entries) >= 3
    first = entries[0]
    assert f"{'LONG' if first.side == 'BUY' else 'SHORT'}" in trade_lines[0]
    assert f"entry {first.price}" in trade_lines[0]
    assert f"fills={len(engine.fills)}" in out


def _real_port_folder(root: Path, allowed: list[str]) -> Path:
    import yaml

    doc = {
        "name": "timeframe_guard", "pine": "sample.pine", "tv": {"initial_capital": 1000},
        "params": {}, "constraints": [], "allowed_timeframes": allowed, "dsl": _scenario_dsl(),
    }
    folder = root / "port"
    folder.mkdir()
    (folder / "port.yaml").write_text(yaml.safe_dump(doc))
    return folder


def test_cli_render_goes_through_the_real_ports_timeframe_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--timeframe`` is ``Port.render(timeframe=...)``'s allowed_timeframes gate, never a
    post-render overwrite: a 1h-only port refuses 15m and renders its allowed 1h override."""
    import yaml

    from vibe_quant.discovery.port import PortError

    tv = _load("tv")
    folder = _real_port_folder(tmp_path, ["1h", "4h"])
    with pytest.raises(PortError, match="does not support timeframe 15m"):
        tv.main(["render", "--port", str(folder), "--timeframe", "15m"])
    with pytest.raises(PortError, match="does not support timeframe 15m"):
        tv.main(["emulate", "--port", str(folder), "--timeframe", "15m", "--symbol", "BTCUSDT",
                 "--start", "2024-01-01", "--end", "2024-01-02"])
    assert capsys.readouterr().out == ""  # nothing rendered before the refusal
    assert tv.main(["render", "--port", str(folder), "--timeframe", "4h"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["timeframe"] == "4h"
    assert tv.main(["render", "--port", str(folder)]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["timeframe"] == "1h"
