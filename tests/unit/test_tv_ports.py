"""Every TradingView port folder: Pine signals (R62) and engine parity (R63).

``scripts/research/tv_ports/<NN_name>/`` carries a ``port.yaml`` and a ``pine_fixture.py``.
The fixture is an INDEPENDENT numpy transcription of the Pine script's own entry/exit rules
over a deterministic bar set, so it cannot drift with the plugins or the compiler.

* ``test_port_pine_fixture_signals`` (R62): the fixture's ``(bar, side, action)`` events
  equal the compiled strategy's signals from :mod:`vibe_quant.fast_tier.signals` on the same
  bars, bar for bar. The fixture declares ``WARMUP``; bars before it are skipped because the
  engine's indicator readiness gate opens later than Pine's first defined series value.
* ``test_port_engine_mode_matches_screening`` (R63): the real screening engine and the broker
  emulator in ``engine`` mode produce the same fills, levels, commissions and costed equity
  on the fixture, with at least three closed trades.

Ports are DISCOVERED: adding a folder with a ``port.yaml`` adds its cases.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.discovery.port import load_port
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.screening.nt_runner import NTScreeningRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
TV_DIR = REPO / "scripts" / "research" / "tv_ports"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
#: 1m volume of the fixture catalog: large enough that a quarter of a bar's volume covers a
#: 50 %-of-equity position (NT and the emulator split fills by the bar's own volume).
MINUTE_VOLUME = 1000.0
DAY0 = datetime(2024, 1, 1, tzinfo=UTC)
DAY0_MS = int(DAY0.timestamp()) * 1000
#: A 10,000 USDT account: at the 1,000 default the journal's 50 % stop sizes a 40 USDT
#: notional, under the instrument's 50 USDT minimum, so no port with no stop would trade.
LAUNCH = LaunchSettings(starting_balance=10_000)


def _port_dirs() -> list[Path]:
    """Every port folder on disk; the red-first sentinel is the first port of the task."""
    dirs = sorted(path.parent for path in TV_DIR.glob("*/port.yaml"))
    return dirs or [TV_DIR / "03_supertrend"]


PORT_DIRS = _port_dirs()
PORT_IDS = [path.name for path in PORT_DIRS]


def _load_fixture(folder: Path) -> ModuleType:
    """Import ``<folder>/pine_fixture.py`` by path (vibe_quant never imports scripts/research)."""
    name = f"tv_port_fixture_{folder.name}"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, folder / "pine_fixture.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _render(folder: Path) -> dict[str, Any]:
    """The port's DSL at its own defaults, validated by ``load_port``."""
    port = load_port(folder)
    return dict(port.render(port.defaults()))


def _window(n_bars: int) -> tuple[str, str]:
    """[start, end) date strings covering exactly the fixture's hourly bars from 2024-01-01.

    The runner's fill-tick check requires the last 1m tick to reach ``end - 1h``, so a
    fixture's bar count must be a whole number of days (the catalog ends where it ends).
    """
    assert n_bars % 24 == 0, f"a port fixture must span whole days, got {n_bars} hours"
    end = DAY0 + timedelta(days=n_bars // 24)
    return DAY0.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _write_catalog(root: Path, bars: Mapping[str, Any]) -> Path:
    """A 1m + 1h BTCUSDT catalog whose hourly bars are exactly ``bars``.

    Each hour's 1m path is ``open -> close -> high -> low -> close ...``: the aggregated 1h
    bar is (open, high, low, close), and the first two 1m closes (the hour boundary's fill
    ticks) are that hour's close.
    """
    manager = CatalogManager(root)
    instrument = create_instrument(SYMBOL)
    manager.write_instrument(instrument)
    opens = bars["open"].tolist()
    highs = bars["high"].tolist()
    lows = bars["low"].tolist()
    closes = bars["close"].tolist()
    klines: list[dict[str, float]] = []
    for hour, (o, hi, lo, c) in enumerate(zip(opens, highs, lows, closes, strict=True)):
        path = [c, hi, lo] + [c] * 57
        prev = o
        for minute, close in enumerate(path):
            open_ms = DAY0_MS + hour * HOUR_MS + minute * MINUTE_MS
            klines.append(
                {
                    "open_time": open_ms,
                    "close_time": open_ms + MINUTE_MS - 1,
                    "open": prev,
                    "high": max(prev, close),
                    "low": min(prev, close),
                    "close": close,
                    "volume": MINUTE_VOLUME,
                }
            )
            prev = close
    bars_1m = klines_to_bars(
        klines,
        instrument.id,
        get_bar_type(SYMBOL, "1m"),
        instrument.size_precision,
        instrument.price_precision,
    )
    manager.write_bars(bars_1m)
    manager.write_bars(
        aggregate_bars(
            bars_1m,
            get_bar_type(SYMBOL, TIMEFRAME),
            60,
            instrument.size_precision,
            instrument.price_precision,
        )
    )
    return root


@pytest.fixture(scope="module")
def catalog_for(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Path], Path]:
    """One catalog per port, built once and shared by its R62 and R63 cases."""
    built: dict[str, Path] = {}

    def get(folder: Path) -> Path:
        if folder.name not in built:
            bars = _load_fixture(folder).bars()
            root = tmp_path_factory.mktemp(f"tv_port_{folder.name}") / "catalog"
            built[folder.name] = _write_catalog(root, bars)
        return built[folder.name]

    return get


def _catalog_bars(catalog: Path, start: str, end: str) -> dict[str, np.ndarray]:
    """The hourly bars the compiled strategy itself reads (catalog-rounded prices)."""
    from vibe_quant.fast_tier.signals import load_bars

    rows = load_bars(
        str(catalog), f"{SYMBOL}-PERP.BINANCE-1-HOUR-LAST-EXTERNAL", start, end
    )
    keys = ("open", "high", "low", "close", "volume")
    return {key: np.array([float(getattr(bar, key)) for bar in rows]) for key in keys}


def _events(long_entry: Any, short_entry: Any, long_exit: Any, short_exit: Any) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every signal, in bar order."""
    out: list[tuple[int, str, str]] = []
    for i in range(len(long_entry)):
        if long_entry[i]:
            out.append((i, "BUY", "entry"))
        if short_entry[i]:
            out.append((i, "SELL", "entry"))
        if long_exit[i]:
            out.append((i, "SELL", "exit"))
        if short_exit[i]:
            out.append((i, "BUY", "exit"))
    return out


@pytest.mark.parametrize("port", ["01_bb_rsi_v1_1", "01b_bb_rsi_v1_0", "02_macd_sma200"])
def test_pine_crossover_counts_a_previous_bar_at_the_level(port: str) -> None:
    """Pine ``crossover``/``crossunder`` are ``> now`` / ``<= before`` (``>=`` before).

    A previous value exactly ON the level already counts as crossed, so ``<=`` -> ``<`` or
    ``>=`` -> ``>`` in a fixture transcription would change the signal at that edge.
    """
    module = _load_fixture(TV_DIR / port)
    left = np.array([0.0, 1.0, 1.0, -1.0])
    assert module._cross_up(left, 0.0).tolist() == [False, True, False, False]
    assert module._cross_dn(left, 0.0).tolist() == [False, False, False, True]


@pytest.mark.parametrize("folder", PORT_DIRS, ids=PORT_IDS)
def test_port_renders_with_defaults(folder: Path) -> None:
    """``load_port`` accepts the folder and the default render is a validated DSL."""
    dsl = _render(folder)
    assert dsl["timeframe"] == TIMEFRAME
    assert dsl["entry_conditions"]["long"] or dsl["entry_conditions"]["short"]


@pytest.mark.parametrize("folder", PORT_DIRS, ids=PORT_IDS)
@pytest.mark.slow(reason="compiles the strategy and replays its signals over the fixture")
def test_port_pine_fixture_signals(folder: Path, catalog_for: Callable[[Path], Path]) -> None:
    """R62: the Pine transcription's signals equal the compiled strategy's, bar for bar."""
    from vibe_quant.fast_tier.signals import signals_for_runner

    fixture = _load_fixture(folder)
    dsl = _render(folder)
    start, end = _window(len(fixture.bars()["close"]))
    catalog = catalog_for(folder)
    bars = _catalog_bars(catalog, start, end)
    runner = NTScreeningRunner(dsl, [SYMBOL], start, end, catalog_path=str(catalog))
    sig = signals_for_runner(runner, SYMBOL).signals
    warmup = int(fixture.WARMUP)

    pine = [event for event in fixture.events(bars) if event[0] >= warmup]
    compiled = [
        event
        for event in _events(sig.long_entry, sig.short_entry, sig.long_exit, sig.short_exit)
        if event[0] >= warmup
    ]

    assert compiled == pine
    assert len([event for event in pine if event[2] == "entry"]) >= 3


@pytest.mark.parametrize("folder", PORT_DIRS, ids=PORT_IDS)
@pytest.mark.slow(reason="real screening engine and emulator replay over the fixture")
def test_port_engine_mode_matches_screening(
    folder: Path,
    catalog_for: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R63: the emulator's engine mode equals the screening engine on >= 3 closed trades."""
    from tests.unit.test_tv_emulator import assert_e_parity, emulate, run_engine

    fixture = _load_fixture(folder)
    dsl = _render(folder)
    start, end = _window(len(fixture.bars()["close"]))
    catalog_path = str(catalog_for(folder))

    engine = run_engine(monkeypatch, dsl, start, end, catalog_path=catalog_path, launch=LAUNCH)
    emulator = emulate(dsl, start, end, catalog_path=catalog_path, launch=LAUNCH)

    assert_e_parity(engine, emulator, min_trades=3)
