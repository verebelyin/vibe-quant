"""Screening ``liquidation_mode``: every metric follows the truncated ledgers (bead vibe-quant-2gyk1.1.6).

Positions are duck-typed stubs on a stub engine; MARK bars are REAL parquet in a temp catalog
(the loader, coverage check and scanner run for real). Hand-computed scenario, BTC at 10x
isolated, 40000 account, 1 BTC long opened at 40000 (margin 4000, MMR .004, cum 0):

    LP = (4000 - 40000) / (0.004 - 1) = 36144.57831325301

Position A (07:00 day 0, commission 20) meets a one-minute mark crash to 30000 at 07:30: the
venue closes it at the LP and takes the whole margin, loss 4000 (price loss 3855.42 + clearance
fee 144.58). NT knows nothing of it: its stub sells A at 38000 at 09:00 (realized -2039,
commissions 39), after the 08:00 funding settlement. Later winners/losers B1..B3 are untouched.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from nautilus_trader.model.enums import OrderSide, OrderType

from tests.unit.fake_engine import FakeFill, FakePosition, fake_bt_result, fake_engine
from tests.unit.test_audit_metrics_funding import _RATE, _archive, empty_archive
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.data.mark_bars import build_mark_bars
from vibe_quant.errors import DataUnavailableError, LiquidationIntegrityError
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.metrics import trade_win_rate
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.funding import clear_rate_cache

if TYPE_CHECKING:
    from pathlib import Path
    from types import SimpleNamespace

MIN = 60_000_000_000
DAY_MIN = 1440
START_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
START_NS = START_MS * 1_000_000
BTC = "BTCUSDT-PERP.BINANCE"
BALANCE = 40000.0
N_BARS = 3 * DAY_MIN  # window 2025-01-01..2025-01-04


@pytest.fixture(autouse=True)
def _fresh_funding_cache() -> None:
    clear_rate_cache()


BUY, SELL, MARKET = OrderSide.BUY, OrderSide.SELL, OrderType.MARKET  # real NT enums: the
# unmodified screening extractors compare against them


def at(minute: int) -> int:
    return START_NS + minute * MIN


Fill = FakeFill
Pos = FakePosition


def long_trade(
    pid: str,
    open_min: int,
    close_min: int,
    exit_px: float,
    realized: float,
    commissions: tuple[float, float] = (0.0, 0.0),
) -> Pos:
    fills = [
        Fill(at(open_min), BUY, 1.0, 40000.0, commissions[0]),
        Fill(at(close_min), SELL, 1.0, exit_px, commissions[1]),
    ]
    return Pos(
        pid,
        BTC,
        at(open_min),
        at(close_min),
        fills,
        realized,
        avg_px_open=40000.0,
        peak_qty=1.0,
        commission_list=list(commissions),
    )


def pos_a(*, still_open: bool = False) -> Pos:
    """Liquidated at 07:30 by the venue; NT sells at 38000 at 09:00 (realized -2039)."""
    p = long_trade("A", 420, 540, 38000.0, -2039.0, (20.0, 19.0))
    if still_open:  # NT never closed it: screening would mark it to market at the run end
        p.events.pop()
        p.ts_closed, p.is_closed, p.signed_qty = 0, False, 1.0
        p.realized_pnl, p.commission_list = -20.0, [20.0]
    return p


def winners_and_loser() -> list[Pos]:
    return [
        long_trade("B1", DAY_MIN + 600, DAY_MIN + 700, 40500.0, 500.0),
        long_trade("B2", DAY_MIN + 720, DAY_MIN + 780, 40200.0, 200.0),
        long_trade("B3", 2 * DAY_MIN + 600, 2 * DAY_MIN + 660, 39900.0, -100.0),
    ]


class Bars(NamedTuple):
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]


def flat_bars(overrides: dict[int, tuple[float, float, float, float]] | None = None) -> Bars:
    """40000 flat (low 39900 / high 40100) with per-minute (open, high, low, close) overrides."""
    o, h, lo, c = ([v] * N_BARS for v in (40000.0, 40100.0, 39900.0, 40000.0))
    for i, (bo, bh, bl, bc) in (overrides or {}).items():
        o[i], h[i], lo[i], c[i] = bo, bh, bl, bc
    return Bars(o, h, lo, c)


CRASH = flat_bars({450: (40000.0, 40000.0, 30000.0, 40000.0)})  # 07:30 mark crash to 30000
CALM = flat_bars()


def write_marks(root: Path, bars: Bars) -> Path:
    rows = []
    for i in range(len(bars.open)):  # may run past the window (MARK data after the run end)
        open_ms = START_MS + i * 60_000
        rows.append(
            (open_ms, bars.open[i], bars.high[i], bars.low[i], bars.close[i], 0.0,
             open_ms + 59_999, 0, 60, 0.0, 0.0)
        )  # fmt: skip
    archive = RawDataArchive(root / "mark_archive.db")
    archive.insert_mark_klines("BTCUSDT", "1m", rows, "test")
    build_mark_bars(CatalogManager(root / "catalog"), archive, "BTCUSDT")
    return root / "catalog"


@dataclass(frozen=True)
class _LastBar:
    """One execution (LAST) bar stamped at the minute close less 1 ms, like a catalog bar."""

    ts_init: int
    open: float
    high: float
    low: float
    close: float


def engine_for(positions: list[Pos], bars: Bars) -> SimpleNamespace:
    """NT-cache stub serving the execution (LAST 1m) bars: same OHLC as the mark bars,
    stamped at the minute close less 1 ms like a catalog bar's ``ts_init``.

    The bars are served for whichever execution bar type the extractor asks for
    (the same scenario runs at 1m, 4h and 1d)."""
    last_bars = [
        _LastBar(
            START_NS + (i + 1) * MIN - 1_000_000,
            bars.open[i],
            bars.high[i],
            bars.low[i],
            bars.close[i],
        )
        for i in range(N_BARS)
    ]
    return fake_engine(positions, bars_fallback=last_bars)


def bt_result(n_positions: int = 9) -> SimpleNamespace:
    """NT's (untruncated) stats: deliberately different from every liquidation-aware number."""
    return fake_bt_result(
        n_positions,
        stats_pnls={"USDT": {"PnL% (total)": -5.0, "Win Rate": 0.9}},
        stats_returns={"Sharpe Ratio (252 days)": 1.0},
    )


def make_runner(
    tmp_path: Path,
    mode: str,
    *,
    catalog: Path | None = None,
    funding: bool = True,
    starting_balance: int = 1000,
) -> NTScreeningRunner:
    archive = _archive(tmp_path / f"funding_{mode}.db", list(range(12)))
    runner = NTScreeningRunner(
        dsl_dict={},
        symbols=["BTCUSDT"],
        start_date="2025-01-01",
        end_date="2025-01-04",
        catalog_path=str(catalog) if catalog else None,
        funding_archive_path=str(archive if funding else empty_archive(tmp_path / "none.db")),
        launch=LaunchSettings(
            liquidation=LiquidationSettings(mode=mode),  # type: ignore[arg-type]
            starting_balance=starting_balance,
        ),
    )
    runner._all_timeframes = {"1m"}
    runner._resolved_catalog_path = catalog  # type: ignore[assignment]
    return runner


def extract(
    runner: NTScreeningRunner,
    positions: list[Pos],
    bars: Bars,
    balance: float = BALANCE,
    n_positions: int = 9,
) -> Any:
    return runner._extract_metrics(
        {}, bt_result(n_positions), engine_for(positions, bars), time.time(), balance
    )


def equal_metrics(a: Any, b: Any) -> bool:
    # exit_timing is a mode-on diagnostic (zeros) vs None when the mode is off, not a
    # measured metric: neutralise it so the identity-overlay equality still holds (T7a-R1).
    return bool(
        replace(a, execution_time_seconds=0.0, exit_timing=None)
        == replace(b, execution_time_seconds=0.0, exit_timing=None)
    )


# --------------------------------------------------------------------------- hits


def test_notes_and_runner_state_for_a_hit(tmp_path: Path) -> None:
    catalog = write_marks(tmp_path, CRASH)
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    m = extract(runner, [pos_a(), *winners_and_loser()], CRASH)
    notes = runner.last_liquidation_notes
    assert notes is not None
    assert (notes["mode"], notes["leverage"], notes["count"], notes["dropped"]) == (
        "isolated", 10.0, 1, 0)  # fmt: skip
    assert notes["pct_trades"] == pytest.approx(25.0)  # 1 of the 4 adjusted trades
    assert notes["total_forfeited"] == pytest.approx(m.liquidation_forfeited)


def test_hit_position_never_marked_to_market_when_nt_left_it_open(tmp_path: Path) -> None:
    """NT keeps A open to the end: screening would mark it at the run-end close and charge an
    exit fee; the liquidated position takes the venue close instead."""
    catalog = write_marks(tmp_path, CRASH)
    m = extract(make_runner(tmp_path, "isolated", catalog=catalog), [pos_a(still_open=True)], CRASH)
    assert m.liquidation_count == 1
    assert m.total_trades == 1
    # no 0.05% exit fee of a run-end mark: commission 20 + the clearance fee only
    assert m.total_fees == pytest.approx(20.0 + m.liquidation_forfeited)
    assert m.total_return == pytest.approx(-4020.0 / BALANCE)
    assert m.total_funding == 0.0


def test_liquidated_run_counts_funding(tmp_path: Path) -> None:
    """R9b: a survivor realized +10 that paid 15 funding is a LOSS (net -5). The removed
    ``realized > 0`` basis counted it as a win; the shared net definition does not."""
    catalog = write_marks(tmp_path, CRASH)
    survivor = Pos(
        id="B1",
        instrument_id=BTC,
        ts_opened=at(451),  # after the 07:30 crash: never marked at 30000
        ts_closed=at(960),  # 16:00, across the 08:00 settlement
        events=[
            Fill(at(451), BUY, 1.25, 40000.0, 0.0),
            Fill(at(960), SELL, 1.25, 40000.0, 0.0),
        ],
        realized_pnl=10.0,
        avg_px_open=40000.0,
        peak_qty=1.25,  # entry notional 50000 -> one 0.0003 settlement = 15
        commission_list=[0.0, 0.0],
    )
    m = extract(
        make_runner(tmp_path, "isolated", catalog=catalog), [pos_a(), survivor], CRASH, n_positions=2
    )
    assert m.liquidation_count == 1
    assert m.total_trades == 2
    assert m.total_funding == pytest.approx(15.0)
    assert m.total_return == pytest.approx((-4020.0 - 5.0) / BALANCE)
    assert m.win_rate == trade_win_rate([-4020.0, -5.0]) == 0.0  # was 1/2


def test_mode_off_ignores_the_crash_and_keeps_nt_numbers(tmp_path: Path) -> None:
    """Control: same engine, mode off = NT-untruncated figures (the numbers liquidation fixes)."""
    runner = make_runner(tmp_path, "off")
    m = extract(runner, [pos_a(), *winners_and_loser()], CRASH)
    assert runner.last_liquidation_notes is None
    assert (m.liquidation_count, m.liquidation_forfeited) == (0, 0.0)
    assert m.total_trades == 9
    # the one net definition (bd vibe-quant-neuvu.4): A realized -2039 and paid the 12
    # funding of the 08:00 settlement it held across -> net -2051; B1/B2 won
    nets = [-2051.0, 500.0, 200.0, -100.0]
    assert m.win_rate == trade_win_rate(nets) == 0.5  # 2 of the 4 real trades won
    assert m.win_rate != 0.9  # NT's stub reported 0.9
    assert m.total_funding == pytest.approx(_RATE * 40000.0)  # A held across 08:00
    assert m.max_drawdown == pytest.approx(10020.0 / 40000.0)  # untruncated 30000 low


# ------------------------------------------------------------------------ identity


def test_identity_overlay_leaves_metrics_exactly_as_mode_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = write_marks(tmp_path, CALM)
    positions = [long_trade("A", 420, 540, 38000.0, -2039.0, (20.0, 19.0)), *winners_and_loser()]
    off = extract(make_runner(tmp_path, "off", catalog=catalog), positions, CALM)

    import vibe_quant.trade_ledger as recompute

    def must_not_run(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the recompute helper ran for an identity overlay")

    monkeypatch.setattr(recompute, "screening_liquidation_trades", must_not_run)
    runner = make_runner(tmp_path, "cross", catalog=catalog)
    on = extract(runner, positions, CALM)
    assert equal_metrics(on, off)
    assert on.total_trades == 9  # NT's figure stays: nothing was truncated
    assert runner.last_liquidation_notes is not None
    assert runner.last_liquidation_notes["count"] == 0
    assert runner.last_liquidation_notes["mode"] == "cross"


# --------------------------------------------------------------------------- cross


def test_cross_bust_drops_later_positions_and_ends_at_zero_equity(tmp_path: Path) -> None:
    """4000 account, 1 BTC long at 40000 (10x of equity): the cross bust price equals the
    isolated LP. The account is wiped at 07:30; the winner B1 that NT opens afterwards is
    dropped (no trade, no PnL), equity never goes below 0."""
    catalog = write_marks(tmp_path, CRASH)
    b1 = long_trade("B1", 600, 700, 40500.0, 500.0)
    runner = make_runner(tmp_path, "cross", catalog=catalog)
    m = extract(runner, [pos_a(), b1], CRASH, balance=4000.0, n_positions=2)
    assert m.liquidation_count == 1
    assert m.total_trades == 1  # NT counted 2; the dropped B1 is not a trade
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0  # B1's +500 would make this > 0
    assert m.total_return == pytest.approx(-1.0)  # wallet 4000 forfeited: equity 0
    assert m.max_drawdown == pytest.approx(1.0)
    assert m.total_funding == 0.0
    # equity 0 after the bust: the MTM daily series is unusable -> DSR fails closed
    assert m.mtm_sharpe_daily is None and m.mtm_days is None
    notes = runner.last_liquidation_notes
    assert notes is not None
    assert notes["dropped"] == 1 and notes["bust_ts"] == at(450)


# --------------------------------------------------------------------- fail closed


def test_missing_mark_series_fails_closed_naming_the_symbol(tmp_path: Path) -> None:
    """I1/I5: with the mode on, missing MARK data is a LiquidationIntegrityError carrying the
    original class name and the symbol (the mode-off path never sees this)."""
    empty = tmp_path / "empty_catalog"
    empty.mkdir()
    runner = make_runner(tmp_path, "isolated", catalog=empty)
    with pytest.raises(LiquidationIntegrityError, match=r"MarkDataMissing.*BTCUSDT"):
        extract(runner, [pos_a(), *winners_and_loser()], CALM)


def test_call_reraises_missing_mark_data_but_still_masks_other_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty_catalog"
    empty.mkdir()
    runner = make_runner(tmp_path, "isolated", catalog=empty)
    engine = engine_for([pos_a()], CALM)
    monkeypatch.setattr(
        runner,
        "_run_backtest",
        lambda params, t: runner._extract_metrics(params, bt_result(), engine, t, BALANCE),
    )
    with pytest.raises(LiquidationIntegrityError, match=r"MarkDataMissing.*BTCUSDT"):
        runner({})

    def boom(_params: dict[str, float | int], _t: float) -> None:
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(runner, "_run_backtest", boom)
    assert runner({}).sharpe_ratio == float("-inf")  # unchanged: ordinary failures score -inf
    assert runner.last_liquidation_notes is None


# ------------------------------------------------------------------------- config


def test_venue_leverage_follows_the_chosen_leverage_in_every_mode() -> None:
    """V1-V5: the screening venue runs at the leverage in EVERY mode (off included); the default
    10 stays byte-identical to today."""
    from decimal import Decimal

    off = NTScreeningRunner({}, ["BTCUSDT"], "2025-01-01", "2025-01-02")
    assert off._venue_config().default_leverage == Decimal("10")
    off_three = NTScreeningRunner(
        {}, ["BTCUSDT"], "2025-01-01", "2025-01-02",
        launch=LaunchSettings(liquidation=LiquidationSettings(leverage=3)),
    )  # fmt: skip
    assert off_three._liquidation_mode == "off"
    assert off_three._venue_config().default_leverage == Decimal("3")
    on = NTScreeningRunner(
        {}, ["BTCUSDT"], "2025-01-01", "2025-01-02",
        launch=LaunchSettings(
            liquidation=LiquidationSettings(mode="isolated", leverage=20.0)
        ),
    )  # fmt: skip
    assert on._venue_config().default_leverage == Decimal("20")
    ten = NTScreeningRunner(
        {}, ["BTCUSDT"], "2025-01-01", "2025-01-02",
        launch=LaunchSettings(
            liquidation=LiquidationSettings(mode="cross", leverage=10.0)
        ),
    )  # fmt: skip
    assert ten._venue_config().default_leverage == Decimal("10")


# ------------------------------------------------- fees, missing LAST bars, mixed, timeframe


def spy_marking(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the position ids screening hands to the (real) run-end marking helper."""
    import vibe_quant.validation.extraction as extraction

    seen: list[str] = []
    real = extraction.mark_open_positions

    def spy(engine: Any, positions: Any, **kwargs: Any) -> Any:
        positions = list(positions)
        seen.extend(str(p.id) for p in positions)
        return real(engine, positions, **kwargs)

    monkeypatch.setattr(extraction, "mark_open_positions", spy)
    return seen


def test_hit_whose_open_cannot_be_marked_is_still_a_trade_and_a_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MARK coverage is valid but there is no LAST execution bar to mark the open position:
    the liquidation comes from the overlay, never from LAST, so A is still a -4020 trade."""
    catalog = write_marks(tmp_path, CRASH)
    engine = engine_for([pos_a(still_open=True)], CRASH)
    engine.kernel.cache.bars = lambda *_: []  # no LAST bars anywhere
    seen = spy_marking(monkeypatch)
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    m = runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)
    assert "A" not in seen  # a liquidated position is never marked to market
    assert m.liquidation_count == 1
    assert m.total_trades == 1
    assert m.total_return == pytest.approx(-0.1005)
    assert m.max_drawdown >= 0.1005 - 1e-12  # the closed-trade fallback still sees the loss
    assert m.total_fees == pytest.approx(20.0 + m.liquidation_forfeited)
    assert m.total_funding == 0.0


def test_hit_plus_open_survivor_reconciles_return_final_equity_and_daily_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A is liquidated; S opens on day 2 and is still open at the end of the run, so screening
    marks it and charges the modeled exit fee. The reported return, the trade net sum, the
    final MTM equity of the truncated ledgers and the collected daily series must agree."""
    catalog = write_marks(tmp_path, CRASH)
    s_pos = long_trade("S", 2 * DAY_MIN + 600, 2 * DAY_MIN + 660, 40000.0, 0.0, (20.0, 0.0))
    s_pos.events.pop()
    s_pos.ts_closed, s_pos.is_closed, s_pos.signed_qty = 0, False, 1.0
    s_pos.realized_pnl, s_pos.commission_list = -20.0, [20.0]
    seen = spy_marking(monkeypatch)
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    runner._collect_daily_returns = True
    m = extract(runner, [pos_a(), s_pos], CRASH, n_positions=2)
    exit_fee = 40000.0 * float(runner._fee_schedule.taker_rate)
    assert exit_fee > 0.0
    assert seen == ["S"]
    assert m.liquidation_count == 1 and m.total_trades == 2
    funding_s = m.total_funding  # S paid the day-2 16:00 settlement; A none (liquidated 07:30)
    assert funding_s == pytest.approx(_RATE * 40000.0)
    s_net = -20.0 - exit_fee - funding_s
    expected = (-4020.0 + s_net) / BALANCE
    assert m.total_return == pytest.approx(expected)
    assert m.total_fees == pytest.approx(20.0 + m.liquidation_forfeited + 20.0 + exit_fee)
    # the collected MTM daily series compounds to the same final equity
    series = runner.last_mtm_daily_returns
    assert series is not None
    equity = BALANCE
    for r in series.values:
        equity *= 1.0 + r
    assert equity == pytest.approx(BALANCE * (1.0 + m.total_return), rel=1e-12)
    assert equity == pytest.approx(BALANCE - 4020.0 - 20.0 - exit_fee - funding_s)


def test_dropped_and_liquidated_positions_are_never_marked_survivors_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = write_marks(tmp_path, CRASH)
    b1 = long_trade("B1", 600, 700, 40500.0, 500.0)
    b1.events.pop()  # NT never closed B1 either: open at the end of the run
    b1.ts_closed, b1.is_closed, b1.signed_qty = 0, False, 1.0
    b1.realized_pnl, b1.commission_list = 0.0, []
    seen = spy_marking(monkeypatch)
    runner = make_runner(tmp_path, "cross", catalog=catalog)
    m = extract(runner, [pos_a(still_open=True), b1], CRASH, balance=4000.0, n_positions=2)
    assert m.liquidation_count == 1  # A liquidated at 07:30, B1 (opens 10:00) dropped
    assert m.total_trades == 1
    assert "A" not in seen and "B1" not in seen
    assert m.total_return == pytest.approx(-1.0)


def test_overlay_gets_the_runs_execution_timeframe_not_one_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe_quant.liquidation.apply as apply_module

    catalog = write_marks(tmp_path, CALM)
    captured: list[dict[str, Any]] = []
    real = apply_module.apply_liquidation

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(apply_module, "apply_liquidation", spy)
    runner = make_runner(tmp_path, "cross", catalog=catalog)
    runner._all_timeframes = {"4h"}
    extract(runner, [], CALM, n_positions=0)
    assert [c["execution_bar_ns"] for c in captured] == [14_400_000_000_000]


def open_long(pid: str, open_min: int, instrument: str = BTC) -> Pos:
    """A 1-unit long NT never closed: screening marks it at the run end."""
    p = long_trade(pid, open_min, open_min + 1, 40000.0, 0.0, (0.0, 0.0))
    p.events.pop()
    p.ts_closed, p.is_closed, p.signed_qty = 0, False, 1.0
    p.instrument_id = instrument
    return p


def test_end_of_run_exit_fee_is_a_valuation_cost_never_a_liquidation_trigger(
    tmp_path: Path,
) -> None:
    """Cross wallet 270, 1 BTC long at 40000 (no commission, no funding), calm market: the
    scan without the survivor's exit fee finds nothing. The modeled exit fee (20) is booked
    afterwards: net -20, fees 20, equity 250 (it would bust a 270 wallet if it were scanned)."""
    catalog = write_marks(tmp_path, CALM)
    runner = make_runner(tmp_path, "cross", catalog=catalog)
    nt_flat = fake_bt_result(  # NT realized nothing: 0% PnL, so the return is the modeled costs
        1, stats_pnls={"USDT": {"PnL% (total)": 0.0}}, stats_returns={}
    )
    engine = engine_for([open_long("S", N_BARS - 2)], CALM)
    m = runner._extract_metrics({}, nt_flat, engine, time.time(), 270.0)
    exit_fee = 40000.0 * float(runner._fee_schedule.taker_rate)
    assert exit_fee == pytest.approx(20.0)
    assert m.liquidation_count == 0
    assert m.total_trades == 1
    assert m.total_fees == pytest.approx(exit_fee)
    assert m.total_return == pytest.approx(-20.0 / 270.0)
    assert 270.0 * (1.0 + m.total_return) == pytest.approx(250.0)
    assert runner.last_liquidation_notes is not None
    assert runner.last_liquidation_notes["count"] == 0


def add_eth_marks(root: Path, catalog: Path) -> None:
    rows = [
        (START_MS + i * 60_000, 40000.0, 40100.0, 39900.0, 40000.0, 0.0,
         START_MS + i * 60_000 + 59_999, 0, 60, 0.0, 0.0)
        for i in range(N_BARS)
    ]  # fmt: skip
    archive = RawDataArchive(root / "eth_mark_archive.db")
    archive.insert_mark_klines("ETHUSDT", "1m", rows, "test")
    build_mark_bars(CatalogManager(catalog), archive, "ETHUSDT")


def two_open_longs_with_btc_last_only() -> tuple[list[Pos], SimpleNamespace]:
    positions = [open_long("BTC", 420), open_long("ETH", 420, "ETHUSDT-PERP.BINANCE")]
    engine = engine_for(positions, CALM)
    btc_bars = engine.kernel.cache.bars
    engine.kernel.cache.bars = lambda bar_type: [] if str(bar_type).startswith("ETH") else btc_bars(
        bar_type
    )  # MARK coverage is valid for ETH, but there are no ETH LAST execution bars
    return positions, engine


def test_unpriceable_open_position_is_excluded_from_fees_when_the_mode_is_off(
    tmp_path: Path,
) -> None:
    """Mode off keeps its old rule for an open position with no LAST execution bar: it is left
    out of trades and return (only BTC's exit fee is charged), never valued at zero."""
    catalog = write_marks(tmp_path, CALM)
    add_eth_marks(tmp_path, catalog)
    _, engine = two_open_longs_with_btc_last_only()
    m = make_runner(tmp_path, "off", catalog=catalog)._extract_metrics(
        {}, bt_result(2), engine, time.time(), BALANCE
    )
    assert m.total_fees == pytest.approx(20.0)  # BTC's exit fee only: ETH is excluded


def test_funding_paid_by_an_open_position_before_its_hit_is_counted_once_without_last_bars(
    tmp_path: Path,
) -> None:
    """A opens at 06:00, pays the 08:00 funding (12) from its margin, is liquidated at 08:30
    and NT left it open with no LAST bar to mark it: the loss is commission 20 + the whole
    margin 4000 (12 funding + 3988), funding once."""
    crash_late = flat_bars({510: (40000.0, 40000.0, 30000.0, 40000.0)})
    catalog = write_marks(tmp_path, crash_late)
    a = pos_a(still_open=True)
    a.ts_opened = at(360)
    a.events[0].ts_event = at(360)
    engine = engine_for([a], crash_late)
    engine.kernel.cache.bars = lambda *_: []
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    m = runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)
    assert m.liquidation_count == 1
    assert m.total_funding == pytest.approx(_RATE * 40000.0)
    assert m.total_return == pytest.approx(-4020.0 / BALANCE)
    assert m.total_fees == pytest.approx(20.0 + m.liquidation_forfeited)


# --------------------------------------- one scan, one valuation horizon per open position


DIP = flat_bars({510: (40000.0, 40000.0, 36140.0, 40000.0)})  # 08:30 day 0: mark low 36140


def flat_bt(n_positions: int) -> SimpleNamespace:
    """NT stats of a run that realized nothing (PnL% 0): the return is the modeled cash."""
    return fake_bt_result(
        n_positions, stats_pnls={"USDT": {"PnL% (total)": 0.0}}, stats_returns={}
    )


def receiving_runner(tmp_path: Path, mode: str, catalog: Path | None) -> NTScreeningRunner:
    """Every archived settlement at -0.0003: a 1 BTC long @40000 RECEIVES 12 per 8h."""
    runner = make_runner(tmp_path, mode, catalog=catalog)
    conn = sqlite3.connect(str(runner._funding_archive_path))
    conn.execute("UPDATE raw_funding_rates SET funding_rate = ?", (-_RATE,))
    conn.commit()
    conn.close()
    return runner


def last_bars_until(engine: SimpleNamespace, n_bars: int) -> SimpleNamespace:
    """LAST execution bars stop after the first ``n_bars`` minutes (MARK bars stay complete)."""
    full = engine.kernel.cache.bars
    engine.kernel.cache.bars = lambda bar_type: full(bar_type)[:n_bars]
    return engine


def spy_scans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every liquidation scan (``apply_liquidation`` result) of a run."""
    import vibe_quant.liquidation.apply as apply_module

    scans: list[Any] = []
    real = apply_module.apply_liquidation

    def spy(*args: Any, **kwargs: Any) -> Any:
        scans.append(real(*args, **kwargs))
        return scans[-1]

    monkeypatch.setattr(apply_module, "apply_liquidation", spy)
    return scans


def test_open_position_whose_last_bars_stop_before_the_run_end_fails_closed_when_the_mode_is_on(
    tmp_path: Path,
) -> None:
    """S (1 BTC long @40000 from 06:00) receives 12 funding at 08:00, then the mark dips to 36140
    at 08:30: LP 36132.53 with the receipt, 36144.58 without. Its LAST bars stop at 07:00 while
    the run goes on to day 3, so mode off values it at 07:00 (no funding). A liquidation scan
    has no valuation horizon in the final bar for it: the run fails closed instead of scoring
    a survivor whose hit test depends on where its funding is cut."""
    catalog = write_marks(tmp_path, DIP)
    runner = receiving_runner(tmp_path, "isolated", catalog)
    engine = last_bars_until(engine_for([open_long("S", 360)], DIP), 421)
    last_ts = at(421) - 1_000_000
    with pytest.raises(
        DataUnavailableError,
        match=rf"BTCUSDT-PERP\.BINANCE.*cannot be valued.*{last_ts}.*{at(N_BARS)}",
    ):
        runner._extract_metrics({}, flat_bt(1), engine, time.time(), BALANCE)

    off = receiving_runner(tmp_path, "off", catalog)
    engine = last_bars_until(engine_for([open_long("S", 360)], DIP), 421)
    m = off._extract_metrics({}, flat_bt(1), engine, time.time(), BALANCE)
    assert (m.total_trades, m.total_fees, m.total_funding) == (1, 20.0, 0.0)
    assert m.total_return == pytest.approx(-20.0 / BALANCE)  # the exit fee at the 07:00 mark


def test_received_funding_keeps_a_survivor_alive_in_the_single_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same S, LAST bars through the final bar: the ONE scan funds S up to its horizon (mode
    off's mark, the last minute), so the 08:00 receipt raises its margin to 4012 and the 36140
    dip misses LP 36132.53. No hit; the run equals mode off."""
    catalog = write_marks(tmp_path, DIP)
    scans = spy_scans(monkeypatch)
    on = receiving_runner(tmp_path, "isolated", catalog)._extract_metrics(
        {}, flat_bt(1), engine_for([open_long("S", 360)], DIP), time.time(), BALANCE
    )
    assert [len(s.hits_by_key) for s in scans] == [0]  # exactly one scan, nothing hit
    assert on.liquidation_count == 0
    # 8 settlements received (08:00 day 0 .. 16:00 day 2); the one at the run end is after
    # the mark (end - 1 ms), so neither mode charges it
    assert on.total_funding == pytest.approx(-8 * 12.0)
    assert on.total_return == pytest.approx((8 * 12.0 - 20.0) / BALANCE)
    off = receiving_runner(tmp_path, "off", catalog)._extract_metrics(
        {}, flat_bt(1), engine_for([open_long("S", 360)], DIP), time.time(), BALANCE
    )
    assert equal_metrics(on, off)


def ledger_net(ledgers: dict[str, list[Any]]) -> float:
    """Cash the truncated ledgers realize (fills, fees, funding, clearance, exit fees)."""
    from vibe_quant.metrics import LedgerBook

    total = 0.0
    for events in ledgers.values():
        book = LedgerBook()
        for event in events:
            book.apply(event)
        total += book.realized
    return total


def spy_ledgers(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the (truncated) ledgers screening hands to the MTM path builder."""
    import vibe_quant.validation.extraction as extraction

    captured: list[Any] = []
    real = extraction.mark_to_market_paths

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["ledgers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(extraction, "mark_to_market_paths", spy)
    return captured


def test_funding_paid_up_to_the_horizon_decides_a_marginal_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open long from 06:00 day 2 (commission 20), LAST through the final bar: it pays 08:00 and
    16:00 day 2 (24), so its margin is 3976 and LP 36168.67 when the mark dips to 36160 at 16:30
    (with only the 08:00 payment: LP 36156.63, no hit). The scan funds it to its horizon (the
    last minute): hit, net -4020 = commission + whole margin."""
    dip = flat_bars({2 * DAY_MIN + 990: (40000.0, 40000.0, 36160.0, 40000.0)})
    catalog = write_marks(tmp_path, dip)
    ledgers = spy_ledgers(monkeypatch)
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    runner._all_timeframes = {"1m"}
    s = long_trade("S", 2 * DAY_MIN + 360, 2 * DAY_MIN + 361, 40000.0, -20.0, (20.0, 0.0))
    s.events.pop()
    s.ts_closed, s.is_closed, s.signed_qty, s.commission_list = 0, False, 1.0, [20.0]
    m = runner._extract_metrics({}, flat_bt(1), engine_for([s], dip), time.time(), BALANCE)
    assert m.liquidation_count == 1
    assert m.total_funding == pytest.approx(2 * _RATE * 40000.0)
    assert m.total_return == pytest.approx(-4020.0 / BALANCE)
    assert m.total_fees == pytest.approx(20.0 + m.liquidation_forfeited)
    assert ledger_net(ledgers[0]) == pytest.approx(m.total_return * BALANCE, abs=1e-9)


def test_hit_after_the_horizon_keeps_the_scans_funding_and_reconciles_with_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1d execution, LAST bars stop at 12:00 day 2 (a valid horizon in the final bar). The mark
    crashes at 16:30 day 2: price exposure runs to the run end, so it is a hit. Funding is cut at
    min(hit, horizon) = the horizon: 08:00 (12) only, never the 16:00 settlement, in the scan AND
    in the trade: net -4020 (commission 20 + margin 4000 = 12 funding + 3988 loss)."""
    crash = flat_bars({2 * DAY_MIN + 990: (40000.0, 40000.0, 30000.0, 40000.0)})
    catalog = write_marks(tmp_path, crash)
    ledgers = spy_ledgers(monkeypatch)
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    runner._all_timeframes = {"1d"}
    s = long_trade("S", 2 * DAY_MIN + 360, 2 * DAY_MIN + 361, 40000.0, -20.0, (20.0, 0.0))
    s.events.pop()
    s.ts_closed, s.is_closed, s.signed_qty, s.commission_list = 0, False, 1.0, [20.0]
    engine = last_bars_until(engine_for([s], crash), 2 * DAY_MIN + 720)
    m = runner._extract_metrics({}, flat_bt(1), engine, time.time(), BALANCE)
    assert m.liquidation_count == 1
    assert runner.last_liquidation_notes is not None
    assert runner.last_liquidation_notes["events"][0]["ts"] == at(2 * DAY_MIN + 990)
    assert m.total_funding == pytest.approx(_RATE * 40000.0)
    assert m.total_return == pytest.approx(-4020.0 / BALANCE)
    assert ledger_net(ledgers[0]) == pytest.approx(m.total_return * BALANCE, abs=1e-9)


def test_a_hit_keeps_only_the_fallback_settlements_before_it(tmp_path: Path) -> None:
    """No funding archive: every settlement is charged at the fallback rate and flagged. A is
    funded to its NT close (20:00: 08:00 + 16:00) in the scan and liquidated at 08:30, so it is
    charged and flagged for the 08:00 settlement only."""
    from vibe_quant.validation.funding import DEFAULT_FUNDING_RATE_PER_PERIOD

    crash_late = flat_bars({510: (40000.0, 40000.0, 30000.0, 40000.0)})
    catalog = write_marks(tmp_path, crash_late)
    a = long_trade("A", 360, 1200, 38000.0, -2039.0, (20.0, 19.0))
    runner = make_runner(tmp_path, "isolated", catalog=catalog, funding=False)
    m = extract(runner, [a], crash_late, n_positions=1)
    assert m.liquidation_count == 1
    assert m.funding_fallback_settlements == 1
    assert m.total_funding == pytest.approx(DEFAULT_FUNDING_RATE_PER_PERIOD * 40000.0)
    assert m.total_return == pytest.approx(-4020.0 / BALANCE)


def test_a_mark_crash_after_the_run_end_never_liquidates_an_open_position(tmp_path: Path) -> None:
    """No look-ahead: S opens at 23:58 on the last day and is open at the run end; the MARK data
    goes on past the end and crashes to 30000 at 00:30. The run is scored as mode off."""
    calm = flat_bars()
    after = Bars(*(column + [column[0]] * 60 for column in calm))
    after.low[N_BARS + 30] = 30000.0
    catalog = write_marks(tmp_path, after)

    def run(mode: str) -> Any:
        runner = make_runner(tmp_path, mode, catalog=catalog)
        return runner._extract_metrics(
            {}, flat_bt(1), engine_for([open_long("S", N_BARS - 2)], calm), time.time(), BALANCE
        )

    on, off = run("isolated"), run("off")
    assert on.liquidation_count == 0
    assert equal_metrics(on, off)


@pytest.mark.parametrize(("crash_minute", "hit"), [(N_BARS - 1, True), (N_BARS, False)])
def test_a_mark_crash_in_the_final_minute_hits_and_one_at_the_run_end_does_not(
    tmp_path: Path, crash_minute: int, hit: bool
) -> None:
    """S (commission 20) opens 23:58 on the last day. A 30000 mark low in the bar opening at
    23:59 is inside the run: hit, net -4020. The bar opening at the run end is not: S survives,
    marked at 40000 with its 20 exit fee (net -40)."""
    calm = flat_bars()
    marks = Bars(*(column + [column[0]] for column in calm))
    marks.low[crash_minute] = 30000.0
    catalog = write_marks(tmp_path, marks)
    s = open_long("S", N_BARS - 2)
    s.events[0].commission, s.commission_list, s.realized_pnl = 20.0, [20.0], -20.0
    nt = flat_bt(1)
    nt.stats_pnls["USDT"]["PnL% (total)"] = -0.05  # NT realized the entry commission
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    m = runner._extract_metrics({}, nt, engine_for([s], calm), time.time(), BALANCE)
    if hit:
        assert m.liquidation_count == 1
        assert m.total_return == pytest.approx(-4020.0 / BALANCE)
        assert m.total_fees == pytest.approx(20.0 + m.liquidation_forfeited)
    else:
        assert m.liquidation_count == 0
        assert m.total_return == pytest.approx(-40.0 / BALANCE)
        assert m.total_fees == pytest.approx(40.0)


def test_the_settlement_at_the_run_end_is_never_charged_to_a_position_without_last_bars(
    tmp_path: Path,
) -> None:
    """S (no LAST bars) opens 23:58 on the last day; the mark sits at 36150, just above its LP
    36144.58. Charging the 00:00 settlement AT the run end would lift the LP to 36156.63 and
    liquidate it at end_ns. It is outside the run: no hit, so S is an unpriceable survivor."""
    flat = Bars(*([36150.0] * N_BARS for _ in range(4)))
    catalog = write_marks(tmp_path, flat)
    s = open_long("S", N_BARS - 2)
    s.events[0].commission, s.commission_list, s.realized_pnl = 20.0, [20.0], -20.0
    engine = engine_for([s], flat)
    engine.kernel.cache.bars = lambda *_: []
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    with pytest.raises(DataUnavailableError, match=r"BTCUSDT-PERP\.BINANCE.*cannot be valued"):
        runner._extract_metrics({}, flat_bt(1), engine, time.time(), BALANCE)


# ------------------------------------------------------ shared integrity error (A1)


def test_marks_diverge_raises_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I4: a survivor mark that diverges from the horizon the scan funded it to fails the run
    as a LiquidationIntegrityError (today it scores -inf)."""
    from vibe_quant.errors import LiquidationIntegrityError

    catalog = write_marks(tmp_path, CALM)
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    engine = engine_for([open_long("S", 420)], CALM)
    monkeypatch.setattr(
        runner,
        "_run_backtest",
        lambda params, t: runner._extract_metrics(params, bt_result(1), engine, t, BALANCE),
    )

    import vibe_quant.validation.extraction as extraction

    def no_marks(*_a: Any, **_k: Any) -> list[Any]:
        return []  # the scan funded S to its horizon; the run-end marks come back empty

    monkeypatch.setattr(extraction, "mark_open_positions", no_marks)
    with pytest.raises(
        LiquidationIntegrityError,
        match=r"liquidation_mode='isolated': RuntimeError: liquidation survivor marks diverge",
    ):
        runner({})


def wire_runner(
    monkeypatch: pytest.MonkeyPatch,
    runner: NTScreeningRunner,
    engine: SimpleNamespace,
    n_positions: int,
    balance: float = BALANCE,
) -> None:
    """Route ``runner({})`` through ``_extract_metrics`` (the I4 test's seam)."""
    monkeypatch.setattr(
        runner,
        "_run_backtest",
        lambda params, t: runner._extract_metrics(
            params, bt_result(n_positions), engine, t, balance
        ),
    )


def test_i3_fill_at_the_run_end_fails_closed_naming_the_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I3: the overlay's apply step refuses a fill at ts == end_ns; the run fails closed
    through runner({}) and the message names the original ValueError."""
    catalog = write_marks(tmp_path, CALM)
    a = pos_a()
    a.events[1].ts_event = at(N_BARS)  # == end_ns: not before the run end
    runner = make_runner(tmp_path, "isolated", catalog=catalog)
    wire_runner(monkeypatch, runner, engine_for([a], CALM), 1)
    with pytest.raises(
        LiquidationIntegrityError,
        match=r"liquidation_mode='isolated': ValueError: .*is not before the run end",
    ):
        runner({})


def test_i6_unpriceable_survivor_fails_closed_naming_the_data_unavailable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I6: an open survivor with no LAST bar in the final execution bar is nt_runner's own
    DataUnavailableError; runner({}) wraps it and the message names the original class."""
    catalog = write_marks(tmp_path, CALM)
    add_eth_marks(tmp_path, catalog)
    _, engine = two_open_longs_with_btc_last_only()
    runner = make_runner(tmp_path, "cross", catalog=catalog)
    wire_runner(monkeypatch, runner, engine, 2)
    with pytest.raises(
        LiquidationIntegrityError,
        match=r"liquidation_mode='cross': DataUnavailableError: .*cannot be valued",
    ):
        runner({})


# ------------------------------------------------------- R16: the parsed wallet

#: notional 50000 * 39900 = 1.995e9, above BTCUSDT's last tier cap (1.8e9)
TIER_BREAK_QTY = 50_000.0


def test_r16_cross_wallet_is_the_parsed_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R16: the cross scan's wallet is the run's parsed ``initial_balance``.

    ``_run_backtest`` hands ``_extract_metrics`` exactly
    ``float(self._venue_config().starting_balance_usdt)``, and that value must reach
    ``apply_liquidation`` as the wallet — never a hard-coded 1000 or 40000.
    """
    import vibe_quant.liquidation.apply as apply_module

    catalog = write_marks(tmp_path, CALM)
    captured: list[float] = []
    real = apply_module.apply_liquidation

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(float(kwargs["starting_balance"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(apply_module, "apply_liquidation", spy)
    runner = make_runner(tmp_path, "cross", catalog=catalog, starting_balance=10000)
    assert runner._venue_config().starting_balance_usdt == 10000
    extract(
        runner,
        [],
        CALM,
        balance=float(runner._venue_config().starting_balance_usdt),
        n_positions=0,
    )
    assert captured == [10000.0]


def test_r16b_notional_beyond_the_last_tier_cap_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R16b: with liquidation on, a position whose notional exceeds BTCUSDT's last tier
    cap makes the scan raise ``TierTableExceeded``; the run fails closed as
    ``LiquidationIntegrityError`` instead of scoring -inf."""
    catalog = write_marks(tmp_path, CALM)
    big = long_trade("BIG", 420, 540, 40000.0, 0.0)
    big.peak_qty = TIER_BREAK_QTY
    for fill in big.events:
        fill.last_qty = TIER_BREAK_QTY
    runner = make_runner(tmp_path, "cross", catalog=catalog, starting_balance=1_000_000_000)
    wire_runner(
        monkeypatch, runner, engine_for([big], CALM), 1, balance=1_000_000_000.0
    )
    with pytest.raises(
        LiquidationIntegrityError,
        match=r"liquidation_mode='cross': TierTableExceeded: BTCUSDT: notional "
        r"1995000000\.0 >= last cap 1800000000\.0",
    ):
        runner({})
