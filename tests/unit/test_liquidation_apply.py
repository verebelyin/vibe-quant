"""Run-level liquidation entry point: real MARK bars from a temp catalog (bead vibe-quant-2gyk1.1.6).

A 1.0 BTC long opened at 40000 with a 40000 account at 10x isolated has the liquidation price
``LP = (4000 - 40000) / (0.004 - 1) = 36144.57831325301`` (BTC tier 1: MMR .004, cum 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.data.mark_bars import MarkDataMissing, build_mark_bars
from vibe_quant.liquidation.apply import apply_liquidation, traded_symbols
from vibe_quant.liquidation.overlay import RefinedExitConflict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

MIN = 60_000_000_000
T0_MS = 28_333_333 * 60_000
START_NS = T0_MS * 1_000_000
BTC = "BTCUSDT-PERP.BINANCE"
LP_LONG = 36144.57831325301


class Name(NamedTuple):
    name: str


@dataclass
class Fill:
    ts_event: int
    order_side: Any
    last_qty: float
    last_px: float
    commission: float = 0.0
    order_type: Any = Name("MARKET")


@dataclass
class Pos:
    id: str
    instrument_id: str
    ts_opened: int
    ts_closed: int
    events: list[Fill] = field(default_factory=list)


def write_marks(
    root: Path,
    symbol: str,
    lows: Sequence[float],
    *,
    first_bar: int = 0,
    skip: frozenset[int] = frozenset(),
) -> None:
    """Mark bars open 40000 / high 41000 / close = low, one per minute from ``first_bar``."""
    rows = []
    for i, low in enumerate(lows):
        if i in skip:
            continue
        open_ms = T0_MS + (first_bar + i) * 60_000
        rows.append(
            (open_ms, 40000.0, 41000.0, low, low, 0.0, open_ms + 59_999, 0, 60, 0.0, 0.0)
        )
    archive = RawDataArchive(root / f"{symbol}_archive.db")
    archive.insert_mark_klines(symbol, "1m", rows, "test")
    build_mark_bars(CatalogManager(root / "catalog"), archive, symbol)


def long_btc() -> Pos:
    fill = Fill(START_NS, Name("BUY"), 1.0, 40000.0)
    return Pos("P-1", BTC, START_NS, 0, [fill])


def run(tmp_path: Path, pos: list[Pos], **kw: Any) -> Any:
    args: dict[str, Any] = {
        "mode": "isolated",
        "leverage": 10.0,
        "starting_balance": 40000.0,
        "catalog_path": tmp_path / "catalog",
        "start_ns": START_NS,
        "end_ns": START_NS + 10 * MIN,
        "execution_bar_ns": MIN,
        "funding_by_position": {},
        "cash_by_position": {},
    }
    args.update(kw)
    return apply_liquidation(pos, **args)


def test_isolated_long_through_the_lp_is_liquidated_at_the_lp(tmp_path: Path) -> None:
    write_marks(tmp_path, "BTCUSDT", [39000, 38000, 36000, 37000, 37000] + [37000] * 5)
    out = run(tmp_path, [long_btc()])
    key = f"P-1@{START_NS}"
    assert not out.is_identity
    hit = out.hits_by_key[key]
    assert hit.ts == START_NS + 2 * MIN
    assert hit.price == pytest.approx(LP_LONG)
    assert out.exit_override == {key: (hit.ts, hit.price)}
    assert out.ledgers is not None
    assert out.result.mode == "isolated" and out.result.leverage == 10.0


def test_apply_liquidation_plumbs_the_refined_scan_timestamps(tmp_path: Path) -> None:
    """``apply_liquidation`` passes ``scan_fill_ts`` to the scan (vjno2 R24b): a partial SELL
    refined before the run's hit, with its NT ts after it, makes the scanner (0.5 open) and the
    truncated ledger (full lot) disagree, so the run fails closed with ``RefinedExitConflict``.
    Without the refinement the same run completes and hits at the second mark bar."""
    write_marks(tmp_path, "BTCUSDT", [39000, 38000, 36000, 37000, 37000] + [37000] * 5)
    pos = long_btc()
    nt_ts = START_NS + 3 * MIN
    pos.events.append(Fill(nt_ts, Name("SELL"), 0.5, 36000.0, order_type=Name("STOP_MARKET")))
    key = f"P-1@{START_NS}"
    r = START_NS + MIN
    with pytest.raises(RefinedExitConflict) as excinfo:
        run(tmp_path, [pos], scan_fill_ts={key: {1: r}})
    msg = str(excinfo.value)
    assert key in msg and str(r) in msg and str(nt_ts) in msg
    out = run(tmp_path, [pos])
    assert [h.ts for h in out.result.hits] == [START_NS + 2 * MIN]


def test_no_crossing_is_identity(tmp_path: Path) -> None:
    write_marks(tmp_path, "BTCUSDT", [39000, 38000, 37500, 37000, 37000] + [37000] * 5)
    out = run(tmp_path, [long_btc()])
    assert out.is_identity and out.ledgers is None and out.exit_override == {}


def test_missing_symbol_series_raises_naming_the_symbol(tmp_path: Path) -> None:
    write_marks(tmp_path, "ETHUSDT", [3000.0] * 10)  # BTC has no MARK bars
    with pytest.raises(MarkDataMissing, match="BTCUSDT"):
        run(tmp_path, [long_btc()])


def test_series_not_covering_the_window_raises(tmp_path: Path) -> None:
    """The MARK series must cover the whole run window on both sides."""
    late_start = tmp_path / "late_start"
    late_start.mkdir()
    write_marks(late_start, "BTCUSDT", [39000.0] * 9, first_bar=1)
    with pytest.raises(MarkDataMissing, match="BTCUSDT.*starts"):
        run(late_start, [long_btc()])
    early_end = tmp_path / "early_end"
    early_end.mkdir()
    write_marks(early_end, "BTCUSDT", [39000.0] * 5)
    with pytest.raises(MarkDataMissing, match="BTCUSDT.*ends"):
        run(early_end, [long_btc()])


def test_hole_longer_than_an_hour_in_the_window_raises(tmp_path: Path) -> None:
    write_marks(tmp_path, "BTCUSDT", [39000.0] * 200, skip=frozenset(range(10, 100)))
    with pytest.raises(MarkDataMissing, match="BTCUSDT.*gap"):
        run(tmp_path, [long_btc()], end_ns=START_NS + 200 * MIN)


def test_only_traded_symbols_need_mark_data(tmp_path: Path) -> None:
    write_marks(tmp_path, "BTCUSDT", [39000.0] * 10)
    # ETH is not traded in this run: no ETH series is required
    out = run(tmp_path, [long_btc()])
    assert out.is_identity
    assert traded_symbols([long_btc()]) == ["BTCUSDT"]
    assert traded_symbols([]) == []


def test_unknown_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="liquidation mode"):
        run(tmp_path, [long_btc()], mode="Cross")


def test_empty_window_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty liquidation window"):
        run(tmp_path, [long_btc()], end_ns=START_NS)


def test_a_crash_after_the_run_end_is_never_a_liquidation(tmp_path: Path) -> None:
    """No look-ahead: the long is still open at the run end (minute 10) and the MARK series
    goes on, crashing through the LP at minute 15. The scan stops at the run end: no hit."""
    write_marks(tmp_path, "BTCUSDT", [39000.0] * 15 + [30000.0] * 5)
    out = run(tmp_path, [long_btc()])
    assert out.is_identity and out.result.hits == [] and out.exit_override == {}
    # control: the same crash inside the window is a hit
    later = run(tmp_path, [long_btc()], end_ns=START_NS + 20 * MIN)
    assert [h.ts for h in later.result.hits] == [START_NS + 15 * MIN]


def test_a_bar_before_the_window_still_covers_a_missing_first_bar(tmp_path: Path) -> None:
    """The bar at the window start is missing (a gap under an hour): the last bar before the
    window is kept so the entry at the start is still priced, and nothing is hit."""
    write_marks(tmp_path, "BTCUSDT", [39000.0] * 13, first_bar=-3, skip=frozenset({3}))
    out = run(tmp_path, [long_btc()])
    assert out.is_identity


@pytest.mark.parametrize(("crash_bar", "hit"), [(9, True), (10, False)])
def test_the_final_minute_is_scanned_and_the_bar_opening_at_the_run_end_is_not(
    tmp_path: Path, crash_bar: int, hit: bool
) -> None:
    """Window [start, start + 10 min): the bar opening at minute 9 is the run's last minute (a
    crash there is a hit); the bar opening at minute 10 = end_ns is outside the run."""
    lows = [39000.0] * 11
    lows[crash_bar] = 30000.0
    write_marks(tmp_path, "BTCUSDT", lows)
    out = run(tmp_path, [long_btc()])
    assert [h.ts for h in out.result.hits] == ([START_NS + 9 * MIN] if hit else [])


@pytest.mark.parametrize(
    ("mode", "field", "offset"),
    [("isolated", "funding_by_position", 0), ("cross", "cash_by_position", 30 * MIN)],
)
def test_a_cash_flow_at_or_after_the_run_end_is_refused(
    tmp_path: Path, mode: str, field: str, offset: int
) -> None:
    """The window is half-open: a settlement or cash flow at ts >= end_ns is never charged. One
    handed to the scan is a caller bug (it would liquidate on money the run never paid): raise."""
    write_marks(tmp_path, "BTCUSDT", [39000.0] * 10)
    end_ns = START_NS + 10 * MIN
    flows = {f"P-1@{START_NS}": [(end_ns + offset, -40000.0)]}
    with pytest.raises(ValueError, match=rf"P-1@{START_NS}.*{end_ns + offset}.*run end {end_ns}"):
        run(tmp_path, [long_btc()], mode=mode, **{field: flows})
    # one ns earlier it is inside the window and is charged: the drain liquidates
    flows = {f"P-1@{START_NS}": [(end_ns - 1, -40000.0)]}
    assert run(tmp_path, [long_btc()], mode=mode, **{field: flows}).result.hits


def long_then_late_sell(sell_ts: int) -> Pos:
    """1 BTC long @40000 at the start, then a raw SELL 0.5 @36000 (commission 9) at ``sell_ts``."""
    pos = long_btc()
    pos.events.append(Fill(sell_ts, Name("SELL"), 0.5, 36000.0, 9.0))
    return pos


@pytest.mark.parametrize("offset", [0, 120 * MIN])
def test_a_fill_at_or_after_the_run_end_is_refused(tmp_path: Path, offset: int) -> None:
    """Cross, wallet 4000, mark low 36160 all run (LP 36144.58: no bust in the window). A partial
    close at 36000 at/after the run end would book -2009 and bust the account there: such a fill
    is outside [start_ns, end_ns), a caller bug, so the scan refuses it instead of replaying it.
    The check runs before the scan: 2 h late the scanner itself would fail first (a mark gap
    over an hour) and hide the caller's bug."""
    write_marks(tmp_path, "BTCUSDT", [36160.0] * 10)
    end_ns = START_NS + 10 * MIN
    with pytest.raises(
        ValueError, match=rf"P-1@{START_NS}.*fill at ts {end_ns + offset}.*run end {end_ns}"
    ):
        run(tmp_path, [long_then_late_sell(end_ns + offset)], mode="cross", starting_balance=4000.0)


def test_a_fill_one_millisecond_before_the_run_end_is_scanned(tmp_path: Path) -> None:
    """The same partial close at end - 1 ms is inside the run and replayed: wallet 4000 - 2000
    - 9 = 1991; the remaining 0.5 at the 36160 mark is -1920, so MB 71 <= MM 0.5 * 36160 * .004
    = 72.32 and the account busts at the fill instant, losing the whole 1991."""
    write_marks(tmp_path, "BTCUSDT", [36160.0] * 10)
    end_ns = START_NS + 10 * MIN
    out = run(
        tmp_path, [long_then_late_sell(end_ns - 1_000_000)], mode="cross", starting_balance=4000.0
    )
    (hit,) = out.result.hits
    assert hit.ts == out.result.bust_ts == end_ns - 1_000_000
    assert (hit.qty, hit.margin_balance) == (0.5, pytest.approx(71.0))
    assert hit.maint_margin == pytest.approx(72.32)
    assert hit.loss == pytest.approx(1991.0)


@pytest.mark.parametrize("late", ["opening", "closing"])
def test_an_opening_or_closing_fill_at_or_after_the_run_end_is_refused(
    tmp_path: Path, late: str
) -> None:
    """Every fill is checked, the position's FIRST one included. Opening: a BUY 1 @40000
    (commission 20) at/after the run end would be scanned on the 36160 mark and wipe out the
    4000 wallet there. Closing: the long sold in full @36000 at/after the run end."""
    write_marks(tmp_path, "BTCUSDT", [36160.0] * 10)
    end_ns = START_NS + 10 * MIN
    ts = end_ns
    if late == "opening":
        pos = Pos("P-2", BTC, ts, 0, [Fill(ts, Name("BUY"), 1.0, 40000.0, 20.0)])
    else:
        pos = long_btc()
        pos.ts_closed = ts
        pos.events.append(Fill(ts, Name("SELL"), 1.0, 36000.0, 18.0))
    key = f"{pos.id}@{pos.ts_opened}"
    with pytest.raises(
        ValueError, match=rf"{key}: fill at ts {ts} is not before the run end {end_ns}"
    ):
        run(tmp_path, [pos], mode="cross", starting_balance=4000.0)

