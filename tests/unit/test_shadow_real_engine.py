"""Shadow tracker acceptance on the real NautilusTrader engine and the real BTC/ETH catalog.

The tracker's unit tests (``test_shadow_tracker.py``) drive a fake ``run_frozen``. Here the
real validation engine runs on an APFS clone of the real catalog and a slim copy of the real
archive (BTCUSDT/ETHUSDT rows since 2026-05-01, copied through a read-only attach), so
nothing under ``data/`` is written.

R-prov (design arch-c7-2): a run that ends earlier must reproduce the longer run's daily
mark-to-market returns on every day except its own last day, which carries the run-end
close-out. The tracker relies on it: it commits through ``C - 1`` and retires a row whose
re-run changes a stored day. Checked in six modes, then end to end through ``update_all``
on three enrolled rows (timed, per-row seconds printed; ``-s`` shows them). Two verdicts run
on the real engine too: a losing row ends futile on its drawdown day, and a cross bust (no
daily series) agrees across run lengths and ends its row futile on the bust day.

Skipped when the catalog series, the archive or the data through 2026-11-08 are missing.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.db.state_manager import StateManager
from vibe_quant.shadow import tracker
from vibe_quant.shadow.coverage import coverage_day
from vibe_quant.shadow.tracker import EnrolRequest, enrol, update_all
from vibe_quant.validation.variant import run_frozen

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.validation.results import ValidationResult

REPO = Path(__file__).resolve().parents[2]
SRC_CATALOG = REPO / "data" / "catalog"
SRC_ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SERIES = ("4-HOUR-LAST", "1-MINUTE-LAST", "1-MINUTE-MARK")
ARCHIVE_SINCE = datetime(2026, 5, 1, tzinfo=UTC)
# The last update below asks for days through 2026-11-07, so coverage must reach C >= 11-08.
NEEDED_C = date(2026, 11, 8)
UPDATE_BUDGET_S = 180.0

DSL: dict[str, object] = {
    "name": "shadow_real_rsi",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 45"], "short": ["rsi > 60"]},
    "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 40"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}
PREREG = (
    "RSI mean reversion on BTC perps keeps a positive forward Sharpe; futile when the "
    "forward Sharpe bound drops below 0.5 after 90 days or drawdown exceeds 30 percent."
)


# --- fixtures -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Market:
    """A private catalog clone and archive copy (both under the test's tmp dir)."""

    catalog: Path
    archive: Path


def _clone_catalog(dst: Path) -> None:
    """APFS clone (``cp -c``) of BTC/ETH instruments and their 4h, 1m and 1m MARK bars."""
    src = SRC_CATALOG.resolve()
    for sym in SYMBOLS:
        inst = f"{sym}-PERP.BINANCE"
        pairs = [
            (src / "data" / "crypto_perpetual" / inst, dst / "data" / "crypto_perpetual" / inst)
        ]
        pairs += [
            (
                src / "data" / "bar" / f"{inst}-{s}-EXTERNAL",
                dst / "data" / "bar" / f"{inst}-{s}-EXTERNAL",
            )
            for s in SERIES
        ]
        for s, d in pairs:
            d.parent.mkdir(parents=True, exist_ok=True)
            done = subprocess.run(["cp", "-c", "-R", str(s), str(d)], capture_output=True)
            if done.returncode != 0:
                pytest.skip(f"cp -c clone unsupported here: {done.stderr!r}")


def _copy_archive(dst: Path) -> None:
    """BTC/ETH 1m klines, 1m mark klines, funding and intervals since ARCHIVE_SINCE."""
    arch = RawDataArchive(dst)
    _ = arch.conn  # creates the schema
    arch.close()
    since = int(ARCHIVE_SINCE.timestamp() * 1000)
    con = sqlite3.connect(f"file:{dst}", uri=True)
    try:
        con.execute("ATTACH DATABASE ? AS src", (f"file:{SRC_ARCHIVE.resolve()}?mode=ro",))
        for table, where in (
            ("raw_klines", "interval = '1m' AND open_time >= ?"),
            ("raw_mark_klines", "interval = '1m' AND open_time >= ?"),
            ("raw_funding_rates", "funding_time >= ?"),
            ("raw_funding_intervals", "funding_time >= ?"),
        ):
            cols = ", ".join(
                r[1] for r in con.execute(f"PRAGMA main.table_info({table})") if r[1] != "id"
            )
            for sym in SYMBOLS:
                con.execute(
                    f"INSERT INTO main.{table} ({cols}) SELECT {cols} FROM src.{table} "
                    f"WHERE symbol = ? AND {where}",
                    (sym, since),
                )
        con.commit()
    finally:
        con.close()


@pytest.fixture(scope="module")
def market(tmp_path_factory: pytest.TempPathFactory) -> Market:
    bars = SRC_CATALOG / "data" / "bar"
    missing = [
        f"{sym}-PERP.BINANCE-{s}-EXTERNAL"
        for sym in SYMBOLS
        for s in SERIES
        if not (bars / f"{sym}-PERP.BINANCE-{s}-EXTERNAL").is_dir()
    ]
    if missing or not SRC_ARCHIVE.exists():
        pytest.skip(f"needs the real catalog series and archive (missing: {missing or 'archive'})")
    root = tmp_path_factory.mktemp("shadow_real")
    m = Market(root / "catalog", root / "raw_data.db")
    _clone_catalog(m.catalog)
    _copy_archive(m.archive)
    c_day, why = coverage_day(
        SYMBOLS,
        {"4h", "1m"},
        catalog=m.catalog,
        archive=m.archive,
        mark=True,
        run_start=date(2026, 6, 1),
    )
    if c_day < NEEDED_C:
        pytest.skip(f"data covers only through C = {c_day} ({why}); needs {NEEDED_C}")
    return m


# --- R-prov in six real modes -------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    symbols: tuple[str, ...]
    timeframe: str
    params: dict[str, object]
    start: str
    short_end: str
    long_end: str
    check: Callable[[ValidationResult, ValidationResult, str], None] = field(
        default=lambda short, long, end: None
    )
    dsl: dict[str, object] = field(default_factory=dict)  # overrides on DSL


def _maker_filled(short: ValidationResult, long: ValidationResult, end: str) -> None:
    for r in (short, long):
        maker = r.notes["maker_entries"]
        assert isinstance(maker, dict) and maker["filled"] > 0, (
            maker
        )  # the post-only path really filled entries


def _liquidation(
    mode: str, *, liquidated: bool
) -> Callable[[ValidationResult, ValidationResult, str], None]:
    def check(short: ValidationResult, long: ValidationResult, end: str) -> None:
        for r in (short, long):
            liq = r.notes["liquidation"]
            assert isinstance(liq, dict) and liq["mode"] == mode, liq
            assert liq["count"] >= 1 if liquidated else liq["count"] == 0, liq
            assert r.mtm_probe is not None and r.mtm_probe.bust_ts is None

    return check


def _held_across(short: ValidationResult, long: ValidationResult, end: str) -> None:
    """A position open at the short end: closed there by the short run, held on by the long."""
    closed = {(t.symbol, t.entry_time) for t in short.trades if t.exit_reason == "end_of_data"}
    assert closed, [(t.entry_time, t.exit_reason) for t in short.trades[-3:]]
    held = {
        (t.symbol, t.entry_time) for t in long.trades if t.entry_time < end <= (t.exit_time or "")
    }
    assert closed & held, (closed, held)


MODES = {
    "4h_market": Mode(("BTCUSDT",), "4h", {}, "2026-06-01", "2026-07-25", "2026-08-01"),
    "1m_market": Mode(("BTCUSDT",), "1m", {}, "2026-08-01", "2026-08-08", "2026-08-15"),
    "4h_post_only": Mode(
        ("BTCUSDT",),
        "4h",
        {"entry_order": "post_only", "maker_offset_ticks": 1},
        "2026-06-01",
        "2026-08-25",
        "2026-09-01",
        _maker_filled,
    ),
    # cross ledger with no liquidation: every cross liquidation is an account bust, which
    # nulls the daily series (see test_cross_bust_same_in_short_and_long)
    "4h_cross_no_liq": Mode(
        ("BTCUSDT",),
        "4h",
        {"liquidation_mode": "cross"},
        "2026-06-01",
        "2026-08-25",
        "2026-09-01",
        _liquidation("cross", liquidated=False),
    ),
    # 125x: liquidations fire (31 in the short run when measured), so the overlay ledger is used
    "1m_isolated": Mode(
        ("BTCUSDT",),
        "1m",
        {"liquidation_mode": "isolated", "leverage": 125},
        "2026-08-01",
        "2026-08-08",
        "2026-08-15",
        _liquidation("isolated", liquidated=True),
    ),
    "4h_held_btc_eth": Mode(
        ("BTCUSDT", "ETHUSDT"),
        "4h",
        {},
        "2026-06-01",
        "2026-08-25",
        "2026-09-01",
        _held_across,
    ),
}


def _run(market: Market, mode: Mode, end: str) -> ValidationResult:
    dsl = {**DSL, **mode.dsl, "timeframe": mode.timeframe}
    result, _ = run_frozen(
        strategy_name="shadow_real_rsi",
        dsl_json=json.dumps(dsl),
        run_config={
            "symbols": list(mode.symbols),
            "parameters": dict(mode.params),
            "start_date": mode.start,
            "end_date": end,
        },
        catalog_path=market.catalog,
        archive_path=market.archive,
    )
    assert not result.engine_force_stopped
    assert result.funding_fallback_settlements == 0
    return result


def _epoch_day(iso: str) -> int:
    return (date.fromisoformat(iso) - date(1970, 1, 1)).days


@pytest.mark.slow(reason="two real NT validation runs per mode on the real catalog, 4-13 s")
@pytest.mark.parametrize("name", list(MODES))
def test_prefix_equal_except_short_last_day(market: Market, name: str) -> None:
    mode = MODES[name]
    short, long = _run(market, mode, mode.short_end), _run(market, mode, mode.long_end)
    assert short.total_trades > 0
    sd = short.mtm_probe.daily if short.mtm_probe is not None else None
    ld = long.mtm_probe.daily if long.mtm_probe is not None else None
    assert sd is not None and ld is not None
    start = _epoch_day(mode.start)
    assert (sd.first_day, ld.first_day) == (start, start)
    n = _epoch_day(mode.short_end) - start
    assert (len(sd), len(ld)) == (n, _epoch_day(mode.long_end) - start)
    # every day the tracker may commit (all but the short run's last) is bit-identical
    assert list(sd.values[: n - 1]) == list(ld.values[: n - 1])
    mode.check(short, long, mode.short_end)
    if name == "4h_held_btc_eth":
        # the excluded day is not a formality: the close-out moves it
        assert sd.values[n - 1] != ld.values[n - 1]


# --- the update job on three real rows ----------------------------------------------------


# On 1m the 4h thresholds trade ~40 times a day; the 1m row uses extremes (fewer trades,
# closer to a real champion's count) so the update measures the engine, not a trade flood.
DSL_1M: dict[str, object] = {
    **DSL,
    "timeframe": "1m",
    "entry_conditions": {"long": ["rsi < 20"], "short": ["rsi > 80"]},
    "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
}
ROWS: tuple[tuple[str, tuple[str, ...], dict[str, object]], ...] = (
    ("4h BTC", ("BTCUSDT",), DSL),
    ("4h BTC+ETH", ("BTCUSDT", "ETHUSDT"), DSL),
    ("1m BTC", ("BTCUSDT",), DSL_1M),
)
ENROL_NOW = datetime(2026, 11, 1, 12, tzinfo=UTC)
BACKDATE = 61  # enrolled 2026-09-01 12:00 -> first forward day F = 2026-09-02
WARMUP = 30  # run_start = 2026-08-03
FIRST = date(2026, 9, 2)


def _timed_run_frozen(log: list[tuple[str, float]]) -> Callable[..., object]:
    real = run_frozen

    def timed(**kwargs: object) -> object:
        t0 = time.monotonic()
        out = real(**kwargs)  # type: ignore[arg-type]
        secs = time.monotonic() - t0
        run_config = kwargs["run_config"]
        assert isinstance(run_config, dict)
        label = (
            f"{'+'.join(run_config['symbols'])} {json.loads(str(kwargs['dsl_json']))['timeframe']}"
        )
        log.append((label, secs))
        print(
            f"[shadow-real] row {label}: engine run {secs:.2f} s to {run_config['end_date']}",
            flush=True,
        )
        return out

    return timed


def _update(
    state: StateManager, market: Market, today: date
) -> tuple[tracker.UpdateSummary, float]:
    t0 = time.monotonic()
    summary = update_all(
        state,
        today=today,
        trigger="cli",
        failed_streams={},
        catalog=market.catalog,
        archive=market.archive,
    )
    return summary, time.monotonic() - t0


@pytest.mark.slow(reason="three enrolled rows, two real update passes (4h, 4h x2, 1m), ~55 s")
def test_update_three_real_rows(
    market: Market, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = StateManager(tmp_path / "state.db")
    log: list[tuple[str, float]] = []
    monkeypatch.setattr(tracker, "run_frozen", _timed_run_frozen(log))
    ids = {
        label: enrol(
            state,
            EnrolRequest(
                name=f"shadow_real {label}",
                source_kind="strategy",
                dsl_json=json.dumps(dsl),
                symbols=syms,
                parameters={},
                pre_registration=PREREG,
                max_dd_limit=1.0,  # no DD verdict: every row must reach pass 3
                warmup_days=WARMUP,
                backdate_days=BACKDATE,
            ),
            now=ENROL_NOW,
            catalog=market.catalog,
            archive=market.archive,
        )
        for label, syms, dsl in ROWS
    }

    def rows() -> dict[str, dict[str, object]]:
        by_id = {r["id"]: r for r in state.list_shadow_strategies()}
        return {label: by_id[i] for label, i in ids.items()}

    for row in rows().values():
        assert (row["first_day"], row["run_start"]) == ("2026-09-02", "2026-08-03")

    # pass 1 (timed): today 11-01 -> C 10-31 -> committed through 10-30
    summary, wall = _update(state, market, date(2026, 11, 1))
    print(
        f"[shadow-real] update of {len(ROWS)} rows: {wall:.2f} s (budget {UPDATE_BUDGET_S:.0f} s)"
    )
    assert summary.runs == 3, summary
    assert wall <= UPDATE_BUDGET_S
    first_pass = {}
    for label, row in rows().items():
        assert (row["state"], row["committed_through"]) == ("tracking", "2026-10-30"), row
        days = state.get_shadow_days(ids[label])
        assert [d for d, _ in days] == list(
            range(_epoch_day("2026-09-02"), _epoch_day("2026-10-31"))
        )
        assert any(r != 0.0 for _, r in days), label  # the row really traded forward
        first_pass[label] = days

    # pass 2, same day: nothing new, no engine run, nothing written
    summary, _ = _update(state, market, date(2026, 11, 1))
    assert summary.runs == 0
    assert sorted(note for _, _, note in summary.rows) == ["no new data since 2026-10-30"] * 3

    # pass 3: today 11-09 -> committed through 11-07; the re-run reproduced every stored day
    # (a changed day retires the row), and the stored ones are untouched
    summary, wall = _update(state, market, date(2026, 11, 9))
    print(f"[shadow-real] second update of {len(ROWS)} rows: {wall:.2f} s")
    assert summary.runs == 3, summary
    for label, row in rows().items():
        assert (row["state"], row["committed_through"]) == ("tracking", "2026-11-07"), row
        days = state.get_shadow_days(ids[label])
        assert len(days) == (date(2026, 11, 7) - FIRST).days + 1
        assert days[: len(first_pass[label])] == first_pass[label]
    assert sorted(label for label, _ in log) == sorted(
        ["BTCUSDT 4h", "BTCUSDT+ETHUSDT 4h", "BTCUSDT 1m"] * 2
    )


# --- futility on the real engine ----------------------------------------------------------


# Trend-chasing RSI on 4h BTC: loses over Sept 2026, so its forward drawdown crosses a 3% limit.
DSL_LOSER: dict[str, object] = {
    **DSL,
    "entry_conditions": {"long": ["rsi > 60"], "short": ["rsi < 40"]},
    "exit_conditions": {"long": ["rsi < 50"], "short": ["rsi > 50"]},
}


@pytest.mark.slow(reason="one enrolled 4h BTC row, one real update pass, ~8 s")
def test_losing_row_goes_futile_on_drawdown_day(market: Market, tmp_path: Path) -> None:
    """Forward drawdown 2.91% on 09-12, 3.32% on 09-14 (probed): futile on 09-14, no day later."""
    state = StateManager(tmp_path / "state.db")
    sid = enrol(
        state,
        EnrolRequest(
            name="shadow_real loser",
            source_kind="strategy",
            dsl_json=json.dumps(DSL_LOSER),
            symbols=("BTCUSDT",),
            parameters={},
            pre_registration=PREREG,
            max_dd_limit=0.03,
            warmup_days=WARMUP,
            backdate_days=29,  # enrolled 2026-09-02 12:00 -> F = 2026-09-03
        ),
        now=datetime(2026, 10, 1, 12, tzinfo=UTC),
        catalog=market.catalog,
        archive=market.archive,
    )
    t0 = time.monotonic()
    summary, _ = _update(state, market, date(2026, 10, 1))
    print(f"[shadow-real] futility update: {time.monotonic() - t0:.2f} s", flush=True)
    assert summary.runs == 1, summary
    row = next(r for r in state.list_shadow_strategies() if r["id"] == sid)
    assert row["first_day"] == "2026-09-03"
    assert (row["state"], row["verdict_day"], row["committed_through"]) == (
        "futile",
        "2026-09-14",
        "2026-09-14",
    ), row
    assert row["state_reason"] == "max drawdown 3.32% > limit 3.00%"
    days = state.get_shadow_days(sid)
    assert [d for d, _ in days] == list(range(_epoch_day("2026-09-03"), _epoch_day("2026-09-15")))


# Cross liquidates only when exposure outruns the stop: ATR sizing at 10x exposure (risk 0.5,
# ATR x 0.5) with a 10% stop. From 07-19 the account busts on 2026-08-20 (probed; 7 trades).
CROSS_PARAMS: dict[str, object] = {
    "liquidation_mode": "cross",
    "leverage": 10,
    "max_position_pct": 10,
    "risk_per_trade": 0.5,
    "sizing_mode": "atr",
    "sizing_atr_multiplier": 0.5,
}
CROSS_DSL: dict[str, object] = {"stop_loss": {"type": "fixed_pct", "percent": 10.0}}
CROSS_BUST = Mode(
    ("BTCUSDT",), "4h", CROSS_PARAMS, "2026-07-19", "2026-08-25", "2026-09-01", dsl=CROSS_DSL
)


@pytest.mark.slow(reason="two real NT runs with a cross bust, then one real update pass, ~25 s")
def test_cross_bust_same_in_short_and_long_and_row_goes_futile(
    market: Market, tmp_path: Path
) -> None:
    """A cross liquidation busts the account and nulls the daily series, so the tracker reads
    the bust instant instead: both run lengths must agree on it, and the row ends futile on
    the bust day (design step 6: a bust on a day >= F)."""
    short = _run(market, CROSS_BUST, CROSS_BUST.short_end)
    long = _run(market, CROSS_BUST, CROSS_BUST.long_end)
    events = []
    for r in (short, long):
        liq = r.notes["liquidation"]
        assert isinstance(liq, dict) and (liq["mode"], liq["count"]) == ("cross", 1), liq
        assert r.mtm_probe is not None and r.mtm_probe.daily is None
        assert r.mtm_probe.bust_ts == liq["bust_ts"] == liq["events"][0]["ts"]
        events.append(liq["events"])
    assert events[0] == events[1]
    bust_ts = short.mtm_probe.bust_ts if short.mtm_probe is not None else None
    assert bust_ts is not None
    assert datetime.fromtimestamp(bust_ts / 1e9, UTC).date() == date(2026, 8, 20)

    # the same row through enrol + update: run_start 07-19, F 08-18, bust 08-20 >= F
    state = StateManager(tmp_path / "state.db")
    sid = enrol(
        state,
        EnrolRequest(
            name="shadow_real cross bust",
            source_kind="strategy",
            dsl_json=json.dumps({**DSL, **CROSS_DSL}),
            symbols=("BTCUSDT",),
            parameters=CROSS_PARAMS,
            pre_registration=PREREG,
            max_dd_limit=1.0,  # only the bust can end it
            warmup_days=WARMUP,
            backdate_days=15,  # enrolled 2026-08-17 12:00 -> F = 2026-08-18
        ),
        now=datetime(2026, 9, 1, 12, tzinfo=UTC),
        catalog=market.catalog,
        archive=market.archive,
    )
    summary, wall = _update(state, market, date(2026, 9, 1))
    print(f"[shadow-real] cross bust update: {wall:.2f} s", flush=True)
    # R14: the run to C shows the bust on D, a re-run to end = D commits D - 2 by the normal rules
    assert summary.runs == 2, summary
    row = next(r for r in state.list_shadow_strategies() if r["id"] == sid)
    assert (row["first_day"], row["run_start"]) == ("2026-08-18", "2026-07-19")
    assert (row["state"], row["state_reason"], row["verdict_day"], row["committed_through"]) == (
        "futile",
        "account bust on 2026-08-20",
        "2026-08-20",
        "2026-08-18",
    ), row
    assert [d for d, _ in state.get_shadow_days(sid)] == [_epoch_day("2026-08-18")]
