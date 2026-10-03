"""Tests for paper-vs-validation trade reconciliation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TCH003

import pytest

from vibe_quant.reconciliation import (
    ReconciliationReport,
    Trade,
    load_trades,
    reconcile,
)


def _make_event(
    event: str,
    ts: datetime,
    run_id: str = "r",
    strategy: str = "s",
    **data: object,
) -> dict[str, object]:
    return {
        "ts": ts.isoformat(),
        "event": event,
        "run_id": run_id,
        "strategy": strategy,
        "data": data,
    }


def _write_log(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for e in events:
            fp.write(json.dumps(e) + "\n")


def _trade(
    pid: str,
    side: str,
    entry: datetime,
    symbol: str = "BTCUSDT",
    entry_price: float = 100.0,
    exit_price: float = 110.0,
    quantity: float = 1.0,
    net_pnl: float = 10.0,
    exit_offset_min: int = 30,
) -> Trade:
    return Trade(
        position_id=pid,
        symbol=symbol,
        side=side,
        entry_time=entry,
        exit_time=entry + timedelta(minutes=exit_offset_min),
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        net_pnl=net_pnl,
        gross_pnl=net_pnl + 0.1,
        exit_reason="take_profit",
    )


# ---------------------------------------------------------------------------
# load_trades
# ---------------------------------------------------------------------------


def test_load_trades_pairs_open_and_close(tmp_path: Path) -> None:
    entry = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    exit_ts = entry + timedelta(minutes=15)
    events = [
        _make_event(
            "POSITION_OPEN",
            entry,
            position_id="p1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.5,
            entry_price=50000.0,
            leverage=1,
        ),
        _make_event(
            "POSITION_CLOSE",
            exit_ts,
            position_id="p1",
            symbol="BTCUSDT",
            exit_price=51000.0,
            gross_pnl=150.0,
            net_pnl=145.0,
            exit_reason="take_profit",
        ),
    ]
    _write_log(tmp_path / "r1.jsonl", events)

    trades = load_trades("r1", base_path=tmp_path)
    assert len(trades) == 1
    t = trades[0]
    assert t.position_id == "p1"
    assert t.side == "LONG"
    assert t.entry_price == 50000.0
    assert t.exit_price == 51000.0
    assert t.net_pnl == 145.0
    assert t.exit_time == exit_ts


def test_load_trades_drops_orphan_open(tmp_path: Path) -> None:
    entry = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    events = [
        _make_event(
            "POSITION_OPEN",
            entry,
            position_id="orphan",
            symbol="BTCUSDT",
            side="LONG",
        )
    ]
    _write_log(tmp_path / "r1.jsonl", events)

    trades = load_trades("r1", base_path=tmp_path)
    assert trades == []


def test_load_trades_drops_orphan_close(tmp_path: Path) -> None:
    exit_ts = datetime(2026, 4, 1, 12, 15, tzinfo=UTC)
    events = [
        _make_event(
            "POSITION_CLOSE",
            exit_ts,
            position_id="no_open",
            exit_price=100.0,
            net_pnl=5.0,
        )
    ]
    _write_log(tmp_path / "r1.jsonl", events)

    trades = load_trades("r1", base_path=tmp_path)
    assert trades == []


def test_load_trades_missing_log_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trades("does_not_exist", base_path=tmp_path)


def test_load_trades_ignores_malformed_lines(tmp_path: Path) -> None:
    entry = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    path = tmp_path / "r1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("not valid json\n")
        fp.write(
            json.dumps(
                _make_event("POSITION_OPEN", entry, position_id="p1", symbol="X", side="LONG")
            )
            + "\n"
        )
        fp.write(
            json.dumps(
                _make_event(
                    "POSITION_CLOSE",
                    entry + timedelta(minutes=1),
                    position_id="p1",
                    exit_price=101.0,
                    net_pnl=1.0,
                )
            )
            + "\n"
        )
        fp.write("\n")  # blank line

    trades = load_trades("r1", base_path=tmp_path)
    assert len(trades) == 1


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------


def test_reconcile_exact_match() -> None:
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    paper = [_trade("p1", "LONG", t)]
    validation = [_trade("v1", "LONG", t)]

    report = reconcile(paper, validation)
    assert len(report.matches) == 1
    assert not report.paper_only
    assert not report.validation_only
    assert report.parity_rate() == 1.0


def test_reconcile_within_tolerance() -> None:
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    paper = [_trade("p1", "LONG", t)]
    validation = [_trade("v1", "LONG", t + timedelta(seconds=60))]

    report = reconcile(paper, validation, tolerance_seconds=120)
    assert len(report.matches) == 1
    # Validation was 60s later → paper "anticipated", slippage captured
    assert report.matches[0].validation.entry_time > report.matches[0].paper.entry_time


def test_reconcile_outside_tolerance_splits() -> None:
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    paper = [_trade("p1", "LONG", t)]
    validation = [_trade("v1", "LONG", t + timedelta(seconds=300))]

    report = reconcile(paper, validation, tolerance_seconds=120)
    assert len(report.matches) == 0
    assert len(report.paper_only) == 1
    assert len(report.validation_only) == 1


def test_reconcile_symbol_mismatch_never_matches() -> None:
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    paper = [_trade("p1", "LONG", t, symbol="BTCUSDT")]
    validation = [_trade("v1", "LONG", t, symbol="ETHUSDT")]

    report = reconcile(paper, validation)
    assert len(report.matches) == 0
    assert len(report.paper_only) == 1
    assert len(report.validation_only) == 1


def test_reconcile_picks_closest_when_ambiguous() -> None:
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    paper = [_trade("p1", "LONG", t)]
    validation = [
        _trade("v_far", "LONG", t + timedelta(seconds=90)),
        _trade("v_close", "LONG", t + timedelta(seconds=10)),
    ]

    report = reconcile(paper, validation, tolerance_seconds=120)
    assert len(report.matches) == 1
    assert report.matches[0].validation.position_id == "v_close"
    assert len(report.validation_only) == 1


def test_reconcile_mean_slippage_and_pnl_delta() -> None:
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    paper = [
        _trade("p1", "LONG", t, entry_price=100.5, net_pnl=9.5),
        _trade("p2", "LONG", t + timedelta(minutes=5), entry_price=200.3, net_pnl=19.0),
    ]
    validation = [
        _trade("v1", "LONG", t, entry_price=100.0, net_pnl=10.0),
        _trade("v2", "LONG", t + timedelta(minutes=5), entry_price=200.0, net_pnl=20.0),
    ]

    report = reconcile(paper, validation)
    assert len(report.matches) == 2
    # Slippage (paper - validation): avg of 0.5 and 0.3 = 0.4
    assert report.mean_entry_slippage() == pytest.approx(0.4, abs=1e-9)
    # PnL delta: avg of -0.5 and -1.0 = -0.75
    assert report.mean_pnl_delta() == pytest.approx(-0.75, abs=1e-9)


def test_reconcile_side_disagreement_is_counted() -> None:
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    paper = [_trade("p1", "LONG", t)]
    validation = [_trade("v1", "SHORT", t)]

    report = reconcile(paper, validation)
    assert len(report.matches) == 1
    assert report.side_disagreements() == 1


def test_to_dict_is_json_serializable() -> None:
    report = ReconciliationReport(paper_run="paper_1", validation_run="1")
    blob = json.dumps(report.to_dict())
    parsed = json.loads(blob)
    assert parsed["paper_run"] == "paper_1"
    assert parsed["counts"]["matched"] == 0
    assert parsed["metrics"]["parity_rate"] is None


def test_empty_inputs_give_no_data_not_full_parity() -> None:
    """Audit: zero trades reported 100% parity."""
    report = reconcile([], [])
    assert report.parity_rate() is None
    assert report.mean_pnl_delta() is None
    assert report.mean_entry_slippage() is None
    assert "n/a (no data)" in report.summary()


def test_short_entry_slippage_sign_is_adverse_positive() -> None:
    """Audit: shorts had the slippage sign inverted.

    Paper sold @101 vs validation @100 -> better fill -> slippage -1.
    Paper bought @101 vs validation @100 -> worse fill -> slippage +1.
    """
    t = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    short = reconcile(
        [_trade("p", "SHORT", t, entry_price=101.0)], [_trade("v", "SHORT", t, entry_price=100.0)]
    )
    assert short.matches[0].entry_slippage == pytest.approx(-1.0)
    long_ = reconcile(
        [_trade("p", "LONG", t, entry_price=101.0)], [_trade("v", "LONG", t, entry_price=100.0)]
    )
    assert long_.matches[0].entry_slippage == pytest.approx(1.0)
    both = reconcile(
        [
            _trade("p1", "SHORT", t, entry_price=99.0),
            _trade("p2", "LONG", t + timedelta(minutes=5), entry_price=101.0),
        ],
        [
            _trade("v1", "SHORT", t, entry_price=100.0),
            _trade("v2", "LONG", t + timedelta(minutes=5), entry_price=100.0),
        ],
    )
    # Both fills were 1.0 worse than validation.
    assert both.mean_entry_slippage() == pytest.approx(1.0)


def test_validation_trades_load_from_db(tmp_path: Path) -> None:
    """Audit: validation POSITION_* events have position_id='' -> all dropped.

    Validation trades now come from the trades table (with venue timestamps)."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.reconciliation import load_validation_trades

    sm = StateManager(tmp_path / "s.db")
    sid = sm.create_strategy("s", {"name": "s"})
    rid = sm.create_backtest_run(
        sid, "validation", ["BTCUSDT"], "4h", "2024-01-01", "2024-12-31", {}
    )
    sm.save_trades_batch(
        rid,
        [
            {
                "symbol": "BTCUSDT-PERP.BINANCE",
                "direction": "SHORT",
                "leverage": 10,
                "entry_time": "2024-02-07T00:00:59.999000+00:00",
                "exit_time": "2024-02-07T04:00:59.999000+00:00",
                "entry_price": 43078.3,
                "exit_price": 42793.3,
                "quantity": 0.007,
                "gross_pnl": 1.995,
                "net_pnl": 1.6778,
                "exit_reason": "signal",
            },
            {
                "symbol": "BTCUSDT-PERP.BINANCE",
                "direction": "LONG",
                "leverage": 10,
                "entry_time": "2024-03-01T00:00:59.999000+00:00",
                "exit_time": None,
                "entry_price": 60000.0,
                "exit_price": None,
                "quantity": 0.005,
                "gross_pnl": None,
                "net_pnl": None,
                "exit_reason": None,
            },
        ],
    )
    trades = load_validation_trades(sm, rid)
    sm.close()
    assert len(trades) == 1, "open-at-end trade is not comparable"
    t = trades[0]
    assert t.side == "SHORT" and t.symbol == "BTCUSDT-PERP.BINANCE"
    assert t.entry_time == datetime(2024, 2, 7, 0, 0, 59, 999000, tzinfo=UTC)
    assert t.net_pnl == pytest.approx(1.6778)
    paper = _trade("p", "SHORT", t.entry_time + timedelta(seconds=30), symbol=t.symbol)
    report = reconcile([paper], trades)
    assert report.parity_rate() == 1.0
