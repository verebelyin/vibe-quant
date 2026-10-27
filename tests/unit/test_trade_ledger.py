"""Shared trade-ledger helpers: NT stat parsing, return moments, position split.

Bead vibe-quant-rabmo.2 L4 moves the duplicated copies out of the screening runner
(``NTScreeningRunner._extract_metrics`` and its private helpers) and the validation
extractor (``extract_stats`` / ``_extract_return_moments``) into
``vibe_quant.trade_ledger``. These tests pin the behaviour the move must preserve:

* the n<4, zero-notional, flat and failing-engine cases of the return moments (R10),
* the cutoff that returns BEFORE any notional is read (r3 F11),
* the per-tier NT stat parse: key sets verbatim, non-numeric values log a WARNING in
  screening and a DEBUG in validation, and only validation reads Avg Winner/Loser (R11),
* the closed / still-open position split.

L5 pins the one cost/accumulate loop's per-tier values (``TierRules``): the fee
association (R5), the liquidation overlay's per-leg cash shape and the screening
path's rule that never reads a notional. The open-mark PnL association (R14), the
same-day cash order (R15) and the summary order (R16) are ONE rule in both tiers
(validation's, bead nxmxd.3).

The literal expectations were measured from the base methods before they were deleted.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.fake_engine import FakePosition, fake_bt_result, fake_engine
from vibe_quant.metrics import (
    daily_balance_returns,
    profit_factor,
    trade_win_rate,
)
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.trade_ledger import (
    SCREENING_RULES,
    VALIDATION_RULES,
    NtStats,
    PositionCosts,
    accumulate,
    apply_validation_moments,
    closed_positions,
    closed_returns,
    cost_positions,
    moments_from_returns,
    parse_nt_stats,
    position_key,
    screening_liquidation_moments,
    screening_liquidation_trades,
    screening_moments,
    split_positions,
    validation_engine_moments,
    validation_liquidated_run,
)
from vibe_quant.validation.extraction import OpenPositionMark, extract_stats
from vibe_quant.validation.funding import FundingAccrual
from vibe_quant.validation.results import ValidationResult

BTC = "BTCUSDT-PERP.BINANCE"
DAY = 86_400_000_000_000
NO_ARCHIVE = "/nonexistent/vq_test_funding_archive.db"


def _closed(pid: str, *, qty: float = 1.0) -> FakePosition:
    """A closed trade whose per-trade return is ``1.0 / (1.0 * 10.0) = 0.1``."""
    return FakePosition.closed(
        pid, BTC, entry_ns=0, exit_ns=DAY, entry_price=10.0, exit_price=11.0, qty=qty
    )


def _pos(pid: str, ret: float) -> FakePosition:
    """A closed position with exactly the per-trade return ``ret``."""
    return FakePosition(
        id=pid,
        instrument_id=BTC,
        ts_opened=0,
        ts_closed=DAY,
        events=[],
        realized_pnl=ret * 10.0,
        avg_px_open=10.0,
        peak_qty=1.0,
    )


class _RaisingNotional:
    """A closed position whose notional read raises: the n<4 cutoff must precede it."""

    is_closed = True
    is_open = False
    avg_px_open = 10.0
    realized_pnl = 1.0

    @property
    def peak_qty(self) -> float:
        raise AssertionError("peak_qty read before the n<4 cutoff")


def _runner() -> NTScreeningRunner:
    return NTScreeningRunner(
        dsl_dict={},
        symbols=["BTCUSDT"],
        start_date="",
        end_date="",
        funding_archive_path=NO_ARCHIVE,
    )


def test_screening_moments_cases() -> None:
    """<4 closed, four with one zero notional, and a flat series, pinned literally."""
    assert screening_moments(fake_engine([_closed(f"p{i}") for i in range(3)])) == (
        0.0,
        3.0,
        (),
    )

    # 4 closed positions, one of them zero-notional: 3 usable returns < 4 -> defaults,
    # but the usable returns are still reported (R10).
    mixed = [_closed(f"p{i}") for i in range(3)] + [_closed("zero", qty=0.0)]
    assert screening_moments(fake_engine(mixed)) == (0.0, 3.0, (0.1, 0.1, 0.1))

    flat = [_closed(f"p{i}") for i in range(5)]
    assert screening_moments(fake_engine(flat)) == (0.0, 3.0, (0.1,) * 5)


def test_validation_moments_cases_leave_the_defaults() -> None:
    """The validation twin never sets trade_returns and leaves 0.0 / 3.0 on each case."""
    cases = [
        [_closed(f"p{i}") for i in range(3)],
        [_closed(f"p{i}") for i in range(3)] + [_closed("zero", qty=0.0)],
        [_closed(f"p{i}") for i in range(5)],
    ]
    for positions in cases:
        result = ValidationResult()
        validation_engine_moments(result, fake_engine(positions))
        assert (result.skewness, result.kurtosis) == (0.0, 3.0)
        assert result.trade_returns == ()


def test_validation_engine_moments_sets_the_literal_values() -> None:
    """A varying four-trade series sets the bias-corrected skew/kurtosis (base values)."""
    result = ValidationResult()
    validation_engine_moments(
        result, fake_engine([_pos("a", 0.1), _pos("b", -0.2), _pos("c", 0.05), _pos("d", 0.3)])
    )
    assert (result.skewness, result.kurtosis) == (-0.3557, 4.2821)
    assert result.trade_returns == ()


def test_engine_failure_logs_the_warning_and_returns_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broken: Any = SimpleNamespace()
    with caplog.at_level(logging.WARNING):
        assert screening_moments(broken) == (0.0, 3.0, ())
        result = ValidationResult()
        validation_engine_moments(result, broken)
    assert (result.skewness, result.kurtosis) == (0.0, 3.0)
    assert caplog.text.count("Could not compute return moments") == 2


def test_moments_cutoff_reads_no_notional(caplog: pytest.LogCaptureFixture) -> None:
    """r3 F11: one closed position is below the cutoff, so its notional is never read."""
    engine = fake_engine([_RaisingNotional()])  # type: ignore[list-item]
    with caplog.at_level(logging.DEBUG):
        assert screening_moments(engine) == (0.0, 3.0, ())
        result = ValidationResult()
        validation_engine_moments(result, engine)
    assert (result.skewness, result.kurtosis) == (0.0, 3.0)
    assert "Could not compute return moments" not in caplog.text


def test_moments_from_returns_is_the_three_tuple_the_base_method_returned() -> None:
    """synth.py unpacks three values: the tuple shape and the rounding stay put."""
    assert moments_from_returns([0.1, 0.2, 0.3, 0.4]) == (
        0.0,
        1.8,
        (0.1, 0.2, 0.3, 0.4),
    )
    assert moments_from_returns([0.1, -0.2, 0.05, 0.3]) == (
        -0.3557,
        4.2821,
        (0.1, -0.2, 0.05, 0.3),
    )
    assert moments_from_returns([0.1] * 5) == (0.0, 3.0, (0.1,) * 5)
    assert moments_from_returns([0.5]) == (0.0, 3.0, (0.5,))


def test_screening_liquidation_moments_cutoff_and_values() -> None:
    returns = [0.1, 0.2, 0.3, 0.4]
    assert screening_liquidation_moments(returns, 4) == (0.0, 1.8, (0.1, 0.2, 0.3, 0.4))
    assert screening_liquidation_moments([0.1, 0.2, 0.3], 3) == (0.0, 3.0, ())
    assert screening_liquidation_moments([0.1] * 5, 5) == (0.0, 3.0, (0.1,) * 5)


def test_closed_positions_and_returns_split_the_notional_read() -> None:
    """``closed_positions`` touches no notional; ``closed_returns`` does."""
    raising = _RaisingNotional()
    closed = closed_positions(fake_engine([raising, _closed("p1")]))  # type: ignore[list-item]
    assert closed[0] is raising
    assert len(closed) == 2

    assert closed_returns([_closed("p1"), _closed("zero", qty=0.0)]) == [0.1]


def test_apply_validation_moments_sets_the_result() -> None:
    result = ValidationResult()
    apply_validation_moments(result, [0.1, 0.2, 0.3, 0.4])
    assert (result.skewness, result.kurtosis) == (0.0, 1.8)

    short = ValidationResult()
    apply_validation_moments(short, [0.1, 0.2, 0.3])
    assert (short.skewness, short.kurtosis) == (0.0, 3.0)


def test_split_positions_returns_closed_and_open() -> None:
    closed = _closed("closed")
    open_pos = FakePosition.open("open", BTC, entry_ns=0, entry_price=10.0)
    snapshot = _closed("snapshot")
    cache = SimpleNamespace(
        positions=lambda: [closed, open_pos],
        position_snapshots=lambda: [snapshot],
    )
    engine = SimpleNamespace(kernel=SimpleNamespace(cache=cache))

    found_closed, still_open = split_positions(engine)
    assert found_closed == [closed, snapshot]
    assert still_open == [open_pos]


def test_split_positions_engine_failure_warns_and_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert split_positions(SimpleNamespace()) == ([], [])
    assert "Could not read positions from engine cache" in caplog.text


STATS_A = {
    "PnL% (total)": 12.5,
    "Sharpe Ratio (252 days)": 0.5,
    "Sortino Ratio (252 days)": 0.25,
    "Max Drawdown": -0.4,
    "Win Rate": 0.6,
    "Avg Winner": 3.0,
    "Avg Loser": -2.0,
}
STATS_B = {
    "PnL% (total)": -3.0,
    "Sharpe Ratio (252 days)": 2.0,
    "Sortino Ratio (252 days)": -1.5,
    "Max Drawdown": 0.1,
    "Win Rate": 0.25,
    "Avg Winner": 7.0,
    "Avg Loser": -1.5,
}


def _parse(stats: dict[str, Any], tier: str) -> NtStats:
    bt = fake_bt_result(stats_pnls={"USDT": stats}, stats_returns={})
    return parse_nt_stats(bt, tier=tier)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        (
            STATS_A,
            NtStats(
                total_return=0.125,
                sharpe=0.5,
                sortino=0.25,
                max_drawdown=0.4,
                win_rate=0.6,
                avg_win=3.0,
                avg_loss=-2.0,
            ),
        ),
        (
            STATS_B,
            NtStats(
                total_return=-0.03,
                sharpe=2.0,
                sortino=-1.5,
                max_drawdown=0.1,
                win_rate=0.25,
                avg_win=7.0,
                avg_loss=-1.5,
            ),
        ),
    ],
    ids=["A", "B"],
)
def test_parse_nt_stats_validation_reads_every_field(
    stats: dict[str, Any], expected: NtStats
) -> None:
    assert _parse(stats, "validation") == expected


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        (STATS_A, NtStats(0.125, 0.5, 0.25, 0.4, 0.6)),
        (STATS_B, NtStats(-0.03, 2.0, -1.5, 0.1, 0.25)),
    ],
    ids=["A", "B"],
)
def test_parse_nt_stats_screening_never_reads_avg_winner_or_loser(
    stats: dict[str, Any], expected: NtStats
) -> None:
    parsed = _parse(stats, "screening")
    assert parsed == expected
    assert (parsed.avg_win, parsed.avg_loss) == (None, None)


def test_parse_nt_stats_absent_keys_are_none() -> None:
    bt = fake_bt_result(stats_pnls={}, stats_returns={})
    assert parse_nt_stats(bt, tier="screening") == NtStats()
    assert parse_nt_stats(bt, tier="validation") == NtStats()


def _screening_stats(stats_pnls: dict[str, dict[str, Any]]) -> Any:
    runner = _runner()
    bt = fake_bt_result(stats_pnls=stats_pnls, stats_returns={})
    return runner._extract_metrics({}, bt, fake_engine([]), 0.0, starting_balance=1000.0)


def _validation_stats(stats_pnls: dict[str, dict[str, Any]]) -> ValidationResult:
    result = ValidationResult()
    extract_stats(result, fake_bt_result(stats_pnls=stats_pnls, stats_returns={}))
    return result


@pytest.mark.parametrize(
    ("stats", "total_return", "sharpe", "sortino"),
    [
        (STATS_A, 0.125, 0.5, 0.25),
        (STATS_B, -0.03, 2.0, -1.5),
    ],
    ids=["A", "B"],
)
def test_nt_stats_forwarded_identically_by_both_tiers(
    stats: dict[str, Any],
    total_return: float,
    sharpe: float,
    sortino: float,
) -> None:
    """Two unrelated stat sets, literal expected values, through both extractors (R11).

    Screening overwrites ``win_rate`` and ``max_drawdown`` later in its extraction, so
    this comparison covers the three stats both tiers keep; ``extract_stats`` pins the
    validation-only fields and the direct parser test pins the whole parse.
    """
    pnls = {"USDT": stats}
    screening = _screening_stats(pnls)
    assert (screening.total_return, screening.sharpe_ratio, screening.sortino_ratio) == (
        total_return,
        sharpe,
        sortino,
    )

    validation = _validation_stats(pnls)
    assert (validation.total_return, validation.sharpe_ratio, validation.sortino_ratio) == (
        total_return,
        sharpe,
        sortino,
    )


@pytest.mark.parametrize(
    ("stats", "total_return", "max_drawdown", "win_rate", "avg_win", "avg_loss"),
    [
        (STATS_A, 0.125, 0.4, 0.6, 3.0, -2.0),
        (STATS_B, -0.03, 0.1, 0.25, 7.0, -1.5),
    ],
    ids=["A", "B"],
)
def test_validation_extract_stats_pins_the_whole_parse(
    stats: dict[str, Any],
    total_return: float,
    max_drawdown: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> None:
    """Max drawdown is absolute, Avg Winner/Loser are read, over two unrelated inputs."""
    validation = _validation_stats({"USDT": stats})
    assert (
        validation.total_return,
        validation.max_drawdown,
        validation.win_rate,
        validation.avg_win,
        validation.avg_loss,
    ) == (total_return, max_drawdown, win_rate, avg_win, avg_loss)


def test_non_numeric_stat_warns_in_screening_and_debugs_in_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pnls = {"USDT": {"PnL% (total)": "bad"}}
    with caplog.at_level(logging.DEBUG):
        screening = _screening_stats(pnls)
    assert screening.total_return == 0.0
    assert "Could not convert PnL stat PnL% (total)='bad' to float" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        validation = _validation_stats(pnls)
    assert validation.total_return == 0.0
    assert "Non-numeric PnL stat skipped: PnL% (total) = 'bad'" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_validation_alone_reads_avg_winner_and_loser() -> None:
    """Avg Winner / Avg Loser reach the validation result; screening has no such fields."""
    pnls = {"USDT": STATS_A}
    validation = _validation_stats(pnls)
    assert (validation.avg_win, validation.avg_loss) == (3.0, -2.0)
    screening = _screening_stats(pnls)
    assert not hasattr(screening, "avg_win")


def test_stats_returns_fill_only_fields_the_pnl_stats_left_unset() -> None:
    """PnL stats win; a returns-only field fills, never overwriting a legitimate 0.0."""
    pnls = {"USDT": {"Sharpe Ratio (252 days)": 0.0}}
    returns = {"Sharpe Ratio (252 days)": 0.8, "Sortino Ratio (252 days)": 1.2}
    bt = fake_bt_result(stats_pnls=pnls, stats_returns=returns)

    screening = _runner()._extract_metrics({}, bt, fake_engine([]), 0.0, starting_balance=1000.0)
    assert (screening.sharpe_ratio, screening.sortino_ratio) == (0.0, 1.2)

    result = ValidationResult()
    extract_stats(result, bt)
    assert (result.sharpe_ratio, result.sortino_ratio) == (0.0, 1.2)


# Several matching stats_returns keys per field, ordered: the base extractors tested
# only the fields stats_pnls had populated, so the LAST matching returns key won.
_MULTI_RETURNS = {
    "Sharpe Ratio (252 days)": 0.1,
    "Sharpe Ratio (365 days)": 0.2,
    "Sortino Ratio (252 days)": 0.3,
    "Sortino Ratio (365 days)": 0.4,
    "Max Drawdown": -0.5,
    "Max Drawdown (peak)": -0.6,
    "Win Rate": 0.7,
    "win rate": 0.8,
}


def test_multiple_matching_returns_keys_last_one_wins_in_both_tiers() -> None:
    """No stats_pnls: later matching returns keys keep overwriting (base behaviour)."""
    bt = fake_bt_result(stats_pnls={}, stats_returns=_MULTI_RETURNS)

    stats = parse_nt_stats(bt, tier="validation")
    assert (stats.sharpe, stats.sortino, stats.max_drawdown, stats.win_rate) == (
        0.2,
        0.4,
        0.6,
        0.8,
    )
    assert parse_nt_stats(bt, tier="screening") == stats

    screening = _runner()._extract_metrics({}, bt, fake_engine([]), 0.0, starting_balance=1000.0)
    assert (screening.sharpe_ratio, screening.sortino_ratio) == (0.2, 0.4)

    result = ValidationResult()
    extract_stats(result, bt)
    assert (result.sharpe_ratio, result.sortino_ratio) == (0.2, 0.4)
    assert (result.max_drawdown, result.win_rate) == (0.6, 0.8)


def test_pnl_populated_fields_beat_every_returns_key_including_zero() -> None:
    """Control: PnL-populated fields (legitimate 0.0 too) ignore all returns keys."""
    pnls = {
        "USDT": {
            "Sharpe Ratio (252 days)": 0.0,
            "Sortino Ratio (252 days)": 9.0,
            "Max Drawdown": 0.0,
            "Win Rate": 0.0,
        }
    }
    bt = fake_bt_result(stats_pnls=pnls, stats_returns=_MULTI_RETURNS)

    stats = parse_nt_stats(bt, tier="validation")
    assert (stats.sharpe, stats.sortino, stats.max_drawdown, stats.win_rate) == (0.0, 9.0, 0.0, 0.0)
    assert parse_nt_stats(bt, tier="screening") == stats

    screening = _runner()._extract_metrics({}, bt, fake_engine([]), 0.0, starting_balance=1000.0)
    assert (screening.sharpe_ratio, screening.sortino_ratio) == (0.0, 9.0)

    result = ValidationResult()
    extract_stats(result, bt)
    assert (result.sharpe_ratio, result.sortino_ratio) == (0.0, 9.0)
    assert (result.max_drawdown, result.win_rate) == (0.0, 0.0)


def _accrual(payments: list[tuple[int, float]]) -> Any:
    """A ``FundingAccrual`` stub: positive amounts are paid, cash events negate them."""
    return SimpleNamespace(
        payments=payments,
        total=sum(amount for _, amount in payments),
        fallback_settlements=0,
    )


def _costs(
    realized_pnl: float,
    *,
    exit_ns: int = DAY,
    accrual: Any = None,
    pos: Any = None,
    mark: Any = None,
    pos_fees: float = 0.0,
    slippage_cost: float = 0.0,
    entry_slippage: float = 0.0,
    exit_slippage: float = 0.0,
) -> PositionCosts:
    return PositionCosts(
        pos=_closed("P1") if pos is None else pos,
        mark=mark,
        realized_pnl=realized_pnl,
        exit_price=0.0,
        exit_ns=exit_ns,
        exit_reason="signal" if mark is None else "end_of_data",
        pos_fees=pos_fees,
        slippage_cost=slippage_cost,
        accrual=accrual,
        entry_slippage=entry_slippage,
        exit_slippage=exit_slippage,
    )


def test_cash_events_book_funding_before_the_exit() -> None:
    """R15: a day's cash sums funding payments first, then realized - slippage, in BOTH tiers."""
    item = _costs(22.15, accrual=_accrual([(DAY, 0.2365), (DAY, 0.9458)]))
    other = _costs(7.7667, accrual=_accrual([(DAY, 0.9751), (DAY, 1.1824)]))

    for rules in (SCREENING_RULES, VALIDATION_RULES):
        ledger = accumulate([item], rules)
        assert ledger.cash_events == [(DAY, -0.2365), (DAY, -0.9458), (DAY, 22.15)]
        assert daily_balance_returns(1.0, ledger.cash_events, 0, 2 * DAY) == {
            DAY: 20.967699999999997
        }
        assert daily_balance_returns(
            1.0, accumulate([other], rules).cash_events, 0, 2 * DAY
        ) == {DAY: 5.6092}


def _flat(pid: str, entry_ns: int, pnl: float) -> PositionCosts:
    """A zero-fee closed trade with exactly ``pnl`` realized and a 0.1+ per-trade return."""
    pos = FakePosition(
        id=pid,
        instrument_id=BTC,
        ts_opened=entry_ns,
        ts_closed=entry_ns + DAY,
        events=[],
        realized_pnl=pnl,
        avg_px_open=10.0,
        avg_px_close=10.0,
        peak_qty=1.0,
        commission_list=[0.0],
    )
    return _costs(pnl, exit_ns=entry_ns + DAY, pos=pos)


def test_fee_rule_witness() -> None:
    """R5: sum of |commission| per leg vs |sum of commissions| (a rebate leg differs)."""
    rebate = FakePosition.closed(
        "P1",
        BTC,
        entry_ns=0,
        exit_ns=DAY,
        entry_price=10.0,
        exit_price=11.0,
        entry_fee=0.2,
        exit_fee=-0.04,
    )
    netted = FakePosition.closed(
        "P2",
        BTC,
        entry_ns=0,
        exit_ns=DAY,
        entry_price=10.0,
        exit_price=11.0,
        entry_fee=-6.0,
        exit_fee=0.0,
    )
    for pos, rules, expected in (
        (rebate, SCREENING_RULES, 0.24000000000000002),
        (rebate, VALIDATION_RULES, 0.16),
        (netted, SCREENING_RULES, 6.0),
        (netted, VALIDATION_RULES, 6.0),
    ):
        ledger = accumulate(cost_positions([(pos, None)], rules), rules)
        assert ledger.total_fees == expected


def test_open_mark_nets_fee_after_unrealized() -> None:
    """R14: the open mark's net is (realized + unrealized) - exit fee in BOTH tiers."""
    pos = FakePosition.open("P1", BTC, entry_ns=0, entry_price=9146.0, qty=0.01, entry_fee=0.26)
    mark = OpenPositionMark(position=pos, ts=DAY, price=9100.0, unrealized=-0.46, exit_fee=0.0455)
    other = FakePosition.open("P2", BTC, entry_ns=0, entry_price=9146.0, qty=0.01, entry_fee=0.9276)
    other_mark = OpenPositionMark(
        position=other, ts=DAY, price=9000.0, unrealized=-0.4641, exit_fee=0.0366
    )
    for rules in (SCREENING_RULES, VALIDATION_RULES):
        assert accumulate(cost_positions([(pos, mark)], rules), rules).net_pnls == [-0.7655]
        assert accumulate(cost_positions([(other, other_mark)], rules), rules).net_pnls == [
            -1.4283
        ]


def test_summary_uses_entry_time_order() -> None:
    """R16: PF / win-rate input is in entry-time order (stable on ties) in BOTH tiers."""
    costed = [
        _flat("P1", 900 * 60 * 1_000_000_000, 0.1),
        _flat("P2", 800 * 60 * 1_000_000_000, 0.2),
        _flat("P3", 700 * 60 * 1_000_000_000, 0.3),
        _flat("P4", 600 * 60 * 1_000_000_000, -1.0),
    ]
    other = [
        _flat("Q1", 900 * 60 * 1_000_000_000, 0.95),
        _flat("Q2", 800 * 60 * 1_000_000_000, 0.66),
        _flat("Q3", 700 * 60 * 1_000_000_000, 0.24),
        _flat("Q4", 600 * 60 * 1_000_000_000, -0.85),
    ]
    for rules in (SCREENING_RULES, VALIDATION_RULES):
        ledger = accumulate(costed, rules)
        assert ledger.net_pnls == [-1.0, 0.3, 0.2, 0.1]
        assert profit_factor(ledger.net_pnls) == 0.6
        assert trade_win_rate(ledger.net_pnls) == 0.75
        assert profit_factor(accumulate(other, rules).net_pnls) == 2.1764705882352944
        # equal entry times keep the input (cache) order
        ties = [_flat("T1", 5 * DAY, 0.5), _flat("T2", 5 * DAY, -0.25), _flat("T3", 4 * DAY, 0.75)]
        assert accumulate(ties, rules).net_pnls == [0.75, 0.5, -0.25]
        # the closed-trade ledger (daily Sharpe input) stays in cache order
        assert [net for _, net in accumulate(ties, rules).closed_net] == [0.5, -0.25, 0.75]


def test_per_leg_cash_rule_witness() -> None:
    """The liquidation overlay books each slippage leg at its own instant."""
    item = _costs(22.15, slippage_cost=1.0, entry_slippage=0.7, exit_slippage=0.3)

    lump = accumulate([item], VALIDATION_RULES)
    assert lump.cash_events == [(DAY, 21.15)]

    per_leg = accumulate([item], VALIDATION_RULES, per_leg_cash=True)
    assert per_leg.cash_events == [(0, -0.7), (DAY, 21.849999999999998)]

    # one rule in both tiers: the lump nets the slippage (screening's is 0.0 in production)
    assert accumulate([item], SCREENING_RULES).cash_events == [(DAY, 21.15)]


class _RaisingPeakQty:
    """A closed position whose notional read raises: screening must not read it."""

    instrument_id = BTC
    ts_opened = 0
    ts_closed = DAY
    is_closed = True
    is_open = False
    avg_px_open = 10.0
    avg_px_close = 11.0
    realized_pnl = 1.0

    def commissions(self) -> list[float]:
        return [0.0]

    @property
    def peak_qty(self) -> float:
        raise AssertionError("peak_qty read by the screening ledger")


def test_accumulate_screening_never_reads_the_notional() -> None:
    """r3 F11 extended to the ledger: screening's moment_returns rule is off."""
    item = _costs(1.0, pos=_RaisingPeakQty())  # type: ignore[arg-type]
    assert accumulate([item], SCREENING_RULES).moment_returns == []
    assert accumulate([_flat("P1", 0, 1.0)], VALIDATION_RULES).moment_returns == [0.1]


def test_returns_fill_only_unpopulated_fields_with_last_key_winning() -> None:
    """Mixed: PnL sets sharpe only; sortino/drawdown/win rate come from the last returns key."""
    bt = fake_bt_result(
        stats_pnls={"USDT": {"Sharpe Ratio (252 days)": 3.0}}, stats_returns=_MULTI_RETURNS
    )
    stats = parse_nt_stats(bt, tier="validation")
    assert (stats.sharpe, stats.sortino, stats.max_drawdown, stats.win_rate) == (3.0, 0.4, 0.6, 0.8)


def test_screening_liquidated_fee_one_sum_witness() -> None:
    """R17: a liquidated trade's fee is ONE sum (kept commissions + clearance) added to the
    running total: 0.1417 + (0.8374 + 0.04384) = 1.02294, where stepwise adding gives
    1.0229400000000002. The float order is the screening tier's, kept verbatim."""
    closed = FakePosition.closed(
        "C", BTC, entry_ns=0, exit_ns=DAY, entry_price=10.0, exit_price=11.0, entry_fee=0.1417
    )
    liquidated = FakePosition.open(
        "L", BTC, entry_ns=2 * DAY, entry_price=10.0, entry_fee=0.8374
    )
    key = position_key(liquidated)
    overlay = SimpleNamespace(
        dropped_keys=frozenset(),
        hits_by_key={key: SimpleNamespace(ts=3 * DAY, loss=5.0, clearance_fee=0.04384)},
        exit_override={key: (3 * DAY, 9.0)},
    )
    scan_funding = {position_key(closed): FundingAccrual(), key: FundingAccrual()}

    out = screening_liquidation_trades(
        overlay,  # type: ignore[arg-type]
        [(closed, None), (liquidated, None)],
        scan_funding,
        None,  # type: ignore[arg-type]
    )

    assert out.total_fees == 1.02294
    assert out.total_fees != (0.1417 + 0.8374) + 0.04384
    assert out.n_closed == 2


def test_screening_liquidation_keeps_the_fill_at_the_hit_instant() -> None:
    """Equality edge: a fill at ts == hit.ts is kept (<=), one a tick later is not."""
    at_hit = FakePosition.open("H", BTC, entry_ns=5, entry_price=10.0, entry_fee=0.5)
    key = position_key(at_hit)
    hit = SimpleNamespace(ts=5, loss=2.0, clearance_fee=0.25)
    overlay = SimpleNamespace(
        dropped_keys=frozenset(), hits_by_key={key: hit}, exit_override={key: (5, 9.0)}
    )
    out = screening_liquidation_trades(
        overlay,  # type: ignore[arg-type]
        [(at_hit, None)],
        {key: FundingAccrual()},
        None,  # type: ignore[arg-type]
    )
    assert out.total_fees == 0.75  # 0.5 kept commission + 0.25 clearance

    after = FakePosition.open("A", BTC, entry_ns=6, entry_price=10.0, entry_fee=0.5)
    key = position_key(after)
    overlay = SimpleNamespace(
        dropped_keys=frozenset(), hits_by_key={key: hit}, exit_override={key: (5, 9.0)}
    )
    out = screening_liquidation_trades(
        overlay,  # type: ignore[arg-type]
        [(after, None)],
        {key: FundingAccrual()},
        None,  # type: ignore[arg-type]
    )
    assert out.total_fees == 0.25  # the later fill is not kept


def test_validation_liquidated_run_keeps_the_fill_at_the_hit_instant() -> None:
    """Equality edge: validation keeps a fill at ts == hit.ts (commission booked), drops a later one."""
    hit = SimpleNamespace(ts=5, realized_pnl=-3.0, clearance_fee=0.25)
    for entry_ns, kept_fee in ((5, 0.5), (6, 0.0)):
        pos = FakePosition.open("V", BTC, entry_ns=entry_ns, entry_price=10.0, entry_fee=0.5)
        key = position_key(pos)
        overlay = SimpleNamespace(
            dropped_keys=frozenset(),
            hits_by_key={key: hit},
            exit_override={key: (5, 9.0)},
            ledgers={},
        )
        slip = SimpleNamespace(leg=lambda _pos, _price: 0.0)
        adjusted, ledgers = validation_liquidated_run(
            [], [pos], overlay, slip=slip, funding_by_key={}  # type: ignore[arg-type]
        )
        assert ledgers == {}
        (item,) = adjusted
        assert item.exit_reason == "liquidation"
        assert item.fee_legs == (kept_fee, 0.25)
        assert item.pos_fees == kept_fee + 0.25
        assert item.exit_ns == 5
        assert item.exit_price == 9.0
