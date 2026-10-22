"""Sparse sizing-skip logging in the generated strategy code (vibe-quant-jgr32).

Run 976's log was 632100 lines, 631922 of them the same ``Sizing: notional ...
below minimum ... -- entry skipped`` line: once a genome's equity collapses the
line fires on every signal bar. Sparse per-reason counters keep count 1 byte
identical to today, log counts 2, 4, 8 ... and print one summary on stop.

User ruling Q7=yes: the generated code is shared by screening, discovery,
validation AND paper, so paper gets the same sparse logging.

Visibility only: no order, size, fill or metric may change (R18).
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.objects import Price

from tests.unit.test_audit_dsl_runtime import _BAR, _BTC, _Acct, _sizer
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.templates import ON_RESET_LINES, ON_STOP_LINES

if TYPE_CHECKING:
    from collections.abc import Callable

_CHEAP = types.SimpleNamespace(close=Price.from_str("5000.0"))

_MIN_NOTIONAL_TEXT = "Sizing: notional 5.00 below minimum 10.00000000 USDT -- entry skipped"


class _Instr:
    """BTC instrument whose size increment (not its min quantity) rejects the size."""

    min_quantity = None
    min_notional = _BTC.min_notional

    def get_settlement_currency(self) -> Any:
        return USDT

    def make_qty(self, value: float, round_down: bool = True) -> Any:
        return _BTC.make_qty(value, round_down=round_down)


def _bind(fake: types.SimpleNamespace, lines: tuple[str, ...]) -> types.SimpleNamespace:
    """Exec a template method block onto ``fake`` as bound methods."""
    ns: dict[str, Any] = {}
    exec("\n".join(lines), ns)  # noqa: S102
    for fn in ns.values():
        if isinstance(fn, types.FunctionType):
            setattr(fake, fn.__name__, types.MethodType(fn, fake))
    return fake


def _controllable(release: str = "") -> tuple[types.SimpleNamespace, list[str]]:
    """A fake strategy with on_stop/on_reset bound and cancel/close recorded."""
    fake = _sizer(_Acct({USDT: 1000.0}))
    fake.config.command_release = release
    calls: list[str] = []
    fake.cancel_all_orders = lambda instrument_id: calls.append("cancel")
    fake.close_all_positions = lambda instrument_id: calls.append("close")
    _bind(fake, ON_STOP_LINES)
    _bind(fake, ON_RESET_LINES)
    return fake, calls


def _size_increment_sizer() -> types.SimpleNamespace:
    fake = _sizer(_Acct({USDT: 3.0}))
    fake.instrument = _Instr()
    return fake


_CASES: list[tuple[str, Callable[[], types.SimpleNamespace], Any, bool, str]] = [
    (
        "no_account",
        lambda: _sizer(None),
        _BAR,
        True,
        "Sizing: no account for venue yet -- entry skipped",
    ),
    (
        "no_equity",
        lambda: _sizer(_Acct({USDT: 0.0})),
        _BAR,
        True,
        "Sizing: no positive USDT equity (0.0) -- entry skipped",
    ),
    (
        "invalid_price",
        lambda: _sizer(_Acct({USDT: 1000.0})),
        types.SimpleNamespace(close=Price.from_str("0.0")),
        True,
        "Sizing: invalid price 0.0 -- entry skipped",
    ),
    (
        "non_positive_size",
        lambda: _sizer(_Acct({USDT: 1000.0}), sl_pct=float("inf")),
        _BAR,
        False,
        "Sizing: non-positive size 0.0 -- entry skipped",
    ),
    (
        "min_qty_cap",
        lambda: _sizer(_Acct({USDT: 50.0})),
        _BAR,
        True,
        "Sizing: min quantity 0.001 exceeds max_position_pct cap 0.00041667 "
        "-- entry skipped",
    ),
    (
        "size_increment",
        _size_increment_sizer,
        _BAR,
        True,
        "Sizing: 2.5e-05 rounds below the size increment -- entry skipped",
    ),
    (
        "min_notional",
        lambda: _sizer(_Acct({USDT: 15.0})),
        _CHEAP,
        True,
        _MIN_NOTIONAL_TEXT,
    ),
]


def test_skip_warnings_log_at_powers_of_two_and_summarise_on_stop() -> None:
    """R16 (a): 6000 skips -> 13 WARNINGs (1,2,...,4096) + one on_stop summary."""
    s = _sizer(_Acct({USDT: 15.0}))
    for _ in range(6000):
        assert s._calculate_position_size(_CHEAP, True) is None

    assert len(s.log.warnings) == 13
    assert s.log.warnings[0] == _MIN_NOTIONAL_TEXT
    for count, msg in zip((2**k for k in range(1, 13)), s.log.warnings[1:], strict=True):
        assert msg == f"{_MIN_NOTIONAL_TEXT} [{count} skips: min_notional]"

    s.config.command_release = "bar"
    _bind(s, ON_STOP_LINES)
    s.on_stop()
    assert s.log.warnings[-1] == "Sizing: 6000 entries skipped (min_notional=6000)"
    assert len(s.log.warnings) == 14


def test_first_skip_text_is_unchanged_for_every_reason() -> None:
    """R16b: count 1 logs TODAY's exact text, byte for byte, for all 7 reasons."""
    for reason, build, bar, is_long, expected in _CASES:
        s = build()
        assert s._calculate_position_size(bar, is_long) is None, reason
        assert s.log.warnings == [expected], reason
        assert s._sizing_skips == {reason: 1}, reason


def test_mixed_reasons_count_per_reason_on_one_strategy() -> None:
    """Counters are per reason: a second reason's first skip logs today's text."""
    accounts: list[_Acct | None] = [None, None, _Acct({USDT: 0.0})]
    s = _sizer(None)
    s.cache.account_for_venue = lambda venue: accounts.pop(0)

    for _ in range(3):
        assert s._calculate_position_size(_BAR, True) is None

    no_account, no_equity = _CASES[0][4], _CASES[1][4]
    assert s.log.warnings == [no_account, f"{no_account} [2 skips: no_account]", no_equity]
    assert s._sizing_skips == {"no_account": 2, "no_equity": 1}


def test_on_stop_summary_in_both_modes() -> None:
    """R17: one summary WARNING, sorted reasons, before the outbox early return."""
    for release, expected_calls in (("bar", []), ("", ["cancel", "close"])):
        fake, calls = _controllable(release)
        fake._sizing_skips = {"no_equity": 2, "min_notional": 5}
        fake.on_stop()
        assert fake.log.warnings == [
            "Sizing: 7 entries skipped (min_notional=5, no_equity=2)"
        ]
        assert calls == expected_calls, release


def test_on_reset_zeroes_counts() -> None:
    """R17 (c): the next run starts at skip #1 and logs today's text again."""
    s = _sizer(_Acct({USDT: 15.0}))
    _bind(s, ON_RESET_LINES)
    s._calculate_position_size(_CHEAP, True)
    s._calculate_position_size(_CHEAP, True)
    assert s.log.warnings[1] == f"{_MIN_NOTIONAL_TEXT} [2 skips: min_notional]"

    s.on_reset()
    assert s._sizing_skips == {}
    s.log.warnings.clear()
    s._calculate_position_size(_CHEAP, True)
    assert s.log.warnings == [_MIN_NOTIONAL_TEXT]


def test_no_skips_no_summary() -> None:
    """R17 (d): nothing skipped -> no summary line at all."""
    fake, _ = _controllable("bar")
    fake._sizing_skips = {}
    fake.on_stop()
    assert fake.log.warnings == []


def test_two_to_the_twentieth_skips_log_21_lines() -> None:
    """R16b: the 21st line is skip count 2**20 (one call per power, no million loop)."""
    s = _sizer(_Acct({USDT: 15.0}))
    lines: list[str] = []
    for k in range(21):
        s._sizing_skips = {"min_notional": 2**k - 1}
        s.log.warnings.clear()
        s._calculate_position_size(_CHEAP, True)
        assert len(s.log.warnings) == 1, k
        lines.extend(s.log.warnings)
    assert len(lines) == 21
    assert lines[0] == _MIN_NOTIONAL_TEXT
    assert lines[-1] == f"{_MIN_NOTIONAL_TEXT} [1048576 skips: min_notional]"


def test_compiled_strategy_initialises_skip_counters() -> None:
    """(e): a real compiled strategy starts with empty per-reason counters."""
    dsl = validate_strategy_dict(
        {
            "name": "skip_probe_1m",
            "timeframe": "1m",
            "indicators": {},
            "entry_conditions": {"long": ["close > 1"]},
            "exit_conditions": {},
            "stop_loss": {"type": "fixed_pct", "percent": 50.0},
            "take_profit": {"type": "fixed_pct", "percent": 50.0},
        }
    )
    assert "self._sizing_skips: dict[str, int] = {}" in StrategyCompiler().compile(dsl)

    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    config = getattr(module, f"{camel}Config")(instrument_id="BTCUSDT-PERP.BINANCE")
    strategy = getattr(module, f"{camel}Strategy")(config=config)
    assert strategy._sizing_skips == {}
