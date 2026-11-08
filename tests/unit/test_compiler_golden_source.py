"""Byte-identical compiled-source regression guard for the compiler.

Phases 3 and 4 of the indicator-plugin-system refactor carry an explicit
safety contract: adding the callback fields to ``IndicatorSpec`` (P3) and
wiring the compiler through those callbacks (P4) MUST NOT change the
compiled output for any existing strategy. If you're editing the compiler
and this test fails, investigate before regenerating the goldens.

Goldens live under ``tests/fixtures/compiled_source/*.py.golden``. The
embedded ``Generated: <ts>`` line is normalized away so the comparison is
stable across runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "vibe_quant" / "strategies" / "examples"
GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "compiled_source"

_TS_PLACEHOLDER = "Generated: <timestamp-stripped>"
_TS_PATTERN = re.compile(r"Generated: [0-9T:.\-]+")


def _normalize(src: str) -> str:
    """Strip the embedded generation timestamp so goldens are stable."""
    return _TS_PATTERN.sub(_TS_PLACEHOLDER, src)


def _example_yaml_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.yaml"))


def _assert_matches_golden(stem: str, compiled: str) -> None:
    """Compare normalized ``compiled`` source against ``<stem>.py.golden``."""
    golden_path = GOLDEN_DIR / (stem + ".py.golden")
    assert golden_path.exists(), (
        f"Missing golden file for {stem}. If you added a new example "
        f"strategy, regenerate goldens via the helper in the P3 commit."
    )
    expected = _normalize(golden_path.read_text())

    if compiled != expected:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                compiled.splitlines(),
                fromfile=f"{stem}.golden",
                tofile=f"{stem}.compiled",
                lineterm="",
            )
        )
        pytest.fail(f"Compiled source diverged from golden for {stem}:\n{diff[:4000]}")


@pytest.mark.parametrize(
    "yaml_file",
    _example_yaml_files(),
    ids=lambda p: p.stem,
)
def test_no_example_compiles_the_retired_indicator_tail_memo(yaml_file: Path) -> None:
    """A compute_fn indicator is called straight; the per-bar tail memo is retired.

    The precomputed columns (PTA precompute) already serve the NT path, and a
    measured NT discovery logs zero memo calls, so generated
    strategies must not import or call the memo at all.
    """
    with yaml_file.open() as f:
        raw = yaml.safe_load(f)
    compiled = StrategyCompiler().compile(validate_strategy_dict(raw))

    assert "indicator_memo" not in compiled
    assert "memo_lookup" not in compiled
    assert "memo_store" not in compiled


@pytest.mark.parametrize(
    "yaml_file",
    _example_yaml_files(),
    ids=lambda p: p.stem,
)
def test_example_strategy_compiles_to_golden_source(yaml_file: Path) -> None:
    """Every example strategy must compile byte-identically to its golden file."""
    with yaml_file.open() as f:
        raw = yaml.safe_load(f)
    compiled = _normalize(StrategyCompiler().compile(validate_strategy_dict(raw)))
    _assert_matches_golden(yaml_file.stem, compiled)


# Two differently named as-of context indicators: each must emit its OWN scalar
# call with its OWN params (no shared/vector path), which is what the golden pins.
TAKER_SCALAR_DSL: dict[str, object] = {
    "name": "taker_scalar",
    "timeframe": "1m",
    "indicators": {
        "ti_fast": {"type": "TAKER_IMBALANCE", "period": 5},
        "ti_slow": {"type": "TAKER_IMBALANCE", "period": 30},
    },
    "entry_conditions": {"long": ["ti_fast > 0.05"]},
    "exit_conditions": {"long": ["ti_slow < -0.05"]},
    "stop_loss": {"type": "fixed_pct", "percent": 1.0},
    "take_profit": {"type": "fixed_pct", "percent": 2.0},
}


def test_taker_scalar_strategy_compiles_to_golden_source() -> None:
    """The as-of scalar codegen branch has a representative golden of its own.

    No example strategy uses TAKER_IMBALANCE, so without this case the
    ``asof_context`` branch of ``_update_pta_indicators`` (the ``asof_scalar``
    call instead of a per-bar DataFrame) could change unnoticed.
    """
    compiled = _normalize(StrategyCompiler().compile(validate_strategy_dict(TAKER_SCALAR_DSL)))

    for name in ("ti_fast", "ti_slow"):
        assert (
            f'self._pta_values["{name}"] = taker_imbalance_scalar('
            f'self.config.instrument_id, _buf.last_close_ns(), self._pta_params["{name}"])'
        ) in compiled
    assert "compute_taker_imbalance(_df" not in compiled  # never the vector path

    _assert_matches_golden("taker_scalar", compiled)


# Strategy A's shape (4h long, RSI dip filtered by the long SMA): the time exit is
# its only exit, so the golden pins the whole ``max_bars_in_trade`` codegen surface
# (the __init__ counter and the on_bar block, N as an int literal).
MAX_BARS_DSL: dict[str, object] = {
    "name": "max_bars_exit",
    "timeframe": "4h",
    "indicators": {
        "rsi": {"type": "RSI", "period": 2},
        "sma": {"type": "SMA", "period": 200},
    },
    "entry_conditions": {"long": ["rsi crosses_below 30", "close > sma"]},
    "stop_loss": {"type": "fixed_pct", "percent": 50.0},
    "take_profit": {"type": "fixed_pct", "percent": 100.0},
    "max_bars_in_trade": 12,
}


def test_max_bars_strategy_compiles_to_golden_source() -> None:
    compiled = _normalize(StrategyCompiler().compile(validate_strategy_dict(MAX_BARS_DSL)))

    assert "        self._bars_in_trade = 0" in compiled
    assert "        if self._bars_in_trade >= 12:" in compiled
    assert "self.config.max_bars_in_trade" not in compiled  # a LITERAL, not a config field

    _assert_matches_golden("max_bars_exit", compiled)


@pytest.mark.parametrize("n", [1, 12])
def test_max_bars_count_is_emitted_as_an_int_literal(n: int) -> None:
    """N is typed by the compiler, so a sweep/override key can never reach it."""
    dsl = dict(MAX_BARS_DSL, max_bars_in_trade=n)
    compiled = _normalize(StrategyCompiler().compile(validate_strategy_dict(dsl)))

    assert f"        if self._bars_in_trade >= {n}:" in compiled
    assert "self.config.max_bars_in_trade" not in compiled


def test_no_time_exit_block_without_the_field() -> None:
    """Field absent -> nothing emitted anywhere (the example goldens gate the rest)."""
    compiled = _normalize(StrategyCompiler().compile(validate_strategy_dict(TAKER_SCALAR_DSL)))

    assert "_bars_in_trade" not in compiled


# position_mode "reverse": an EMA cross that is always in the market. The golden pins the whole
# reversal codegen surface (the intent record, the __init__ state, the expiry call, the reverse
# branch of the entry block, the exit-block guard, the PositionClosed hook and the reverse-aware
# order methods); a strategy without the key compiles to the other goldens unchanged.
REVERSE_EMA_CROSS_DSL: dict[str, object] = {
    "name": "reverse_ema_cross",
    "timeframe": "1h",
    "indicators": {
        "fast": {"type": "EMA", "period": 9},
        "slow": {"type": "EMA", "period": 21},
    },
    "entry_conditions": {"long": ["fast crosses_above slow"], "short": ["fast crosses_below slow"]},
    "exit_conditions": {"long": ["close < slow"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
    "position_mode": "reverse",
}


def test_reverse_strategy_compiles_to_golden_source() -> None:
    compiled = _normalize(StrategyCompiler().compile(validate_strategy_dict(REVERSE_EMA_CROSS_DSL)))

    assert "class ReversalIntent(NamedTuple):" in compiled
    assert "if not self._reverse_on_closed(event):" in compiled
    assert "elif self._position_open:" in compiled
    assert "self._begin_reverse(bar, OrderSide.BUY)" in compiled

    _assert_matches_golden("reverse_ema_cross", compiled)


def test_reverse_hooks_are_absent_when_the_key_is_absent_or_null() -> None:
    without = {k: v for k, v in REVERSE_EMA_CROSS_DSL.items() if k != "position_mode"}
    absent = _normalize(StrategyCompiler().compile(validate_strategy_dict(without)))
    null = _normalize(StrategyCompiler().compile(validate_strategy_dict({**without, "position_mode": None})))

    assert absent == null
    assert "ReversalIntent" not in absent and "_begin_reverse" not in absent
