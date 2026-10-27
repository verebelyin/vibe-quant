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
