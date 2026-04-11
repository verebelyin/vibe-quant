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


@pytest.mark.parametrize(
    "yaml_file",
    _example_yaml_files(),
    ids=lambda p: p.stem,
)
def test_example_strategy_compiles_to_golden_source(yaml_file: Path) -> None:
    """Every example strategy must compile byte-identically to its golden file."""
    with yaml_file.open() as f:
        raw = yaml.safe_load(f)
    dsl = validate_strategy_dict(raw)
    compiled = _normalize(StrategyCompiler().compile(dsl))

    golden_path = GOLDEN_DIR / (yaml_file.stem + ".py.golden")
    assert golden_path.exists(), (
        f"Missing golden file for {yaml_file.stem}. If you added a new example "
        f"strategy, regenerate goldens via the helper in the P3 commit."
    )
    expected = _normalize(golden_path.read_text())

    if compiled != expected:
        # Write the diff to a temp file so humans can inspect it without
        # drowning pytest output.
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                compiled.splitlines(),
                fromfile=f"{yaml_file.stem}.golden",
                tofile=f"{yaml_file.stem}.compiled",
                lineterm="",
            )
        )
        pytest.fail(
            f"Compiled source diverged from golden for {yaml_file.stem}:\n{diff[:4000]}"
        )
