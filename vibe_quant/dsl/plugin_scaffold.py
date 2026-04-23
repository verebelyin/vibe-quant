"""Scaffolding CLI for new indicator plugins.

Generates two files from templates: the plugin module under
``vibe_quant/dsl/plugins/<name>.py`` and a matching golden-contract test
stub under ``tests/unit/test_plugins/test_<name>.py``. The test stub
uses :func:`vibe_quant.dsl.invoke_compute_fn` so plugin authors inherit
the output-contract validation for free.

Invoked via ``vibe-quant plugin new <NAME>``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

VALID_CATEGORIES = ("Trend", "Momentum", "Volatility", "Volume", "Custom")

_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _normalize_name(raw: str) -> tuple[str, str]:
    """Return ``(UPPER, lower)`` identifier variants.

    Raises ``ValueError`` if ``raw`` is not a valid Python identifier
    after upper-casing — an indicator name must survive being used both
    as a YAML ``type:`` value and as a module filename.
    """
    upper = raw.upper()
    if not _NAME_RE.match(upper):
        msg = (
            f"Invalid indicator name {raw!r}: must start with a letter and "
            "contain only A-Z, 0-9, or underscores."
        )
        raise ValueError(msg)
    return upper, upper.lower()


PLUGIN_TEMPLATE = '''"""{name_upper} — custom indicator plugin.

Drop-in registered via ``indicator_registry.register_spec`` at import
time; the loader auto-imports this file from
``vibe_quant/dsl/plugins/``. See ``plugins/README.md`` for the full
``IndicatorSpec`` field reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def compute_{name_lower}(
    df: "pd.DataFrame", params: dict[str, object]
) -> "pd.Series":
    """Compute {name_upper} from an OHLCV DataFrame.

    Perf note: this runs once per bar in the hot path. Prefer numpy /
    pandas vectorized ops (``rolling``, ``np.convolve``, ``ewm``) over
    Python-level for-loops — a 1m backtest over 3 months is ~130k bars.
    """
    import pandas as pd

    period = int_param(params, "period", 14)
    close = df["close"]
    return pd.Series(close.rolling(period).mean(), index=df.index)


indicator_registry.register_spec(
    IndicatorSpec(
        name="{name_upper}",
        nt_class=None,
        pandas_ta_func=None,
        default_params={{"period": 14}},
        param_schema={{"period": int}},
        compute_fn=compute_{name_lower},
        display_name="{display_name}",
        description="{description}",
        category="{category}",
        param_ranges={{"period": (5.0, 50.0)}},
        threshold_range=(20.0, 80.0),
    )
)
'''

TEST_TEMPLATE = '''"""Contract + smoke tests for the {name_upper} plugin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl.indicators import indicator_registry, invoke_compute_fn


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {{
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1000.0, 10000.0, n),
        }},
        index=idx,
    )


def test_{name_lower}_registered() -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None, "{name_upper} plugin did not register"
    assert spec.compute_fn is not None


def test_{name_lower}_contract(sample_ohlcv: pd.DataFrame) -> None:
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None
    out = invoke_compute_fn(spec, sample_ohlcv, spec.default_params)
    assert len(out) == len(sample_ohlcv)


def test_{name_lower}_short_input() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    df = pd.DataFrame(
        {{
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }},
        index=idx,
    )
    spec = indicator_registry.get("{name_upper}")
    assert spec is not None
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert len(out) == len(df)
'''


def _render_plugin(
    name_upper: str,
    name_lower: str,
    category: str,
    display_name: str,
    description: str,
) -> str:
    return PLUGIN_TEMPLATE.format(
        name_upper=name_upper,
        name_lower=name_lower,
        category=category,
        display_name=display_name,
        description=description,
    )


def _render_test(name_upper: str, name_lower: str) -> str:
    return TEST_TEMPLATE.format(name_upper=name_upper, name_lower=name_lower)


def _write_file(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        msg = f"{path} already exists (pass --force to overwrite)"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def scaffold_plugin(
    name: str,
    *,
    category: str = "Custom",
    plugin_dir: Path | None = None,
    tests_dir: Path | None = None,
    skip_tests: bool = False,
    force: bool = False,
    display_name: str | None = None,
    description: str | None = None,
) -> list[Path]:
    """Render plugin + test stub from templates. Returns paths written."""
    if category not in VALID_CATEGORIES:
        msg = (
            f"Invalid category {category!r}. Choose from: "
            f"{', '.join(VALID_CATEGORIES)}"
        )
        raise ValueError(msg)

    name_upper, name_lower = _normalize_name(name)
    plugin_dir = plugin_dir or Path("vibe_quant/dsl/plugins")
    tests_dir = tests_dir or Path("tests/unit/test_plugins")

    plugin_path = plugin_dir / f"{name_lower}.py"
    plugin_content = _render_plugin(
        name_upper,
        name_lower,
        category,
        display_name or name_upper.replace("_", " ").title(),
        description or f"{name_upper} — custom indicator.",
    )
    _write_file(plugin_path, plugin_content, force=force)

    written = [plugin_path]

    if not skip_tests:
        test_path = tests_dir / f"test_{name_lower}.py"
        test_content = _render_test(name_upper, name_lower)
        _write_file(test_path, test_content, force=force)
        written.append(test_path)

    return written


def _run_pytest(test_path: Path) -> int:
    """Run pytest on the generated test stub and return the exit code."""
    return subprocess.call(
        [sys.executable, "-m", "pytest", str(test_path), "-q"]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vibe-quant plugin",
        description="Scaffolding helpers for indicator plugins.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    new = sub.add_parser("new", help="Scaffold a new indicator plugin")
    new.add_argument("name", help="Indicator name (uppercase, e.g. MY_IND)")
    new.add_argument(
        "--category",
        choices=VALID_CATEGORIES,
        default="Custom",
        help="Catalog category (default: Custom)",
    )
    new.add_argument(
        "--plugin-dir",
        type=Path,
        default=None,
        help="Plugin output directory (default: vibe_quant/dsl/plugins)",
    )
    new.add_argument(
        "--tests-dir",
        type=Path,
        default=None,
        help="Test output directory (default: tests/unit/test_plugins)",
    )
    new.add_argument(
        "--display-name",
        default=None,
        help="Human-readable display name (default: title-cased NAME)",
    )
    new.add_argument(
        "--description",
        default=None,
        help="One-line description for UI tooltips",
    )
    new.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip the test stub",
    )
    new.add_argument(
        "--run-tests",
        action="store_true",
        help="After scaffolding, invoke pytest on the generated test",
    )
    new.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action != "new":
        parser.print_help()
        return 0

    try:
        written = scaffold_plugin(
            args.name,
            category=args.category,
            plugin_dir=args.plugin_dir,
            tests_dir=args.tests_dir,
            skip_tests=args.no_tests,
            force=args.force,
            display_name=args.display_name,
            description=args.description,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Generated:")
    for path in written:
        print(f"  {path}")

    test_path = next((p for p in written if "test_" in p.name), None)
    if args.run_tests and test_path is not None:
        print()
        print(f"Running pytest on {test_path}…")
        return _run_pytest(test_path)

    print()
    print(
        "Next: edit the plugin's compute_fn and description, then "
        "`pytest` the test stub."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
