# Indicator Plugins

Drop a `.py` file in this directory to add a custom indicator. The file is auto-imported at startup via `vibe_quant.dsl.plugin_loader`. No compiler, schema, genome, or frontend edits required.

## Minimal template

```python
"""My custom indicator plugin."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd

def compute_my_indicator(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    import pandas as pd
    period = int(params.get("period", 14))
    close = df["close"]
    return pd.Series(close.rolling(period).mean(), index=df.index)

indicator_registry.register_spec(
    IndicatorSpec(
        name="MY_INDICATOR",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 14},
        param_schema={"period": int},
        compute_fn=compute_my_indicator,
        display_name="My Custom Indicator",
        description="One-line description for UI tooltips.",
        category="Custom",
        param_ranges={"period": (5.0, 50.0)},
        threshold_range=(20.0, 80.0),
    )
)
```

## How it works

1. At startup, `load_builtin_plugins()` walks this directory with `pkgutil.iter_modules`.
2. Files prefixed with `_` are skipped (useful for helpers).
3. Each module is imported; a failing import is logged and swallowed (won't crash the app).
4. The module calls `indicator_registry.register_spec(spec)` at module scope.
5. The new spec is immediately available to the DSL parser, compiler, GA pool, and API catalog.

## IndicatorSpec field reference

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | yes | Canonical uppercase name (e.g. `"MY_IND"`) |
| `nt_class` | `type \| None` | yes | NT indicator class, or `None` for compute_fn-only |
| `pandas_ta_func` | `str \| None` | yes | Legacy pandas-ta function name (set `None` for plugins) |
| `default_params` | `dict[str, object]` | yes | Default parameter values |
| `param_schema` | `dict[str, type]` | yes | Parameter name → Python type for validation |
| `output_names` | `tuple[str, ...]` | no | Output names; default `("value",)`. Multi-output: `("k", "d")` |
| `compute_fn` | `Callable[[DataFrame, dict], Series \| dict]` | no* | Pure Python computation. *Required if `nt_class` is None |
| `nt_kwargs_fn` | `Callable[[dict], dict] \| None` | no | Maps merged params → NT constructor kwargs |
| `nt_codegen_kwargs` | `tuple[tuple[str, str], ...]` | no | `(nt_kwarg, dsl_field)` pairs for code generation |
| `nt_output_attrs` | `dict[str, str]` | no | Maps output_name → NT attribute name; default `{"value": "value"}` |
| `computed_outputs` | `dict[str, str]` | no | Output names derived at runtime → helper function name in `derived.py` |
| `pta_lookback_fn` | `Callable[[dict], int] \| None` | no | Min bars before compute_fn is valid; default = `params["period"]` |
| `primary_output` | `str` | no | Output for bare references (no sub-value); default = first in `output_names` |
| `requires_high_low` | `bool` | no | Indicator needs high/low columns |
| `requires_volume` | `bool` | no | Indicator needs volume column |
| `display_name` | `str` | no | Human-readable name for UI |
| `description` | `str` | no | One-line description for tooltips |
| `category` | `str` | no | `"Trend"`, `"Momentum"`, `"Volatility"`, `"Volume"`, or `"Custom"` |
| `popular` | `bool` | no | Highlight in the UI indicator picker |
| `chart_placement` | `str` | no | `"overlay"` (on price) or `"oscillator"` (own pane); default `"oscillator"` |
| `param_ranges` | `dict[str, tuple[float, float]]` | no | GA mutation bounds; empty = excluded from GA |
| `threshold_range` | `tuple[float, float] \| None` | no | GA threshold range; `None` = excluded from GA |

## compute_fn contract

```python
def compute_fn(df: pd.DataFrame, params: dict[str, object]) -> pd.Series | dict[str, pd.Series]:
```

**Input**: `df` has columns `["open", "high", "low", "close", "volume"]` as float64. `params` is the merged default + DSL-override dict.

**Output**:
- Single-output indicators: return a `pd.Series` with the same index as `df`.
- Multi-output indicators: return a `dict[str, pd.Series]` keyed by output name (must match `output_names`).

NaN values are expected during the warmup period. The compiler handles NaN filtering.

## GA auto-enrollment

Set both `param_ranges` and `threshold_range` to auto-enroll in the genetic discovery pool. Leave `threshold_range=None` to exclude price-relative indicators (where threshold comparisons don't make sense).

## NT Rust subclass path

For maximum speed, a plugin can subclass `nautilus_trader.indicators.base.Indicator` in Python and pass the class as `nt_class`. The compiler will generate `register_indicator_for_bars()` calls and read values from the NT attribute namespace. See the [NautilusTrader indicators API](https://docs.nautilustrader.io/api_reference/indicators.html) for the base class contract.

## See also

- `example_adaptive_rsi.py` — fully worked example with Kaufman adaptive smoothing
- `vibe_quant/dsl/indicators.py` — built-in spec registrations (20 indicators)
- `vibe_quant/dsl/compute_builtins.py` — compute_fn implementations for built-ins
- `vibe_quant/dsl/derived.py` — derived-output helpers (percent_b, bandwidth, position)
