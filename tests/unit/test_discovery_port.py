"""Tests for the TradingView port loader (port.yaml -> Strategy DSL dict)."""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

import pytest
import yaml

from vibe_quant.discovery.port import (
    Port,
    PortError,
    PortParam,
    load_port,
)

if TYPE_CHECKING:
    from pathlib import Path

_BASE_DSL: dict[str, object] = {
    "name": "ema_cross",
    "timeframe": "1h",
    "indicators": {
        "ema_fast": {"type": "{{ma}}", "period": "{{fast}}"},
        "ema_slow": {"type": "SMA", "period": "{{slow}}"},
        "atr": {"type": "ATR", "period": 14},
    },
    "entry_conditions": {
        "long": ["ema_fast crosses_above ema_slow"],
        "short": ["ema_fast crosses_below ema_slow"],
    },
    "exit_conditions": {
        "long": ["ema_fast crosses_below ema_slow"],
        "short": ["ema_fast crosses_above ema_slow"],
    },
    "stop_loss": {"type": "atr_fixed", "atr_multiplier": "{{mult}}", "indicator": "atr"},
    "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
}


def _base_doc() -> dict[str, object]:
    return {
        "name": "09_ema_cross",
        "pine": "09_single_ema_cross.pine",
        "tv": {"initial_capital": 10000},
        "params": {
            "fast": {"kind": "int", "low": 5, "high": 50, "step": 1, "default": 12},
            "slow": {"kind": "int", "low": 10, "high": 200, "step": 1, "default": 26},
            "mult": {"kind": "float", "low": 1.0, "high": 5.0, "step": 0.5, "default": 3.0},
            "ma": {"kind": "choice", "choices": ["EMA", "SMA"], "default": "EMA"},
        },
        "constraints": ["fast < slow"],
        "dsl": yaml.safe_load(yaml.safe_dump(_BASE_DSL)),
    }


def _write(tmp_path: Path, doc: object, *, name: str = "port.yaml") -> Path:
    folder = tmp_path / "09_ema_cross"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _load(tmp_path: Path, doc: object | None = None) -> Port:
    return load_port(_write(tmp_path, _base_doc() if doc is None else doc))


def _set_param(doc: dict[str, object], name: str, **changes: object) -> None:
    params = doc["params"]
    assert isinstance(params, dict)
    spec = params[name]
    assert isinstance(spec, dict)
    spec.update(changes)


def test_render_draws_no_rng(tmp_path: Path) -> None:
    """load/render/defaults/free_params never draw from the global RNG."""
    port = _load(tmp_path)
    before = random.getstate()
    values = port.defaults()
    port.render(values)
    port.check_values(values)
    port.satisfies_constraints(values)
    port.free_params()
    port.defaults()
    load_port(port.path)
    assert random.getstate() == before


def test_load_accepts_folder_and_file(tmp_path: Path) -> None:
    path = _write(tmp_path, _base_doc())
    folder = path.parent
    assert load_port(folder).name == "09_ema_cross"
    assert load_port(path).name == "09_ema_cross"
    from_file = load_port(path)
    assert from_file.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert from_file.tv == {"initial_capital": 10000}
    assert from_file.path == path


def test_defaults_and_free_params(tmp_path: Path) -> None:
    port = _load(tmp_path)
    assert port.defaults() == {"fast": 12, "slow": 26, "mult": 3.0, "ma": "EMA"}
    assert {p.name for p in port.free_params()} == {"fast", "slow", "mult", "ma"}
    assert all(isinstance(p, PortParam) for p in port.params)


def test_render_exact_placeholder_keeps_scalar_type(tmp_path: Path) -> None:
    port = _load(tmp_path)
    dsl = port.render(port.defaults())
    indicators = dsl["indicators"]
    assert isinstance(indicators, dict)
    ema_fast = indicators["ema_fast"]
    atr_multiplier = dsl["stop_loss"]
    assert isinstance(ema_fast, dict)
    assert isinstance(atr_multiplier, dict)
    assert ema_fast["period"] == 12
    assert type(ema_fast["period"]) is int
    assert ema_fast["type"] == "EMA"
    assert atr_multiplier["atr_multiplier"] == 3.0
    assert type(atr_multiplier["atr_multiplier"]) is float


def test_render_placeholder_inside_string_uses_str(tmp_path: Path) -> None:
    doc = _base_doc()
    dsl = doc["dsl"]
    assert isinstance(dsl, dict)
    dsl["description"] = "fast={{fast}} mult={{mult}}"
    port = _load(tmp_path, doc)
    rendered = port.render(port.defaults())
    assert rendered["description"] == "fast=12 mult=3.0"


def test_render_float_snaps_to_grid(tmp_path: Path) -> None:
    port = _load(tmp_path)
    values = port.defaults()
    values["mult"] = 3.0000000001
    dsl = port.render(values)
    stop_loss = dsl["stop_loss"]
    assert isinstance(stop_loss, dict)
    assert stop_loss["atr_multiplier"] == 3.0


def test_float_off_grid_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "mult", default=3.1)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


# --- domain table: numeric field types -------------------------------------


@pytest.mark.parametrize("field", ["low", "high", "step", "default"])
def test_bool_numeric_field_rejected(tmp_path: Path, field: str) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", **{field: True})
    with pytest.raises(PortError):
        _load(tmp_path, doc)


@pytest.mark.parametrize("field", ["low", "high", "step", "default", "kind"])
def test_null_numeric_field_rejected(tmp_path: Path, field: str) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", **{field: None})
    with pytest.raises(PortError):
        _load(tmp_path, doc)


@pytest.mark.parametrize("field", ["low", "high", "step", "default"])
def test_numeric_string_rejected(tmp_path: Path, field: str) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", **{field: "14"})
    with pytest.raises(PortError):
        _load(tmp_path, doc)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(tmp_path: Path, bad: float) -> None:
    doc = _base_doc()
    _set_param(doc, "mult", low=bad)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_huge_int_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", high=2**53 + 1)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_float_for_int_param_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", low=5.0)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_low_greater_than_high_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", low=50, high=5, default=12)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_low_equals_high_is_fixed(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", low=12, high=12, default=12)
    port = _load(tmp_path, doc)
    assert "fast" not in {p.name for p in port.free_params()}
    assert port.defaults()["fast"] == 12
    assert port.render(port.defaults())["timeframe"] == "1h"


@pytest.mark.parametrize("step", [0, -1, 0.0, -0.5])
def test_step_not_positive_rejected(tmp_path: Path, step: float) -> None:
    doc = _base_doc()
    _set_param(doc, "mult", step=step)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_default_off_grid_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", step=2, default=12)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_default_out_of_bounds_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", default=500)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


# --- domain table: choices -------------------------------------------------


@pytest.mark.parametrize(
    "choices",
    [
        [],
        ["EMA", "EMA"],
        ["EMA", 3],
        ["EMA", True],
        ["EMA", None],
        ["EMA", 1.0],
    ],
)
def test_bad_choices_rejected(tmp_path: Path, choices: list[object]) -> None:
    doc = _base_doc()
    _set_param(doc, "ma", choices=choices)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_choice_default_must_be_in_choices_type_exact(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "ma", default="ema")
    with pytest.raises(PortError):
        _load(tmp_path, doc)


@pytest.mark.parametrize("kind", ["real", "Int", True, 1])
def test_unknown_kind_rejected(tmp_path: Path, kind: object) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", kind=kind)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_unknown_spec_key_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", bogus=1)
    with pytest.raises(PortError):
        _load(tmp_path, doc)


# --- domain table: constraints ---------------------------------------------


@pytest.mark.parametrize(
    "constraint",
    [
        "fast + slow",
        "fast = slow",
        "fast < missing",
        "ma < slow",
        "fast <",
        "fast <> slow",
    ],
)
def test_bad_constraint_rejected(tmp_path: Path, constraint: str) -> None:
    doc = _base_doc()
    doc["constraints"] = [constraint]
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_unsatisfiable_constraints_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", low=5, high=10, default=10)
    _set_param(doc, "slow", low=20, high=200, default=26)
    doc["constraints"] = ["fast > slow"]
    with pytest.raises(PortError, match="constraints unsatisfiable"):
        _load(tmp_path, doc)


def test_defaults_violating_constraint_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    doc["constraints"] = ["fast > slow"]
    with pytest.raises(PortError) as excinfo:
        _load(tmp_path, doc)
    assert "unsatisfiable" not in str(excinfo.value)


def test_constraint_lt_rejects_equal_values(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", low=12, high=12, default=12)
    _set_param(doc, "slow", low=12, high=12, default=12)
    doc["constraints"] = ["fast < slow"]
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_constraint_le_allows_equal_values(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", low=12, high=12, default=12)
    _set_param(doc, "slow", low=12, high=12, default=12)
    doc["constraints"] = ["fast <= slow"]
    port = _load(tmp_path, doc)
    assert port.render(port.defaults())["timeframe"] == "1h"


def test_zero_free_params_loads_and_renders(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "fast", low=12, high=12, default=12)
    _set_param(doc, "slow", low=26, high=26, default=26)
    _set_param(doc, "mult", low=3.0, high=3.0, default=3.0)
    _set_param(doc, "ma", choices=["EMA"])
    port = _load(tmp_path, doc)
    assert port.free_params() == ()
    rendered = port.render(port.defaults())
    assert rendered["timeframe"] == "1h"


def test_more_than_ten_free_params_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    doc["params"] = {
        f"p{i}": {"kind": "int", "low": 0, "high": 10, "step": 1, "default": 5} for i in range(11)
    }
    doc["constraints"] = []
    dsl = {
        "name": "ema_cross",
        "timeframe": "1h",
        "description": " ".join(f"{{{{p{i}}}}}" for i in range(11)),
        "indicators": {"ema": {"type": "EMA", "period": 12}},
        "entry_conditions": {"long": ["close > ema"], "short": []},
        "exit_conditions": {"long": ["close < ema"], "short": []},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 5.0},
    }
    doc["dsl"] = dsl
    with pytest.raises(PortError):
        _load(tmp_path, doc)


# --- domain table: render values -------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.pop("fast"),
        lambda v: v.update(bogus=1),
        lambda v: v.update(fast=5.0),
        lambda v: v.update(fast=True),
        lambda v: v.update(fast=999),
        lambda v: v.update(mult=3.1),
        lambda v: v.update(ma="ema"),
    ],
)
def test_render_bad_values_rejected(tmp_path: Path, mutate: object) -> None:
    port = _load(tmp_path)
    values = port.defaults()
    assert callable(mutate)
    mutate(values)
    with pytest.raises(PortError):
        port.render(values)


def test_render_returns_deep_copy(tmp_path: Path) -> None:
    port = _load(tmp_path)
    first = port.render(port.defaults())
    indicators = first["indicators"]
    assert isinstance(indicators, dict)
    first["timeframe"] = "4h"
    indicators["ema_slow"] = {}
    second = port.render(port.defaults())
    assert second["timeframe"] == "1h"
    assert second["indicators"] != indicators


# --- domain table: rendered DSL validation ---------------------------------


def test_rendered_dsl_failure_is_port_error(tmp_path: Path) -> None:
    doc = _base_doc()
    dsl = doc["dsl"]
    assert isinstance(dsl, dict)
    dsl["timeframe"] = "2h"
    with pytest.raises(PortError, match="Invalid timeframe"):
        _load(tmp_path, doc)


# --- domain table: file errors ---------------------------------------------


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(PortError):
        load_port(tmp_path / "nope" / "port.yaml")


def test_not_a_mapping_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "09_ema_cross"
    folder.mkdir()
    (folder / "port.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(PortError):
        load_port(folder / "port.yaml")


def test_yaml_error_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "09_ema_cross"
    folder.mkdir()
    (folder / "port.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(PortError):
        load_port(folder / "port.yaml")


# --- placeholders ----------------------------------------------------------


def test_unknown_placeholder_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    dsl = doc["dsl"]
    assert isinstance(dsl, dict)
    dsl["description"] = "{{bogus}}"
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_unused_param_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    params = doc["params"]
    assert isinstance(params, dict)
    params["dead"] = {"kind": "int", "low": 0, "high": 10, "step": 1, "default": 5}
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_placeholder_in_mapping_key_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    dsl = doc["dsl"]
    assert isinstance(dsl, dict)
    indicators = dsl["indicators"]
    assert isinstance(indicators, dict)
    indicators["period_{{fast}}"] = {"type": "SMA", "period": 5}
    with pytest.raises(PortError):
        _load(tmp_path, doc)


# --- timeframe guard (D5) --------------------------------------------------


def test_allowed_timeframes_gate_render(tmp_path: Path) -> None:
    doc = _base_doc()
    doc["allowed_timeframes"] = ["1h"]
    port = _load(tmp_path, doc)
    assert port.allowed_timeframes == ("1h",)
    assert port.render(port.defaults())["timeframe"] == "1h"
    assert port.render(port.defaults(), timeframe="1h")["timeframe"] == "1h"
    with pytest.raises(PortError, match="does not support timeframe 15m"):
        port.render(port.defaults(), timeframe="15m")


def test_template_timeframe_outside_allowed_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    doc["allowed_timeframes"] = ["4h"]
    with pytest.raises(PortError, match="does not support timeframe 1h"):
        _load(tmp_path, doc)


@pytest.mark.parametrize("allowed", [[], ["1h", "1h"], ["2h"], ["1h", 4]])
def test_bad_allowed_timeframes_rejected(tmp_path: Path, allowed: list[object]) -> None:
    doc = _base_doc()
    doc["allowed_timeframes"] = allowed
    with pytest.raises(PortError):
        _load(tmp_path, doc)


def test_explicit_null_allowed_timeframes_rejected(tmp_path: Path) -> None:
    doc = _base_doc()
    doc["allowed_timeframes"] = None
    with pytest.raises(PortError, match="allowed_timeframes"):
        _load(tmp_path, doc)


def test_omitted_allowed_timeframes_is_unrestricted(tmp_path: Path) -> None:
    assert _load(tmp_path).allowed_timeframes is None


def _doc_with_numeric_choice(choices: list[object], default: object) -> dict[str, object]:
    doc = _base_doc()
    params = doc["params"]
    assert isinstance(params, dict)
    params["c"] = {"kind": "choice", "choices": choices, "default": default}
    dsl = doc["dsl"]
    assert isinstance(dsl, dict)
    dsl["description"] = "c={{c}}"
    return doc


@pytest.mark.parametrize(
    ("choices", "default"),
    [
        ([1.0, float("inf")], 1.0),
        ([1.0, float("-inf")], 1.0),
        ([1.0, float("nan")], 1.0),
        ([1, 2**54], 1),
        ([2**54, 1], 2**54),
    ],
)
def test_numeric_choices_reject_invalid_scalars(
    tmp_path: Path, choices: list[object], default: object
) -> None:
    with pytest.raises(PortError, match="params.c"):
        _load(tmp_path, _doc_with_numeric_choice(choices, default))


def test_numeric_choices_at_the_integer_limit_load(tmp_path: Path) -> None:
    port = _load(tmp_path, _doc_with_numeric_choice([1, 2**53, -(2**53)], 1))
    param = next(p for p in port.params if p.name == "c")
    assert param.choices == (1, 2**53, -(2**53))


def test_huge_int_value_for_float_param_is_domain_error(tmp_path: Path) -> None:
    port = _load(tmp_path)
    param = next(p for p in port.params if p.name == "mult")
    values: dict[str, object] = dict(port.defaults())
    values["mult"] = 10**400
    with pytest.raises(PortError):
        port.render(values)
    with pytest.raises(PortError):
        port.check_values(values)
    with pytest.raises(PortError):
        param.canonical(10**400)
    assert param.in_domain(10**400) is False
    assert param.in_domain(2**53 + 1) is False


def test_huge_int_value_for_int_param_is_domain_error(tmp_path: Path) -> None:
    port = _load(tmp_path)
    fast = next(p for p in port.params if p.name == "fast")
    assert fast.in_domain(10**400) is False
    with pytest.raises(PortError):
        fast.canonical(10**400)


@pytest.mark.parametrize(
    ("low", "high", "step", "default"),
    [
        (-1e308, 1e308, 0.1, 0.5),
        (-1e308, 1e308, 1e300, 0.0),
        (0.0, 1e308, 1e-10, 0.0),
    ],
)
def test_overflowing_float_declaration_is_port_error(
    tmp_path: Path, low: float, high: float, step: float, default: float
) -> None:
    doc = _base_doc()
    _set_param(doc, "mult", low=low, high=high, step=step, default=default)
    with pytest.raises(PortError, match="params.mult"):
        _load(tmp_path, doc)


def _doc_with_float_x(**spec: object) -> dict[str, object]:
    doc = _base_doc()
    params = doc["params"]
    assert isinstance(params, dict)
    params["x"] = {"kind": "float", **spec}
    dsl = doc["dsl"]
    assert isinstance(dsl, dict)
    dsl["description"] = "x={{x}}"
    return doc


def test_float_grid_index_overflow_is_port_error(tmp_path: Path) -> None:
    doc = _doc_with_float_x(low=-1e300, high=1e300, step=1e-9, default=0.0)
    with pytest.raises(PortError, match="params.x"):
        _load(tmp_path, doc)


def test_float_value_whose_offset_overflows_is_not_in_domain(tmp_path: Path) -> None:
    port = _load(tmp_path, _doc_with_float_x(low=-1e308, high=0.0, step=1e300, default=0.0))
    x = next(p for p in port.params if p.name == "x")
    assert x.in_domain(1e308) is False
    assert x.in_domain(0.0) is True
    with pytest.raises(PortError):
        x.canonical(1e308)


def test_satisfies_constraints_compares_snapped_operands(tmp_path: Path) -> None:
    doc = _base_doc()
    _set_param(doc, "mult", low=0.0, high=1.0, step=0.1, default=0.5)
    params = doc["params"]
    assert isinstance(params, dict)
    params["lim"] = {"kind": "float", "low": 0.0, "high": 1.0, "step": 0.1, "default": 0.5}
    dsl = doc["dsl"]
    assert isinstance(dsl, dict)
    dsl["description"] = "lim={{lim}}"
    doc["constraints"] = ["fast < slow", "mult <= lim"]
    port = _load(tmp_path, doc)
    values: dict[str, object] = dict(port.defaults())
    values["mult"] = 0.5000000001
    values["lim"] = 0.5
    assert port.satisfies_constraints(values) is True
    port.check_values(values)


def test_try_canonical_values_never_raises(tmp_path: Path) -> None:
    port = _load(tmp_path)
    values: dict[str, object] = dict(port.defaults())
    assert port.try_canonical_values(values) == port.defaults()
    values["mult"] = 10**400
    assert port.try_canonical_values(values) is None
    assert port.try_canonical_values({}) is None
    violating = dict(port.defaults(), fast=40, slow=20)
    assert port.try_canonical_values(violating) is None
