"""Unit tests for the proposed_indicators → IndicatorSpec mapper."""

from __future__ import annotations

import pytest

from vibe_quant.research.indicator_scaffold import (
    InvalidProposalError,
    normalize_name,
    proposed_to_spec_args,
    suggest_alt_name,
)


def test_normalize_name_uppercases_snake_case() -> None:
    assert normalize_name("my_ind") == "MY_IND"


def test_normalize_name_coerces_spaces_and_dashes() -> None:
    assert normalize_name(" my ind ") == "MY_IND"
    assert normalize_name("my-ind") == "MY_IND"


def test_normalize_name_rejects_leading_digit() -> None:
    with pytest.raises(InvalidProposalError):
        normalize_name("1ind")


def test_normalize_name_rejects_empty() -> None:
    with pytest.raises(InvalidProposalError):
        normalize_name("")


def test_suggest_alt_name_returns_first_unused() -> None:
    assert suggest_alt_name("RSI", {"RSI"}) == "RSI_V2"
    assert suggest_alt_name("RSI", {"RSI", "RSI_V2"}) == "RSI_V3"


def test_proposed_to_spec_args_minimum_happy_path() -> None:
    args = proposed_to_spec_args(
        {
            "name": "rsi_variant",
            "formula": "100 - 100 / (1 + RS)",
            "parameters": {"period": {"default": 14, "range": [5, 30]}},
            "output_range": "0..100",
        }
    )
    assert args.name == "RSI_VARIANT"
    assert args.default_params == {"period": 14}
    assert args.param_schema == {"period": int}
    assert args.param_ranges == {"period": (5.0, 30.0)}
    assert args.threshold_range == (20.0, 80.0)
    assert args.output_names == ("value",)
    assert args.range_provenance == {"period": "llm"}


def test_proposed_to_spec_args_rejects_missing_formula() -> None:
    with pytest.raises(InvalidProposalError):
        proposed_to_spec_args({"name": "x", "parameters": {}})


def test_proposed_to_spec_args_rejects_blank_formula() -> None:
    with pytest.raises(InvalidProposalError):
        proposed_to_spec_args({"name": "x", "formula": "   "})


def test_proposed_to_spec_args_heuristic_period_range() -> None:
    args = proposed_to_spec_args(
        {
            "name": "ind",
            "formula": "sma(close, period)",
            "parameters": {"period": 14},
        }
    )
    assert args.param_ranges == {"period": (5.0, 50.0)}
    assert args.range_provenance == {"period": "heuristic"}


def test_proposed_to_spec_args_heuristic_threshold_range() -> None:
    args = proposed_to_spec_args(
        {
            "name": "ind",
            "formula": "x > threshold",
            "parameters": {"overbought": 70},
        }
    )
    assert args.param_ranges == {"overbought": (10.0, 90.0)}
    assert args.range_provenance == {"overbought": "heuristic"}


def test_proposed_to_spec_args_no_range_for_unknown_param() -> None:
    args = proposed_to_spec_args(
        {
            "name": "ind",
            "formula": "f(x)",
            "parameters": {"opaque_knob": "auto"},
        }
    )
    # String param with no name/value hint — register default + schema
    # but skip GA enrollment for this param.
    assert args.default_params == {"opaque_knob": "auto"}
    assert args.param_schema == {"opaque_knob": str}
    assert args.param_ranges == {}
    assert args.range_provenance == {}


def test_proposed_to_spec_args_output_range_fraction() -> None:
    args = proposed_to_spec_args(
        {"name": "ind", "formula": "f", "output_range": "fraction"}
    )
    assert args.threshold_range == (0.2, 0.8)


def test_proposed_to_spec_args_output_range_unbounded() -> None:
    args = proposed_to_spec_args(
        {"name": "ind", "formula": "f", "output_range": "unbounded"}
    )
    assert args.threshold_range is None


def test_proposed_to_spec_args_output_range_garbage_drops_to_none() -> None:
    args = proposed_to_spec_args(
        {"name": "ind", "formula": "f", "output_range": "wibble"}
    )
    assert args.threshold_range is None


def test_proposed_to_spec_args_multi_output() -> None:
    args = proposed_to_spec_args(
        {
            "name": "stoch_v2",
            "formula": "k = ..., d = sma(k, 3)",
            "outputs": ["k", "d"],
        }
    )
    assert args.output_names == ("k", "d")


def test_proposed_to_spec_args_outputs_default_when_missing() -> None:
    args = proposed_to_spec_args({"name": "x", "formula": "f"})
    assert args.output_names == ("value",)


def test_proposed_to_spec_args_display_name_fallback() -> None:
    args = proposed_to_spec_args({"name": "my_ind", "formula": "f"})
    assert args.display_name == "My Ind"


def test_proposed_to_spec_args_display_name_passthrough() -> None:
    args = proposed_to_spec_args(
        {"name": "x", "formula": "f", "display_name": "Custom Label"}
    )
    assert args.display_name == "Custom Label"


def test_proposed_to_spec_args_description_fallback() -> None:
    args = proposed_to_spec_args({"name": "x", "formula": "f"})
    assert "X" in args.description


def test_proposed_to_spec_args_bare_param_value() -> None:
    # LLM emits a bare int instead of {default: 14}.
    args = proposed_to_spec_args(
        {"name": "x", "formula": "f", "parameters": {"period": 14}}
    )
    assert args.default_params == {"period": 14}
    assert args.param_schema == {"period": int}


def test_proposed_to_spec_args_float_param() -> None:
    args = proposed_to_spec_args(
        {"name": "x", "formula": "f", "parameters": {"alpha": 0.5}}
    )
    assert args.default_params == {"alpha": 0.5}
    assert args.param_schema == {"alpha": float}


def test_proposed_to_spec_args_invalid_range_falls_through_to_heuristic() -> None:
    # LLM emits range with lo >= hi — invalid, so heuristic kicks in.
    args = proposed_to_spec_args(
        {
            "name": "x",
            "formula": "f",
            "parameters": {"period": {"default": 14, "range": [50, 5]}},
        }
    )
    assert args.param_ranges == {"period": (5.0, 50.0)}
    assert args.range_provenance == {"period": "heuristic"}
