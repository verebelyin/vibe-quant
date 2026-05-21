"""Mapper from LLM ``proposed_indicators`` JSON to ``IndicatorSpec`` args.

Slice 1 of 4 for bd-3p1k.1 — backend foundation only. The mapper takes a
single proposal dict (as emitted by the extractor under
``proposed_indicators_json``) and produces the kwargs that the
``IndicatorSpec`` constructor needs, plus a separate ``compute_fn``
synthesis hook that subsequent slices will fill in (LLM codegen).

The mapper is intentionally pure (no LLM calls, no I/O). It validates
the input shape, normalizes the indicator name, applies range
heuristics where the LLM didn't supply one, and reports per-parameter
provenance (``llm`` vs ``heuristic``) so the slice-2 file writer can
record it in the AUTO-GENERATED header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Names must survive being used as a YAML ``type:`` value AND as a Python
# module filename — same rule as ``plugin_scaffold._NAME_RE``.
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Heuristic param ranges applied when the LLM omits ``range`` on a
# parameter. The rule mirrors the design notes on bd-3p1k.1: period-like
# params (typical default 5-100) get (5, 50); threshold-like params
# (typical default 0-100) get (10, 90); otherwise no GA enrollment.
_PERIOD_RANGE: tuple[float, float] = (5.0, 50.0)
_THRESHOLD_RANGE: tuple[float, float] = (10.0, 90.0)
_PERIOD_DEFAULT_HI = 100
_THRESHOLD_DEFAULT_HI = 100


class InvalidProposalError(ValueError):
    """Raised when a proposal can't be mapped (missing ``formula``, bad name)."""


@dataclass
class IndicatorSpecArgs:
    """Plain dataclass mirror of ``IndicatorSpec.__init__`` kwargs.

    Slice 2 hands this to a rendering function that emits the actual
    ``register_spec(IndicatorSpec(...))`` call in the generated plugin
    file. ``range_provenance`` is *not* on ``IndicatorSpec`` itself — it
    is metadata for the header-comment block only.
    """

    name: str
    default_params: dict[str, Any]
    param_schema: dict[str, type]
    param_ranges: dict[str, tuple[float, float]]
    threshold_range: tuple[float, float] | None
    output_names: tuple[str, ...]
    display_name: str
    description: str
    category: str = "Custom"
    range_provenance: dict[str, str] = field(default_factory=dict)


def normalize_name(raw: Any) -> str:
    """Return the canonical UPPER name for an indicator proposal.

    Raises ``InvalidProposalError`` if the raw string can't be coerced
    into a valid identifier — the snake_case → UPPER coercion drops
    spaces and dashes the same way ``plugin_scaffold._normalize_name``
    does for the CLI path. Accepts ``Any`` because callers pass raw
    LLM JSON straight in; runtime checks the shape.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidProposalError("indicator name is empty")
    coerced = re.sub(r"[\s-]+", "_", raw.strip()).upper()
    if not _NAME_RE.match(coerced):
        raise InvalidProposalError(
            f"indicator name {raw!r} → {coerced!r} is not a valid identifier "
            "(must start with A-Z, contain only A-Z 0-9 _)"
        )
    return coerced


def suggest_alt_name(name: str, existing: set[str]) -> str:
    """Find the first ``<name>_V<n>`` (n >= 2) not present in ``existing``."""
    for n in range(2, 100):
        candidate = f"{name}_V{n}"
        if candidate not in existing:
            return candidate
    # Practically unreachable — caller has bigger problems if they have
    # 99 colliding versions. Fall back to a timestamped name.
    return f"{name}_VX"


def _map_output_range(raw: Any) -> tuple[float, float] | None:
    """Map the LLM ``output_range`` hint to a GA ``threshold_range``.

    The extractor prompt asks for one of ``"0..100"`` | ``"fraction"`` |
    ``"unbounded"``, but real outputs are noisy — anything we don't
    recognize falls through to ``None`` (no GA enrollment for
    thresholds) rather than raising, so a sloppy proposal still
    scaffolds.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if s == "0..100" or s == "0-100" or s == "0 to 100":
        return (20.0, 80.0)
    if s == "fraction" or s == "0..1" or s == "0-1":
        return (0.2, 0.8)
    if s == "unbounded":
        return None
    return None


def _coerce_param_value(raw: Any) -> tuple[Any, type]:
    """Best-effort coerce an LLM-emitted default into (value, type).

    The LLM emits parameters as either bare values (``14``) or
    ``{default, range?, description?}`` dicts. We accept either; this
    function unwraps the ``default`` if present and infers the Python
    type. Strings that parse as numbers are kept as strings unless the
    surrounding ``range`` makes them clearly numeric — slice 2's
    codegen prompt can deal with quoted defaults.
    """
    if isinstance(raw, dict):
        raw = raw.get("default")
    if isinstance(raw, bool):  # bool is a subclass of int — check first
        return raw, bool
    if isinstance(raw, int):
        return raw, int
    if isinstance(raw, float):
        return raw, float
    if isinstance(raw, str):
        return raw, str
    # Fallback: drop unknown types to a string repr so the spec
    # registers something rather than crashing during codegen.
    return str(raw), str


_THRESHOLD_NAME_TOKENS = ("threshold", "overbought", "oversold", "level")
_PERIOD_NAME_TOKENS = ("period", "length", "window", "lookback", "bars")


def _heuristic_range(name: str, default: Any) -> tuple[float, float] | None:
    """Return a (lo, hi) range or None if we can't reasonably guess.

    Order matters: name-based hints win over value-based ones, because
    a default of ``70`` could be either a period or a threshold and the
    parameter name disambiguates (``overbought=70`` is clearly the
    latter, ``period=70`` clearly the former). Only when the name is
    opaque do we fall back to the value-range heuristic.
    """
    low = name.lower()
    if any(tok in low for tok in _THRESHOLD_NAME_TOKENS):
        return _THRESHOLD_RANGE
    if any(tok in low for tok in _PERIOD_NAME_TOKENS):
        return _PERIOD_RANGE
    # Value-based fallback. Periods tend to be small ints (5-50);
    # thresholds tend to land in 20-80. The split at 50 is rough but
    # matches typical default conventions across RSI/Stoch/etc.
    if isinstance(default, int) and 1 <= default <= _PERIOD_DEFAULT_HI:
        if default > 50:
            return _THRESHOLD_RANGE
        return _PERIOD_RANGE
    if isinstance(default, float) and 0 <= default <= _THRESHOLD_DEFAULT_HI:
        return _THRESHOLD_RANGE if default > 1.0 else None
    return None


def _map_parameters(
    raw: Any,
) -> tuple[
    dict[str, Any],
    dict[str, type],
    dict[str, tuple[float, float]],
    dict[str, str],
]:
    """Split a raw ``parameters`` dict into the four IndicatorSpec slots.

    Returns ``(default_params, param_schema, param_ranges,
    range_provenance)``. ``range_provenance`` is keyed by param name
    and is one of ``"llm"`` (LLM supplied an explicit range) or
    ``"heuristic"`` (range came from this module's fallback).
    Parameters with no LLM range AND no plausible heuristic are simply
    not entered into ``param_ranges`` — they are still registered (with
    a default), just not GA-enrolled.
    """
    if not isinstance(raw, dict):
        return {}, {}, {}, {}

    defaults: dict[str, Any] = {}
    schema: dict[str, type] = {}
    ranges: dict[str, tuple[float, float]] = {}
    provenance: dict[str, str] = {}

    for key, val in raw.items():
        if not isinstance(key, str) or not key.isidentifier():
            continue
        default, py_type = _coerce_param_value(val)
        defaults[key] = default
        schema[key] = py_type

        if isinstance(val, dict):
            llm_range = val.get("range")
            if (
                isinstance(llm_range, (list, tuple))
                and len(llm_range) == 2
                and all(isinstance(x, (int, float)) for x in llm_range)
                and llm_range[0] < llm_range[1]
            ):
                ranges[key] = (float(llm_range[0]), float(llm_range[1]))
                provenance[key] = "llm"
                continue

        guessed = _heuristic_range(key, default)
        if guessed is not None:
            ranges[key] = guessed
            provenance[key] = "heuristic"

    return defaults, schema, ranges, provenance


def _map_outputs(raw: Any) -> tuple[str, ...]:
    """Map the optional ``outputs`` array to ``IndicatorSpec.output_names``."""
    if isinstance(raw, list) and raw:
        clean = tuple(s for s in raw if isinstance(s, str) and s.isidentifier())
        if clean:
            return clean
    return ("value",)


def proposed_to_spec_args(proposed: dict[str, Any]) -> IndicatorSpecArgs:
    """Convert one ``proposed_indicators_json`` entry to spec kwargs.

    Refuses (``InvalidProposalError``) when ``formula`` is missing — the
    endpoint maps that refusal to status=invalid_input. All other fields
    have sensible fallbacks: a proposal with no ``parameters`` simply
    registers as a parameterless indicator.
    """
    if not isinstance(proposed, dict):
        raise InvalidProposalError("proposal is not a dict")
    formula = proposed.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        raise InvalidProposalError(
            "proposal missing 'formula' — won't synthesize indicator without one"
        )

    name = normalize_name(proposed.get("name"))
    display = proposed.get("display_name")
    desc = proposed.get("description")
    defaults, schema, ranges, provenance = _map_parameters(
        proposed.get("parameters")
    )
    threshold_range = _map_output_range(proposed.get("output_range"))
    output_names = _map_outputs(proposed.get("outputs"))

    return IndicatorSpecArgs(
        name=name,
        default_params=defaults,
        param_schema=schema,
        param_ranges=ranges,
        threshold_range=threshold_range,
        output_names=output_names,
        display_name=(
            display.strip()
            if isinstance(display, str) and display.strip()
            else name.replace("_", " ").title()
        ),
        description=(
            desc.strip()
            if isinstance(desc, str) and desc.strip()
            else f"{name} — proposed indicator awaiting promotion."
        ),
        range_provenance=provenance,
    )
