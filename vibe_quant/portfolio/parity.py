"""Engine-vs-reference exact comparison: the third exactness gate (design §5).

``parity.compare_runs`` is the rule-level proof that the NT execution layer and
the no-NT reference simulator produce the *same* book: with one shared rule
implementation (``signal``/``sizing``/``ledger``/``ticks.check_release``), the
same tick prices and NT's own Decimal commission formula, both executors run the
identical functions on bitwise-equal inputs, so every downstream number is equal
-- never approximately equal.

The comparison is therefore EXACT (no tolerance, R30): a 1-step quantity or a
1-ulp fee is a DRIFT that fails ``scripts/agents/exactness_carry.py``, and
widening it needs a chief ruling. Rule *correctness* is proven by the contract
tests on hand-computed numbers, not here; this module only proves that no
execution path diverges from the rules.

Chief ruling R54: the comparison is structural and complete. Every field of every
``CarryFill``, ``Decision`` and ``FundingPayment`` is compared through
``dataclasses.fields()`` -- so a field added later is compared automatically and
no hand-written field list can silently drop one -- as are ``CarryRun.spec`` and
``CarryRun.final_steps`` and every ``CarryReport`` field (``summary``, ``daily``,
``weekly``, ``legs``, ``by_year``). ``CarryRun.engine`` is the ONE excluded
field: it differs by design (``"nt"`` vs ``"reference"``).

A value that is NaN in *both* runs counts as EQUAL (``math.isnan`` on both
sides), so a legitimate NaN never reads as a false drift; NaN against a real
number is still reported. Mismatch messages name the path, index and field, e.g.
``decisions[3].excluded: {'ETHUSDT': 'funding_hole'} != {'ETHUSDT': 'no_price'}``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vibe_quant.portfolio.ledger import CarryRun
    from vibe_quant.portfolio.metrics import CarryReport


def compare_runs(a: CarryRun, b: CarryRun, ra: CarryReport, rb: CarryReport) -> list[str]:
    """Every mismatch between two carry runs and their reports; empty = identical.

    Compares, structurally and in order, every field of both runs -- every field
    of every ``CarryFill``, ``Decision`` and ``FundingPayment``, then ``spec`` and
    ``final_steps`` -- and every field of both reports. ``CarryRun.engine`` is the
    only field skipped (it differs by design). Two runs of the same ``CarryInputs``
    (one per executor) must produce an empty list -- R30, exact equality, no
    tolerance; an empty list is the proof, a non-empty one is a DRIFT.

    Raises:
        AttributeError: if the two runs (or the two reports) are not the same type.
    """
    mismatches: list[str] = []
    for field in fields(a):
        if field.name == "engine":
            continue
        mismatches.extend(_diff(field.name, getattr(a, field.name), getattr(b, field.name)))
    mismatches.extend(_diff("", ra, rb))
    return mismatches


def _diff(path: str, one: object, other: object) -> list[str]:
    """Path-qualified mismatches between two values; empty when they are exactly equal."""
    if _equal(one, other):
        return []
    if is_dataclass(one) and is_dataclass(other):
        if type(one) is not type(other):
            return [f"{path}: {type(one).__name__} != {type(other).__name__}"]
        return [
            message
            for field in fields(cast("Any", one))
            for message in _diff(
                _join(path, field.name), getattr(one, field.name), getattr(other, field.name)
            )
        ]
    if isinstance(one, Mapping) and isinstance(other, Mapping):
        return [f"{path}: {one!r} != {other!r}"]
    if isinstance(one, (tuple, list)) and isinstance(other, (tuple, list)):
        messages = []
        if len(one) != len(other):
            messages.append(f"{path} count: {len(one)} != {len(other)}")
        messages.extend(
            message
            for index, (left, right) in enumerate(zip(one, other, strict=False))
            for message in _diff(f"{path}[{index}]", left, right)
        )
        return messages
    return [f"{path}: {one!r} != {other!r}"]


def _equal(one: object, other: object) -> bool:
    """NaN-aware structural equality (R54): a NaN on both sides is equal.

    Containers and dataclasses are walked so a nested NaN (e.g. a decision's
    ``signals`` or a fill's ``price``) is compared by value, not by object
    identity -- ``float("nan") == float("nan")`` is False, which would otherwise
    report a legitimate NaN as a drift.
    """
    if isinstance(one, float) and isinstance(other, float):
        return (math.isnan(one) and math.isnan(other)) or one == other
    is_data = is_dataclass(one) and is_dataclass(other)
    if is_data and type(one) is type(other):
        return all(
            _equal(getattr(one, field.name), getattr(other, field.name))
            for field in fields(cast("Any", one))
        )
    if isinstance(one, Mapping) and isinstance(other, Mapping):
        return one.keys() == other.keys() and all(_equal(one[key], other[key]) for key in one)
    if isinstance(one, (tuple, list)) and isinstance(other, (tuple, list)):
        return len(one) == len(other) and all(
            _equal(x, y) for x, y in zip(one, other, strict=False)
        )
    return one == other


def _join(path: str, name: str) -> str:
    """``path``.``name`` for a nested field, or just ``name`` at the top level."""
    return f"{path}.{name}" if path else name
