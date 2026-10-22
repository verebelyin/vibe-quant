"""The ONE starting-balance rule, shared by the API, validation and screening.

Bead vibe-quant-zpbmo (task Z1). Stdlib only, and deliberately outside
:mod:`vibe_quant.validation`: the API schema imports it on every launch request, and a
mode-off run must not pull the validation/engine stack.

Two directions of the same rule:

- a launch request's ``parameters.initial_balance`` -> :func:`parse_initial_balance`
- a *recorded* balance (``sweep_results.starting_balance`` and
  ``backtest_results.starting_balance``) -> :func:`recorded_balance`, where a NULL
  column (any row written before the column existed) means the default.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_STARTING_BALANCE_USDT: int = 1_000
"""The one default account size; ``validation/venue.py`` re-exports it."""

MIN_STARTING_BALANCE: int = 1
MAX_STARTING_BALANCE: int = 1_000_000_000

MISSING: object = object()
"""Sentinel for an ``initial_balance`` a caller did not supply (:func:`parse_initial_balance`)."""


class BalanceConfigError(ValueError):
    """A starting balance is invalid (it is never silently defaulted)."""


def _invalid(value: object) -> BalanceConfigError:
    return BalanceConfigError(
        "initial_balance must be a whole number in "
        f"[{MIN_STARTING_BALANCE}, {MAX_STARTING_BALANCE}], got {value!r}"
    )


def validate_starting_balance(value: object) -> int:
    """The whole number ``value`` stands for, or :class:`BalanceConfigError`.

    The type and the range are checked BEFORE any ``float()``/``int()`` conversion: an
    int compares against the bounds exactly (``10**400`` would raise OverflowError if
    converted) and a float must be finite and integral (``nan``/``inf``/``1000.5`` are
    not whole numbers). Booleans are not numbers here — ``True`` is not 1.
    """
    if isinstance(value, bool):
        raise _invalid(value)
    if isinstance(value, int) and MIN_STARTING_BALANCE <= value <= MAX_STARTING_BALANCE:
        return value
    if (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
        and MIN_STARTING_BALANCE <= value <= MAX_STARTING_BALANCE
    ):
        return int(value)
    raise _invalid(value)


def parse_initial_balance(params: Mapping[str, object]) -> int:
    """The validated ``parameters.initial_balance`` of a launch request.

    A missing key means :data:`DEFAULT_STARTING_BALANCE_USDT`; an explicit ``None`` is
    rejected like every other non-number, so a client that sends null learns about it
    instead of quietly getting the default. The mapping is never modified.
    """
    value = params.get("initial_balance", MISSING)
    if value is MISSING:
        return DEFAULT_STARTING_BALANCE_USDT
    return validate_starting_balance(value)


def recorded_balance(value: object) -> int:
    """The one rule for reading a balance recorded in the database (R13b).

    ``None`` (a NULL column: an older row) means :=data:`DEFAULT_STARTING_BALANCE_USDT`;
    anything else must pass the same validation, so a corrupt recorded value is an
    error rather than a silent default. Used by the results API and by
    ``launch_settings.parent_effective_balance``.
    """
    if value is None:
        return DEFAULT_STARTING_BALANCE_USDT
    return validate_starting_balance(value)
