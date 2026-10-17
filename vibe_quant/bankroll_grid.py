"""The ONE rule for a discovery bankroll grid (bead vibe-quant-t8nrv, task T8).

A grid is margin mode x leverage x margin cushion (``max_margin_pct``). Discovery searches every
cell and picks across all of them, so each champion records the cell it was found under and
promote / replay inherit exactly that cell.

Stdlib plus the two shared setting rules, nothing else: the API schema imports this on every
launch request, so validating a grid must pull none of the engine stack (import isolation, like
:mod:`vibe_quant.sizing_settings` and :mod:`vibe_quant.liquidation_settings`).

The domain (user decisions 2026-10-16): modes ``isolated`` / ``cross`` only (a grid never runs
``off``), leverage a whole number from 1x up (1x isolated is the spot-like setup), at most
:data:`MAX_CELLS` cells. Each cell is checked with the same rules as a single setup: the shared
``LiquidationSettings`` for mode + leverage, ``SizingSettings`` for the cushion, and
``SizingSettings.check_leverage`` for the run's exposure cap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING, cast

from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings
from vibe_quant.sizing_settings import SizingConfigError, SizingSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.liquidation_settings import LiquidationMode

GRID_MODES: tuple[str, ...] = ("isolated", "cross")
"""The modes a grid may run, in cell order."""

MAX_CELLS = 12
"""The most cells one grid may have (user decision Q7)."""

RUN_LEVEL_CONFLICT = "use either the bankroll grid or --liquidation-mode/--leverage/--max-margin-pct"


class BankrollGridError(ValueError):
    """A bankroll grid (or a recorded cell) is invalid. Never silently defaulted."""


def _check_mode(mode: object) -> str:
    if not isinstance(mode, str) or mode not in GRID_MODES:
        msg = f"bankroll mode must be 'isolated' or 'cross' (a grid never runs 'off'), got {mode!r}"
        raise BankrollGridError(msg)
    return mode


def _check_leverage(mode: str, leverage: object) -> float:
    try:
        return LiquidationSettings(mode=cast("LiquidationMode", mode), leverage=cast("float", leverage)).leverage
    except LiquidationConfigError as exc:
        raise BankrollGridError(f"bankroll {exc}") from exc


def _check_cushion(pct: object) -> float:
    try:
        settings = SizingSettings.from_params({"max_margin_pct": pct})
    except SizingConfigError as exc:
        raise BankrollGridError(f"bankroll {exc}") from exc
    return cast("float", settings.max_margin_pct)


@dataclass(frozen=True)
class BankrollCell:
    """One validated setup of the grid: margin mode, leverage and margin cushion."""

    mode: LiquidationMode
    leverage: float
    max_margin_pct: float

    def __post_init__(self) -> None:
        mode = _check_mode(self.mode)
        object.__setattr__(self, "leverage", _check_leverage(mode, self.leverage))
        object.__setattr__(self, "max_margin_pct", _check_cushion(self.max_margin_pct))

    @property
    def label(self) -> str:
        """``(isolated, 1.0, 0.5)``: how errors and logs name the cell."""
        return f"({self.mode}, {self.leverage!r}, {self.max_margin_pct!r})"

    @property
    def liquidation(self) -> LiquidationSettings:
        """The cell's run-level liquidation settings."""
        return LiquidationSettings(mode=self.mode, leverage=self.leverage)

    def sizing(self, base: SizingSettings) -> SizingSettings:
        """The run's sizing (cap, risk) plus this cell's cushion."""
        return replace(base, max_margin_pct=self.max_margin_pct)

    def to_note(self) -> dict[str, object]:
        """The ``bankroll`` payload a champion entry (and ``notes.bankroll_grid.cells``) carries."""
        return {
            "liquidation_mode": self.mode,
            "leverage": self.leverage,
            "max_margin_pct": self.max_margin_pct,
        }

    @classmethod
    def from_note(cls, raw: object) -> BankrollCell:
        """Read a recorded ``bankroll`` payload; anything but a complete valid one is an error."""
        if not isinstance(raw, dict):
            msg = f"bankroll must be an object, got {raw!r}"
            raise BankrollGridError(msg)
        missing = [
            key for key in ("liquidation_mode", "leverage", "max_margin_pct") if key not in raw
        ]
        if missing:
            msg = f"bankroll is missing {missing}"
            raise BankrollGridError(msg)
        return cls(
            mode=cast("LiquidationMode", raw["liquidation_mode"]),
            leverage=cast("float", raw["leverage"]),
            max_margin_pct=cast("float", raw["max_margin_pct"]),
        )


def _non_empty(field: str, values: Sequence[object]) -> None:
    if not values:
        msg = f"bankroll grid {field} must not be empty"
        raise BankrollGridError(msg)


def build_cells(
    modes: Sequence[object],
    leverages: Sequence[object],
    max_margin_pcts: Sequence[object],
    sizing: SizingSettings | None = None,
) -> list[BankrollCell]:
    """The grid's cells: product(modes in (isolated, cross) order, sorted leverages, sorted
    cushions), deduplicated, 1..:data:`MAX_CELLS` of them.

    ``sizing`` is the run's exposure cap / risk: every cell must be able to margin the cap at its
    leverage (``check_leverage``), and the first cell that cannot rejects the whole grid before
    any search starts.
    """
    _non_empty("modes", modes)
    _non_empty("leverages", leverages)
    _non_empty("max_margin_pcts", max_margin_pcts)
    wanted_modes = {_check_mode(m) for m in modes}
    levs = sorted({_check_leverage("isolated", lev) for lev in leverages})
    pcts = sorted({_check_cushion(p) for p in max_margin_pcts})
    ordered_modes = [m for m in GRID_MODES if m in wanted_modes]
    count = len(ordered_modes) * len(levs) * len(pcts)
    if count > MAX_CELLS:
        msg = f"bankroll grid has {count} cells; at most {MAX_CELLS} are allowed"
        raise BankrollGridError(msg)
    base = sizing if sizing is not None else SizingSettings()
    cells = [
        BankrollCell(cast("LiquidationMode", m), lev, pct)
        for m, lev, pct in product(ordered_modes, levs, pcts)
    ]
    for cell in cells:
        try:
            cell.sizing(base).check_leverage(cell.leverage)
        except SizingConfigError as exc:
            raise BankrollGridError(f"bankroll cell {cell.label}: {exc}") from exc
    return cells


def reject_run_level_with_grid(
    *,
    liquidation_mode: object,
    leverage: object,
    liquidation_leverage: object,
    max_margin_pct: object,
) -> None:
    """A grid owns mode, leverage and cushion: any run-level value alongside it is an error.

    ``liquidation_mode == "off"`` and ``None`` are the unset values (the defaults of both the CLI
    flags and the API fields).
    """
    if (
        liquidation_mode not in ("off", None)
        or leverage is not None
        or liquidation_leverage is not None
        or max_margin_pct is not None
    ):
        raise BankrollGridError(RUN_LEVEL_CONFLICT)


def bankroll_grid_note(cells: Sequence[BankrollCell]) -> dict[str, object]:
    """``notes.bankroll_grid``: every cell tried and how many."""
    return {"cells": [c.to_note() for c in cells], "n_cells": len(cells)}
