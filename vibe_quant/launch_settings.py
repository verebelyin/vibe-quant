"""The launch parameters of one backtest run, parsed in each entry point's order.

Bead vibe-quant-rabmo.3. Stdlib only, and deliberately outside
:mod:`vibe_quant.liquidation` *and* the engine stack: importing this module must not
load ``nautilus_trader`` or ``vibe_quant.liquidation.*`` (R32), because the API
imports launch settings on every request and a mode-off run must stay light.

Two things live here:

- :data:`PARSE_ORDER` — each entry point's parse order (R27). The field order is one
  canonical order (balance, liquidation, sizing, leverage_cap, entry_order, fee); the
  rows differ only at the phase splits (validation parses its balance and fee after
  detail/clamp/latency, the pipeline parses the fee after
  ``resolve_parent_fill_ticks``). :func:`iter_fields` lets a caller print or act
  between two fields, so a partial stdout stays possible.
- :class:`LaunchSettings` — the already-parsed, validated settings of one run.

The component parsers stay where they are (:mod:`vibe_quant.balance_settings`,
:mod:`vibe_quant.liquidation_settings`, :mod:`vibe_quant.sizing_settings`,
:mod:`vibe_quant.fees`); this module only sequences them and holds the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from vibe_quant.balance_settings import (
    DEFAULT_STARTING_BALANCE_USDT,
    parse_initial_balance,
    recorded_balance,
    validate_starting_balance,
)
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import DEFAULT_LEVERAGE, LiquidationSettings
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator, Mapping

Field = Literal["balance", "liquidation", "sizing", "leverage_cap", "entry_order", "fee"]
"""One launch setting, as the parse table names it.

``leverage_cap`` is the sizing cap's leverage check (a validation step, not a stored
setting): it is parsed from the liquidation and sizing fields that came before it.
"""

PARSE_ORDER: dict[str, tuple[Field, ...]] = {
    # screening/__main__.py 124-160: the canonical field order. The CLI never checks
    # the cap; create_screening_pipeline does.
    "screening_cli": ("balance", "liquidation", "sizing", "fee"),
    # create_screening_pipeline 579-613, before the mock/real split.
    "screening_pipeline_pre_tick": (
        "balance",
        "liquidation",
        "sizing",
        "leverage_cap",
        "entry_order",
    ),
    # Real branch only, after resolve_parent_fill_ticks; the mock branch never parses
    # the fee (pipeline.py:610/613).
    "screening_pipeline_post_tick": ("fee",),
    # validation run 388-440 / run_walk_forward 629-664, before detail/clamp/latency.
    "validation_pre_detail": ("liquidation", "sizing", "leverage_cap"),
    # The same entry points, inside _create_venue_config, after detail/clamp/latency.
    "validation_post_detail": ("balance", "fee"),
    # _create_venue_config(run_config, ...) with no already-parsed settings: balance,
    # liquidation, fee; the sizing is never parsed (runner.py:1118-1133).
    "validation_direct_venue": ("balance", "liquidation", "fee"),
    # WFA/CV _resolve: the recorded balance is read from the DB first (R28), then the
    # stored parameters in this order.
    "parent_run": ("liquidation", "sizing", "leverage_cap", "fee"),
}
"""Each entry point's parse order (R27): the canonical field order, phase splits only."""

_EMPTY_PARSED: Mapping[Field, object] = {}
"""Default for :func:`iter_fields`' ``parsed`` (never mutated; a shared empty mapping)."""


def _parse_field(
    name: Field, params: Mapping[str, object], parsed: Mapping[Field, object]
) -> object:
    """Parse ONE field with its component parser, raising the component's own error.

    ``parsed`` carries the fields already parsed (in this call or an earlier one) so
    the ``leverage_cap`` step sees the liquidation and sizing it checks.
    """
    if name == "balance":
        return parse_initial_balance(params)
    if name == "liquidation":
        return LiquidationSettings.from_params(params)
    if name == "sizing":
        return SizingSettings.from_params(params)
    if name == "leverage_cap":
        liquidation = parsed.get("liquidation")
        sizing = parsed.get("sizing")
        checked = sizing if isinstance(sizing, SizingSettings) else SizingSettings()
        leverage = (
            liquidation.leverage
            if isinstance(liquidation, LiquidationSettings)
            else DEFAULT_LEVERAGE
        )
        checked.check_leverage(leverage)
        return None
    if name == "entry_order":
        return str(params.get("entry_order", "market"))
    return FeeSchedule.from_params(params)


def iter_fields(
    params: Mapping[str, object],
    order: tuple[Field, ...],
    *,
    parsed: Mapping[Field, object] = _EMPTY_PARSED,
) -> Iterator[tuple[Field, object]]:
    """Yield ``(field, value)`` one at a time, in ``order``, parsing lazily.

    Each step runs only that field's component parser, so it raises that component's
    own error at that step and the caller can print or act between steps (the
    screening CLI's partial stdout, R27c). Reuse ``parsed`` to continue in a second
    phase (validation parses around the detail/clamp/latency block): the fields it
    carries are visible to a later ``leverage_cap`` step.

    Args:
        params: The run's stored parameters (missed keys take the component default).
        order: The entry point's parse order, from :data:`PARSE_ORDER`.
        parsed: Fields parsed by an earlier call, if any.

    Yields:
        ``(field, value)``; ``leverage_cap`` yields ``None`` (it is a check, not a
        stored setting).
    """
    seen: dict[Field, object] = dict(parsed)
    for name in order:
        value = _parse_field(name, params, seen)
        seen[name] = value
        yield name, value


@dataclass(frozen=True)
class LaunchSettings:
    """The validated launch settings of one run.

    Built from :func:`iter_fields` output (:meth:`from_fields`), from a run's
    parameters (:meth:`from_params`) or from a parent run (:meth:`from_parent_run`)
    so no entry point re-parses (or forgets to validate) a field.
    """

    fee_schedule: FeeSchedule = FeeSchedule()
    liquidation: LiquidationSettings = LiquidationSettings()
    sizing: SizingSettings = SizingSettings()
    starting_balance: int = DEFAULT_STARTING_BALANCE_USDT
    entry_order: str = "market"

    def __post_init__(self) -> None:
        # The same two rules the entry points ran inline: a bad balance always fails
        # (never defaulted), and a cap the run's leverage cannot cover is refused.
        object.__setattr__(
            self, "starting_balance", validate_starting_balance(self.starting_balance)
        )
        self.sizing.check_leverage(self.liquidation.leverage)

    @classmethod
    def from_fields(cls, fields: Mapping[Field, object]) -> LaunchSettings:
        """Build from already-parsed fields; a missing field takes its default."""
        fee = fields.get("fee")
        liquidation = fields.get("liquidation")
        sizing = fields.get("sizing")
        balance = fields.get("balance")
        entry_order = fields.get("entry_order")
        return cls(
            fee_schedule=fee if isinstance(fee, FeeSchedule) else FeeSchedule(),
            liquidation=(
                liquidation
                if isinstance(liquidation, LiquidationSettings)
                else LiquidationSettings()
            ),
            sizing=sizing if isinstance(sizing, SizingSettings) else SizingSettings(),
            starting_balance=(
                cast("int", balance) if balance is not None else DEFAULT_STARTING_BALANCE_USDT
            ),
            entry_order=str(entry_order) if entry_order is not None else "market",
        )

    @classmethod
    def from_params(cls, params: Mapping[str, object], order: tuple[Field, ...]) -> LaunchSettings:
        """Parse ``params`` in ``order`` and build the settings (R25: ``{}`` = defaults)."""
        return cls.from_fields(dict(iter_fields(params, order)))

    @classmethod
    def from_parent_run(cls, conn: sqlite3.Connection, run_id: int) -> LaunchSettings:
        """The settings a WFA/CV rerun of ``run_id`` must run at (R28).

        The balance is the parent's RECORDED one (:func:`parent_effective_balance`),
        read before any stored parameter is parsed, and the stored
        ``parameters.initial_balance`` is ignored (a legacy run's params may disagree
        with every sweep row). The remaining fields come from
        ``backtest_runs.parameters`` in :data:`PARSE_ORDER`'s ``parent_run`` order;
        ``entry_order`` is always ``"market"``.

        Raises:
            ValueError: the parent recorded no balance (from
                :func:`parent_effective_balance`), or a stored setting is invalid.
        """
        starting_balance = parent_effective_balance(conn, run_id)
        row = conn.execute(
            "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        params = _run_params(row[0] if row is not None else None)
        fields = dict(iter_fields(params, PARSE_ORDER["parent_run"]))
        fields["balance"] = starting_balance
        return cls.from_fields(fields)


def parent_effective_balance(conn: sqlite3.Connection, run_id: int) -> int:
    """The account size a rerun (WFA / CV) of ``run_id`` must run at (R11).

    The parent run's own RECORDED balance, never its ``parameters`` (a legacy run's
    params may say 10000 while every sweep row is NULL — R11b):

    - the run has sweep rows: their single distinct ``starting_balance`` through
      :func:`~vibe_quant.balance_settings.recorded_balance`, where a NULL column means
      the default 1000 (every row written before the column existed).
    - no sweep rows: ``backtest_results.starting_balance``, the same way.
    - neither: ``ValueError`` — the run recorded no balance and the rerun fails closed.

    Raises:
        ValueError: the sweep rows disagree (no single value), or the run has no
            recorded balance at all.
    """
    rows = conn.execute(
        "SELECT DISTINCT starting_balance FROM sweep_results WHERE run_id = ?", (run_id,)
    ).fetchall()
    if rows:
        values = {row[0] for row in rows}
        if len(values) > 1:
            raise ValueError(
                f"run {run_id} has conflicting sweep starting_balance values: "
                f"{sorted(values, key=str)}"
            )
        return recorded_balance(values.pop())
    row = conn.execute(
        "SELECT starting_balance FROM backtest_results WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"run {run_id} has no recorded starting balance")
    return recorded_balance(row[0])


def _run_params(raw: object) -> dict[str, object]:
    """The run's stored ``parameters`` as a dict (``{}`` when unusable)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}
