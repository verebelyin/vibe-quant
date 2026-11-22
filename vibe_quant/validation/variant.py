"""State-DB-free validation variant of a completed run (bankroll report seam, 2gyk1.3 §2.2).

:func:`run_variant` re-runs a validation run's recorded inputs, optionally with parameter
overrides, through the same helpers and engine path as
:meth:`~vibe_quant.validation.runner.ValidationRunner.run`, in ``run()``'s order. It never
opens the state DB or an event log, never runs the maker market twin, and reads market data
only from the catalog and archive paths it is handed (both required: no default resolves to
``data/catalog`` or the live archive).

With no overrides it first proves it is replaying the parent's inputs: its own
:func:`~vibe_quant.validation.runner.validation_input_identity` must equal the parent's
recorded ``notes["validation_input"]`` key for key, else :class:`VariantInputsMismatch`
before any engine is built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibe_quant.launch_settings import PARSE_ORDER
from vibe_quant.validation.funding import FundingCalculator
from vibe_quant.validation.runner import (
    ValidationRunner,
    ValidationRunnerError,
    _launch_params,
    _parse_launch_fields,
    validation_input_identity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from vibe_quant.dsl.schema import StrategyDSL
    from vibe_quant.launch_settings import LaunchSettings
    from vibe_quant.validation.latency import LatencyPreset
    from vibe_quant.validation.results import ValidationResult
    from vibe_quant.validation.venue import VenueConfig


@dataclass(frozen=True)
class VariantInput:
    """The parent's inputs, snapshotted once (no DB read after this).

    Attributes:
        strategy_name: The strategy's stored name.
        dsl_json: The stored ``dsl_config`` as JSON.
        run_config_json: The parent's ``backtest_runs`` row as JSON
            (``StateManager.get_backtest_run`` shape: ``symbols``/``parameters`` decoded).
        parent_identity: The parent's ``notes["validation_input"]``.
    """

    strategy_name: str
    dsl_json: str
    run_config_json: str
    parent_identity: Mapping[str, object]


class VariantInputsMismatch(ValidationRunnerError):  # noqa: N818 — named by the design
    """The variant's resolved inputs differ from the parent's recorded ones."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        super().__init__(f"variant inputs differ from the parent's in: {', '.join(keys)}")


@dataclass(frozen=True)
class _Resolved:
    """``run()``'s pre-engine helper chain, resolved for one set of inputs."""

    dsl: StrategyDSL
    run_config: dict[str, object]
    window_note: dict[str, object] | None
    detail: str | None
    latency: LatencyPreset | str | None
    venue_config: VenueConfig
    launch: LaunchSettings
    identity: dict[str, object]


def _resolve(
    runner: ValidationRunner,
    *,
    strategy_name: str,
    dsl_raw: Mapping[str, object],
    run_config: dict[str, object],
    detail_override: str | None,
    latency_override: str | None,
) -> _Resolved:
    """run()'s helper calls before its engine, in run()'s order (its failure precedence)."""
    dsl = runner._validate_dsl(dict(dsl_raw), strategy_name=strategy_name)
    entry_order = runner._resolve_maker_params(run_config)
    pre = _parse_launch_fields(_launch_params(run_config), PARSE_ORDER["validation_pre_detail"])
    effective_detail = runner._resolve_detail_timeframe(run_config, dsl.timeframe, detail_override)
    run_config, window_note = runner._clamp_run_window(
        run_config, effective_detail or dsl.timeframe
    )
    effective_latency = runner._resolve_latency(run_config, latency_override)
    venue_config, launch = runner._create_venue_config(
        run_config,
        effective_latency,
        timeframe=dsl.timeframe,
        has_detail_data=effective_detail is not None,
        pre=pre,
    )
    identity = validation_input_identity(
        strategy_name=strategy_name,
        dsl=dsl,
        run_config=run_config,
        detail_timeframe=effective_detail,
        latency_preset=effective_latency,
        venue_config=venue_config,
        entry_order=entry_order,
    )
    return _Resolved(
        dsl=dsl,
        run_config=run_config,
        window_note=window_note,
        detail=effective_detail,
        latency=effective_latency,
        venue_config=venue_config,
        launch=launch,
        identity=identity,
    )


def _run(
    runner: ValidationRunner,
    r: _Resolved,
    *,
    strategy_name: str,
    archive_path: Path,
    mtm_probe: bool,
) -> ValidationResult:
    """run()'s primary backtest plus the notes it adds (no market twin, nothing persisted)."""
    run_id_raw = r.run_config.get("id", 0)
    result = runner._run_backtest(
        run_id=int(run_id_raw) if isinstance(run_id_raw, int) else 0,
        strategy_name=strategy_name,
        dsl=r.dsl,
        venue_config=r.venue_config,
        run_config=r.run_config,
        writer=None,
        detail_timeframe=r.detail,
        write_trade_events=False,
        launch=r.launch,
        mtm_probe=mtm_probe,
    )
    if r.window_note is not None:
        result.notes["data_window"] = r.window_note
    result.notes.update(runner._execution_notes(r.run_config, r.detail))
    result.notes["validation_input"] = r.identity
    result.notes["fee_schedule"] = r.venue_config.fee_schedule.to_notes()
    if result.funding_fallback_settlements:
        result.notes["funding"] = {
            "fallback_settlements": result.funding_fallback_settlements,
            "fallback_rate_per_8h": FundingCalculator(archive_path).default_rate,
        }
    return result


def _run_resolved(
    runner: ValidationRunner,
    *,
    strategy_name: str,
    dsl_raw: Mapping[str, object],
    run_config: dict[str, object],
    archive_path: Path,
    detail_override: str | None,
    latency_override: str | None,
    mtm_probe: bool,
) -> tuple[ValidationResult, dict[str, object]]:
    """Resolve, then run: ``(result, identity)``."""
    r = _resolve(
        runner,
        strategy_name=strategy_name,
        dsl_raw=dsl_raw,
        run_config=run_config,
        detail_override=detail_override,
        latency_override=latency_override,
    )
    result = _run(
        runner, r, strategy_name=strategy_name, archive_path=archive_path, mtm_probe=mtm_probe
    )
    return result, r.identity


def _frozen_config(run_config: Mapping[str, object]) -> dict[str, object]:
    """A copy of the caller's run config with its parameters as run_variant spells them."""
    config = dict(run_config)
    config["parameters"] = dict(_launch_params(config))
    return config


def resolve_frozen(
    *,
    strategy_name: str,
    dsl_json: str,
    run_config: Mapping[str, object],
    catalog_path: Path,
    archive_path: Path,
    detail_override: str | None = None,
    latency_override: str | None = None,
) -> dict[str, object]:
    """The input identity a frozen run would record, without building an engine.

    Args:
        strategy_name: The strategy's stored name.
        dsl_json: The frozen DSL as JSON.
        run_config: A ``backtest_runs``-shaped config (``symbols``/``parameters`` decoded);
            not mutated.
        catalog_path: The catalog the window clamp reads.
        archive_path: The raw-data archive (funding, intervals, aux data).
        detail_override: A detail timeframe over the config's default.
        latency_override: A latency preset over the config's default.

    Returns:
        :func:`~vibe_quant.validation.runner.validation_input_identity` for these inputs.

    Raises:
        ValidationRunnerError: The same first error ``run()`` raises for these inputs.
    """
    r = _resolve(
        ValidationRunner.detached(catalog_path, archive_path),
        strategy_name=strategy_name,
        dsl_raw=json.loads(dsl_json),
        run_config=_frozen_config(run_config),
        detail_override=detail_override,
        latency_override=latency_override,
    )
    return r.identity


def run_frozen(
    *,
    strategy_name: str,
    dsl_json: str,
    run_config: Mapping[str, object],
    catalog_path: Path,
    archive_path: Path,
    detail_override: str | None = None,
    latency_override: str | None = None,
    mtm_probe: bool = True,
) -> tuple[ValidationResult, dict[str, object]]:
    """Run a frozen DSL on the given market snapshot through :func:`run_variant`'s path.

    Same arguments as :func:`resolve_frozen`, plus ``mtm_probe`` (keep the drawdown's
    mark-to-market paths as ``result.mtm_probe``). No identity check: the caller compares
    the returned identity with its own frozen one.

    Returns:
        ``(result, identity)``; ``identity`` is also ``result.notes["validation_input"]``.

    Raises:
        ValidationRunnerError: The same first error ``run()`` raises for these inputs.
    """
    return _run_resolved(
        ValidationRunner.detached(catalog_path, archive_path),
        strategy_name=strategy_name,
        dsl_raw=json.loads(dsl_json),
        run_config=_frozen_config(run_config),
        archive_path=archive_path,
        detail_override=detail_override,
        latency_override=latency_override,
        mtm_probe=mtm_probe,
    )


def run_variant(
    inp: VariantInput,
    overrides: Mapping[str, object],
    *,
    catalog_path: Path,
    archive_path: Path,
    mtm_probe: bool,
) -> ValidationResult:
    """Run one validation variant of ``inp`` on the given market snapshot.

    Args:
        inp: The parent's snapshotted inputs.
        overrides: Launch parameters merged over a copy of the parent's ``parameters``;
            ``{}`` replays the parent and checks its recorded identity first.
        catalog_path: The catalog this engine reads and writes (a private clone).
        archive_path: The raw-data archive snapshot (funding, intervals, aux data).
        mtm_probe: Keep the drawdown's mark-to-market paths as ``result.mtm_probe``.

    Returns:
        The variant's result; a 0-trade result is returned, not raised.

    Raises:
        VariantInputsMismatch: ``overrides == {}`` and the resolved inputs differ from
            ``inp.parent_identity``.
        ValidationRunnerError: The same first error ``run()`` raises for these inputs.
    """
    run_config: dict[str, object] = json.loads(inp.run_config_json)
    run_config["parameters"] = {**_launch_params(run_config), **overrides}
    runner = ValidationRunner.detached(catalog_path, archive_path)
    parent = inp.parent_identity
    detail_override = parent.get("detail_timeframe")
    latency_override = parent.get("latency_preset")
    r = _resolve(
        runner,
        strategy_name=inp.strategy_name,
        dsl_raw=json.loads(inp.dsl_json),
        run_config=run_config,
        detail_override=detail_override if isinstance(detail_override, str) else None,
        latency_override=latency_override if isinstance(latency_override, str) else None,
    )
    if not overrides:
        differing = tuple(
            sorted(
                k for k in r.identity.keys() | parent.keys() if r.identity.get(k) != parent.get(k)
            )
        )
        if differing:
            raise VariantInputsMismatch(differing)
    return _run(
        runner, r, strategy_name=inp.strategy_name, archive_path=archive_path, mtm_probe=mtm_probe
    )


__all__ = ["VariantInput", "VariantInputsMismatch", "resolve_frozen", "run_frozen", "run_variant"]
