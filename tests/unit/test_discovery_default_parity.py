"""Default-value parity across the three discovery launch paths.

A discovery run can be launched from the dataclass (research scripts), the CLI
(a job subprocess) or the API (the dashboard). A knob whose default differs
between the paths silently changes what a "default" run does depending on how
it was started, so every shared knob is pinned here: equal by default, or an
explicit, reasoned entry in ``INTENTIONAL``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING

from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest
from vibe_quant.discovery.__main__ import build_parser
from vibe_quant.discovery.pipeline import DiscoveryConfig

#: (config field, CLI dest, API field). ``None`` = that path has no such knob.
SHARED: list[tuple[str | None, str, str | None]] = [
    ("population_size", "population_size", "population"),
    ("max_generations", "max_generations", "generations"),
    ("mutation_rate", "mutation_rate", "mutation_rate"),
    ("crossover_rate", "crossover_rate", "crossover_rate"),
    ("elite_count", "elite_count", "elite_count"),
    ("tournament_size", "tournament_size", "tournament_size"),
    ("convergence_generations", "convergence_generations", "convergence_generations"),
    ("max_workers", "max_workers", "max_workers"),
    ("symbols", "symbols", "symbols"),
    ("timeframe", "timeframe", "timeframes"),
    ("start_date", "start_date", "start_date"),
    ("end_date", "end_date", "end_date"),
    ("indicator_pool", "indicator_pool", "indicator_pool"),
    ("direction", "direction", "direction"),
    ("eval_windows", "eval_windows", "eval_windows"),
    ("symbol_agg", "symbol_agg", "symbol_agg"),
    ("train_test_split", "train_test_split", "train_test_split"),
    ("cross_window_months", "cross_window_months", "cross_window_months"),
    ("cross_window_min_sharpe", "cross_window_min_sharpe", "cross_window_min_sharpe"),
    ("wfa_oos_step_days", "wfa_oos_step_days", "wfa_oos_step_days"),
    ("wfa_min_consistency", "wfa_min_consistency", "wfa_min_consistency"),
    ("holdout_min_sharpe", "holdout_min_sharpe", None),
    ("holdout_min_trades", "holdout_min_trades", None),
    ("require_bootstrap_ci", "require_bootstrap_ci", "no_bootstrap_ci"),
    ("bootstrap_min_sharpe", "bootstrap_min_sharpe", "bootstrap_min_sharpe"),
    ("bootstrap_ci_level", "bootstrap_ci_level", None),
    ("require_dsr", "require_dsr", None),
    ("dsr_rule", "dsr_rule", None),
    ("require_buy_hold", "require_buy_hold", None),
    ("use_crowding", "use_crowding", "crowding_enabled"),
    ("immigrant_fraction", "immigrant_fraction", "immigrant_fraction"),
    ("entropy_threshold", "entropy_threshold", "entropy_threshold"),
    (None, "num_seeds", "num_seeds"),
    (None, "seed", "seed"),
    (None, "seed_from_run", "seed_run_id"),
    (None, "evaluator", "evaluator"),
    (None, "bnb_discount", "bnb_discount"),
    (None, "liquidation_mode", "liquidation_mode"),
    (None, "leverage", "leverage"),
    (None, "liquidation_leverage", "liquidation_leverage"),
    (None, "max_position_pct", "max_position_pct"),
    (None, "risk_per_trade", "risk_per_trade"),
    (None, "max_margin_pct", "max_margin_pct"),
]

#: option -> ((config, CLI, API), why the defaults must differ).
INTENTIONAL: dict[str, tuple[tuple[object, object, object], str]] = {
    "max_workers": ((0, 0, 4), "API caps one run at 4 processes, rejects 0 (a9905cde)"),
    "symbols": (([], ["BTCUSDT"], ["BTCUSDT"]), "config takes the caller's symbols"),
    "start_date": (
        ("", "2024-01-01", None),
        "API None = today - 365 d (router); CLI pins a range; config gets the train slice",
    ),
    "end_date": (("", "2026-02-24", None), "API None = today"),
    "train_test_split": ((0.0, 0.8, 0.8), "config takes the CLI's split"),
    "bootstrap_min_sharpe": ((1.0, None, None), "None = CLI timeframe rule"),
}


def _config_defaults() -> dict[str, object]:
    return {
        f.name: f.default if f.default is not MISSING else f.default_factory()
        for f in dataclasses.fields(DiscoveryConfig)
    }


def _cli_defaults() -> dict[str, object]:
    return vars(build_parser().parse_args(["--run-id", "1"]))


def _api_defaults() -> dict[str, object]:
    return {name: field.default for name, field in DiscoveryLaunchRequest.model_fields.items()}


def _normalise_config(name: str, value: object) -> object:
    if name == "cross_window_months" and value == []:
        return None
    return value


def _normalise_cli(dest: str, value: object) -> object:
    if dest == "symbols" and isinstance(value, str):
        return value.split(",")
    return value


def _normalise_api(name: str, value: object) -> object:
    if name == "timeframes" and isinstance(value, list):
        return value[0] if len(value) == 1 else value
    if name == "no_bootstrap_ci":
        return not value
    return value


def _option(row: tuple[str | None, str, str | None]) -> str:
    config_name, cli_dest, _api_name = row
    return config_name or cli_dest


def _values(
    row: tuple[str | None, str, str | None],
    config: dict[str, object],
    cli: dict[str, object],
    api: dict[str, object],
) -> tuple[object, object | None, object | None]:
    config_name, cli_dest, api_name = row
    config_value = (
        _normalise_config(config_name, config[config_name])
        if config_name is not None
        else None
    )
    api_value = _normalise_api(api_name, api[api_name]) if api_name is not None else None
    return config_value, _normalise_cli(cli_dest, cli[cli_dest]), api_value


def test_shared_discovery_defaults_agree() -> None:
    """Every non-intentional shared knob defaults to the same value on all paths."""
    config = _config_defaults()
    cli = _cli_defaults()
    api = _api_defaults()

    for row in SHARED:
        option = _option(row)
        if option in INTENTIONAL:
            continue
        config_value, cli_value, api_value = _values(row, config, cli, api)
        values: list[object] = []
        if row[0] is not None:
            values.append(config_value)
        values.append(cli_value)
        if row[2] is not None:
            values.append(api_value)
        assert all(v == values[0] for v in values), (
            f"{option}: config={config_value!r} cli={cli_value!r} api={api_value!r}"
        )


def test_intentional_default_differences_are_pinned() -> None:
    """Each known difference keeps its exact values; a fixed difference leaves the list."""
    config = _config_defaults()
    cli = _cli_defaults()
    api = _api_defaults()
    rows = {_option(row): row for row in SHARED}

    for option, (expected, reason) in INTENTIONAL.items():
        actual = _values(rows[option], config, cli, api)
        assert actual == expected, f"{option}: {actual!r} != {expected!r} ({reason})"
        assert not all(v == actual[0] for v in actual), (
            f"{option} is now equal on every path; drop it from INTENTIONAL ({reason})"
        )


def test_every_api_field_is_classified() -> None:
    """A new API field must be added to SHARED (or explicitly exempted here)."""
    api = _api_defaults()
    classified = {row[2] for row in SHARED if row[2] is not None}
    assert set(api) - classified == {"entry_order", "bankroll_grid"}
