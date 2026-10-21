"""DiscoveryRunSpec characterization tests (bead vibe-quant-rabmo.5, task T6).

The T1 goldens in ``tests/fixtures/discovery/run_spec_golden.json`` hold every
argc form discovery launches with: 16 API rows (the router argv) and 10
CLI-only rows (flags the API schema does not model). T6 pins the spec against
them before any caller moves onto it:

- R12: ``from_argv(to_argv(spec)) == spec`` for every golden argv, and
  ``given`` is exactly the set of dests that appear on the argv.
- R13: a repeated flag keeps its last value and is emitted once.
- R20: one spec field per ``build_parser()`` dest, minus the process settings
  the CLI keeps (``run_id``, ``db``, ``mock``) and ``help``.
- ``plan()`` is exactly what the shared window check (``plan_windows``) derives.
- R16: ``to_config`` builds the exact ``DiscoveryConfig`` each golden CLI row ran
  on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.bankroll_grid import BankrollGridError
from vibe_quant.discovery.config_check import DiscoveryConfigError
from vibe_quant.discovery.run_spec import (
    DiscoveryRunSpec,
    build_parser,
)
from vibe_quant.liquidation_settings import LiquidationConfigError

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "discovery" / "run_spec_golden.json"
)

# Dests the spec deliberately does not own: process settings the CLI keeps.
PROCESS_DESTS = frozenset({"help", "run_id", "db", "mock"})


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


def _golden_rows() -> list[tuple[str, list[str]]]:
    golden = _golden()
    rows = [(f"api/{name}", row["argv"]) for name, row in golden["api"].items()]
    rows += [(f"cli/{name}", row["argv"]) for name, row in golden["cli"].items()]
    return rows


# --- R12: round trip -------------------------------------------------------


def test_round_trip_every_golden_row() -> None:
    """R12: every golden argv survives spec -> argv -> spec unchanged."""
    rows = _golden_rows()
    assert len(rows) == 26
    for name, argv in rows:
        spec = DiscoveryRunSpec.from_argv(argv)
        assert DiscoveryRunSpec.from_argv(spec.to_argv()) == spec, name


def test_to_argv_matches_golden() -> None:
    """T6 accept: to_argv reproduces the golden argv for every T1 row.

    One CLI row is written ``--liquidation-leverage 5``; parsing yields the
    float 5.0, so the canonical form re-emits ``5.0`` (same spec, same plan).
    """
    golden = _golden()
    for name, row in golden["api"].items():
        assert DiscoveryRunSpec.from_argv(row["argv"]).to_argv() == row["argv"], name
    normalized = {"liquidation_leverage_5": ["--liquidation-leverage", "5.0"]}
    for name, row in golden["cli"].items():
        expected = normalized.get(name, row["argv"])
        assert DiscoveryRunSpec.from_argv(row["argv"]).to_argv() == expected, name
    assert DiscoveryRunSpec.from_argv(["--liquidation-leverage", "5"]) == (
        DiscoveryRunSpec.from_argv(["--liquidation-leverage", "5.0"])
    )


DEFAULT_GIVEN = frozenset(
    {
        "population_size",
        "max_generations",
        "mutation_rate",
        "crossover_rate",
        "elite_count",
        "tournament_size",
        "convergence_generations",
        "max_workers",
        "symbols",
        "timeframe",
        "start_date",
        "end_date",
        "eval_windows",
        "train_test_split",
    }
)

MISC_GIVEN = DEFAULT_GIVEN | {
    "indicator_pool",
    "direction",
    "symbol_agg",
    "num_seeds",
    "use_crowding",
    "seed_from_run",
    "seed",
    "require_bootstrap_ci",
    "bootstrap_min_sharpe",
    "bnb_discount",
    "evaluator",
}


def test_given_is_dests_on_argv() -> None:
    """R12: ``given`` is the dest set that appears on argv, and nothing else."""
    golden = _golden()
    default = DiscoveryRunSpec.from_argv(golden["api"]["default"]["argv"])
    assert default.given == DEFAULT_GIVEN
    misc = DiscoveryRunSpec.from_argv(golden["api"]["misc"]["argv"])
    assert misc.given == MISC_GIVEN
    assert DiscoveryRunSpec.from_argv(["--no-dsr"]).given == frozenset({"require_dsr"})
    assert DiscoveryRunSpec.from_argv(["--symbols", ""]).given == frozenset({"symbols"})
    for name, argv in _golden_rows():
        spec = DiscoveryRunSpec.from_argv(argv)
        assert not (spec.given & PROCESS_DESTS), name
        for dest in spec.given:
            assert hasattr(spec, dest), (name, dest)
            assert getattr(spec, dest) is not None, (name, dest)


def test_unset_dests_take_parser_defaults() -> None:
    """Only argv dests are given; every other dest is the parser's default."""
    parser = build_parser()
    spec = DiscoveryRunSpec.from_argv(["--no-dsr"])
    assert spec.require_dsr is False
    assert spec.population_size == parser.get_default("population_size")
    assert spec.max_workers == 0
    assert spec.end_date == "2026-02-24"
    assert spec.indicator_pool is None
    assert spec.direction is None
    assert spec.cross_window_months is None
    assert spec.seed is None
    assert spec.holdout_min_trades is None
    assert spec.leverage is None
    assert spec.require_bootstrap_ci is True
    assert spec.use_crowding is True
    assert spec.evaluator == "nt"
    assert spec.to_argv() == ["--no-dsr"]


# --- R13: duplicate flags --------------------------------------------------


def test_duplicate_flag_last_wins() -> None:
    """R13: the last ``--population-size`` wins and is emitted once."""
    spec = DiscoveryRunSpec.from_argv(["--population-size", "5", "--population-size", "9"])
    assert spec.population_size == 9
    assert spec.given == frozenset({"population_size"})
    assert spec.to_argv() == ["--population-size", "9"]
    assert DiscoveryRunSpec.from_argv(spec.to_argv()) == spec


# --- R20: one field per parser dest ---------------------------------------


def test_spec_fields_match_parser_dests() -> None:
    """R20: fields minus ``given`` == parser dests minus process settings."""
    parser_dests = {action.dest for action in build_parser()._actions} - PROCESS_DESTS
    assert {f.name for f in fields(DiscoveryRunSpec)} - {"given"} == parser_dests
    assert len(parser_dests) == 47


# --- plan ------------------------------------------------------------------


def _reference_plan(argv: list[str]) -> Any:
    """The window plan argv derives, straight from ``config_check`` (reference)."""
    from vibe_quant.discovery.config_check import parse_cross_window_months, plan_windows

    args = build_parser().parse_args(["--run-id", "0", *argv])
    return plan_windows(
        args.start_date,
        args.end_date,
        train_test_split=args.train_test_split,
        eval_windows=args.eval_windows,
        cross_window_months=parse_cross_window_months(args.cross_window_months or ""),
        wfa_oos_step_days=args.wfa_oos_step_days,
    )


def test_plan_matches_reference_windows_every_golden_row() -> None:
    """``plan()`` returns exactly what the shared window check derives, every row.

    The ``cross_months_negative`` row is the API's 422 row, so both raise the
    same ``DiscoveryConfigError`` text (pinned below).
    """
    raised: list[str] = []
    for name, argv in _golden_rows():
        spec = DiscoveryRunSpec.from_argv(argv)
        try:
            expected = _reference_plan(argv)
        except DiscoveryConfigError as exc:
            with pytest.raises(DiscoveryConfigError) as spec_exc:
                spec.plan()
            assert str(spec_exc.value) == str(exc), name
            raised.append(name)
            continue
        assert spec.plan() == expected, name
    assert raised == ["api/cross_months_negative"]


def test_plan_rejects_invalid_windows_like_the_reference() -> None:
    """A bad window config raises the same ``DiscoveryConfigError`` text."""
    for argv in (
        ["--cross-window-months", "0"],
        ["--cross-window-months", "abc"],
        ["--wfa-oos-step-days", "-1"],
    ):
        spec = DiscoveryRunSpec.from_argv(argv)
        with pytest.raises(DiscoveryConfigError) as spec_exc:
            spec.plan()
        with pytest.raises(DiscoveryConfigError) as args_exc:
            _reference_plan(argv)
        assert str(spec_exc.value) == str(args_exc.value)


# --- from_args ------------------------------------------------------------


def test_from_args_reads_every_dest_and_keeps_given() -> None:
    """``from_args`` reads the dests and forwards the caller's ``given``."""
    parser = build_parser()
    args = parser.parse_args(["--run-id", "42", "--population-size", "5"])
    spec = DiscoveryRunSpec.from_args(args)
    assert spec.population_size == 5
    assert spec.given == frozenset()
    assert spec.to_argv() == []
    given = DiscoveryRunSpec.from_args(args, frozenset({"population_size", "symbols"}))
    assert given.to_argv() == ["--population-size", "5", "--symbols", "BTCUSDT"]
    assert DiscoveryRunSpec.from_argv(["--population-size", "5"]) == DiscoveryRunSpec.from_args(
        parser.parse_args(["--run-id", "0", "--population-size", "5"]),
        frozenset({"population_size"}),
    )


# --- to_config (R16) ------------------------------------------------------


def test_to_config_matches_golden() -> None:
    """R16: ``to_config`` is the exact ``DiscoveryConfig`` every CLI golden row ran on.

    ``prior_trials_fn`` is the only non-data field (``None`` in a mock run), and
    the parser's timeframe rule sets ``min_trades`` to 100 for sub-5m through
    ``DiscoveryConfig.__post_init__``.
    """
    golden = _golden()
    for name, row in golden["cli"].items():
        spec = DiscoveryRunSpec.from_argv(row["argv"])
        captured = asdict(
            spec.to_config(
                spec.plan(),
                symbols=spec.symbol_list() or row["run_symbols"],
                prior_trials_fn=None,
            )
        )
        captured.pop("prior_trials_fn")
        assert captured == row["config"], name


# --- settings helpers ------------------------------------------------------


def test_resolved_bootstrap_min_sharpe_uses_the_timeframe_floor() -> None:
    """The floor table: 1m 0.5, 4h/1d 0.0, anything else 1.0; an explicit value wins."""
    for argv, expected in (
        ([], 0.0),  # 4h
        (["--timeframe", "1m"], 0.5),
        (["--timeframe", "1d"], 0.0),
        (["--timeframe", "1h"], 1.0),
        (["--timeframe", "1m", "--bootstrap-min-sharpe", "0.2"], 0.2),
        (["--timeframe", "1h", "--bootstrap-min-sharpe", "0.0"], 0.0),
    ):
        assert DiscoveryRunSpec.from_argv(argv).resolved_bootstrap_min_sharpe() == expected, argv


def test_list_and_count_helpers() -> None:
    """Two unrelated argv rows: the parsed lists/counts main reads, literally."""
    spec = DiscoveryRunSpec.from_argv([
        "--symbols", " BTCUSDT , ETHUSDT ",
        "--indicator-pool", " RSI , KAMA ",
        "--eval-windows", "0",
        "--max-workers", "-1",
        "--cross-window-months=-1,3",
    ])
    assert spec.symbol_list() == ["BTCUSDT", "ETHUSDT"]
    assert spec.indicator_pool_list() == ["RSI", "KAMA"]
    assert spec.eval_window_count() == 1
    assert spec.worker_count() is None
    assert spec.cross_window_month_list() == [-1, 3]

    other = DiscoveryRunSpec.from_argv([
        "--symbols", "",
        "--eval-windows", "3",
        "--max-workers", "0",
    ])
    assert other.symbol_list() == []
    assert other.indicator_pool_list() is None
    assert other.eval_window_count() == 3
    assert other.worker_count() == 0
    assert other.cross_window_month_list() == []


def test_fee_schedule_helper() -> None:
    """The fee schedule main builds, and the tier error main lets escape."""
    default = DiscoveryRunSpec.from_argv([]).fee_schedule()
    assert default.to_notes() == {"fee_tier": "VIP0", "bnb_discount": False}
    other = DiscoveryRunSpec.from_argv(["--fee-tier", "VIP0", "--bnb-discount"]).fee_schedule()
    assert other.to_notes() == {"fee_tier": "VIP0", "bnb_discount": True}

    with pytest.raises(ValueError) as exc:
        DiscoveryRunSpec.from_argv(["--fee-tier", "VIP99"]).fee_schedule()
    assert str(exc.value) == "unknown fee tier 'VIP99'; known tiers: ['VIP0']"


def test_liquidation_settings_helper() -> None:
    """Mode + resolved leverage, and the alias conflict main fails the run with."""
    off = DiscoveryRunSpec.from_argv(["--leverage", "5"]).liquidation_settings()
    assert (off.mode, off.leverage) == ("off", 5.0)
    cross = DiscoveryRunSpec.from_argv(
        ["--liquidation-mode", "cross", "--leverage", "20"]
    ).liquidation_settings()
    assert (cross.mode, cross.leverage) == ("cross", 20.0)
    alias = DiscoveryRunSpec.from_argv(["--liquidation-leverage", "7"]).liquidation_settings()
    assert (alias.mode, alias.leverage) == ("off", 7.0)

    with pytest.raises(LiquidationConfigError) as exc:
        DiscoveryRunSpec.from_argv(
            ["--leverage", "5", "--liquidation-leverage", "7"]
        ).liquidation_settings()
    assert str(exc.value) == "conflicting leverage 5.0 and liquidation_leverage 7.0"


def test_sizing_settings_helper() -> None:
    """Only given sizing keys reach the settings; every default is unchanged."""
    default = DiscoveryRunSpec.from_argv([]).sizing_settings()
    assert default.max_position_pct == 0.5
    assert default.risk_per_trade == 0.02
    assert default.max_margin_pct is None
    assert default.to_params() == {}

    other = DiscoveryRunSpec.from_argv([
        "--max-position-pct", "2",
        "--risk-per-trade", "0.01",
        "--max-margin-pct", "0.3",
    ]).sizing_settings()
    assert other.to_params() == {
        "max_position_pct": 2.0,
        "risk_per_trade": 0.01,
        "max_margin_pct": 0.3,
    }


def test_is_grid_and_bankroll_cells() -> None:
    """R15: a partial grid is refused with today's text; a full one builds every cell."""
    assert DiscoveryRunSpec.from_argv([]).is_grid() is False

    with pytest.raises(BankrollGridError) as exc:
        DiscoveryRunSpec.from_argv(["--bankroll-modes", "isolated"]).is_grid()
    assert str(exc.value) == (
        "give all three of --bankroll-modes, --bankroll-leverages and "
        "--bankroll-max-margin-pcts together"
    )

    with pytest.raises(BankrollGridError) as exc:
        DiscoveryRunSpec.from_argv([
            "--bankroll-modes", "isolated",
            "--bankroll-leverages", "5",
            "--bankroll-max-margin-pcts", "0.5",
            "--leverage", "5",
        ]).is_grid()
    assert str(exc.value).endswith(
        "use either the bankroll grid or --liquidation-mode/--leverage/--max-margin-pct"
    )

    spec = DiscoveryRunSpec.from_argv([
        "--bankroll-modes", "isolated,cross",
        "--bankroll-leverages", "5,10",
        "--bankroll-max-margin-pcts", "0.5",
    ])
    assert spec.is_grid() is True
    cells = spec.bankroll_cells(spec.sizing_settings())
    assert [(c.mode, c.leverage, c.max_margin_pct) for c in cells] == [
        ("isolated", 5.0, 0.5),
        ("isolated", 10.0, 0.5),
        ("cross", 5.0, 0.5),
        ("cross", 10.0, 0.5),
    ]


# --- bad argv --------------------------------------------------------------


@pytest.mark.parametrize("argv", [["--nope"], ["--population-size", "x"]])
def test_bad_argv_exits_two(argv: list[str]) -> None:
    """Bad argv keeps argparse's own SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        DiscoveryRunSpec.from_argv(argv)
    assert exc.value.code == 2
