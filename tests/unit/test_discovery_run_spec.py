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
- ``plan()`` is exactly what ``plan_from_args`` returns for the same argv.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.discovery.config_check import DiscoveryConfigError
from vibe_quant.discovery.run_spec import (
    DiscoveryRunSpec,
    build_parser,
    plan_from_args,
)

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


# --- plan == plan_from_args ------------------------------------------------


def test_plan_matches_plan_from_args_every_golden_row() -> None:
    """``plan()`` returns exactly what ``plan_from_args`` derives, every row.

    The ``cross_months_negative`` row is the API's 422 row, so both raise the
    same ``DiscoveryConfigError`` text (pinned below).
    """
    parser = build_parser()
    raised: list[str] = []
    for name, argv in _golden_rows():
        spec = DiscoveryRunSpec.from_argv(argv)
        args = parser.parse_args(["--run-id", "0", *argv])
        try:
            expected = plan_from_args(args)
        except DiscoveryConfigError as exc:
            with pytest.raises(DiscoveryConfigError) as spec_exc:
                spec.plan()
            assert str(spec_exc.value) == str(exc), name
            raised.append(name)
            continue
        assert spec.plan() == expected, name
    assert raised == ["api/cross_months_negative"]


def test_plan_rejects_invalid_windows_like_plan_from_args() -> None:
    """A bad window config raises the same ``DiscoveryConfigError`` text."""
    parser = build_parser()
    for argv in (
        ["--cross-window-months", "0"],
        ["--cross-window-months", "abc"],
        ["--wfa-oos-step-days", "-1"],
    ):
        spec = DiscoveryRunSpec.from_argv(argv)
        args = parser.parse_args(["--run-id", "0", *argv])
        with pytest.raises(DiscoveryConfigError) as spec_exc:
            spec.plan()
        with pytest.raises(DiscoveryConfigError) as args_exc:
            plan_from_args(args)
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


# --- bad argv --------------------------------------------------------------


@pytest.mark.parametrize("argv", [["--nope"], ["--population-size", "x"]])
def test_bad_argv_exits_two(argv: list[str]) -> None:
    """Bad argv keeps argparse's own SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        DiscoveryRunSpec.from_argv(argv)
    assert exc.value.code == 2
