"""Tests for screening CLI subcommands (run/list/status)."""

from __future__ import annotations

from vibe_quant.screening.__main__ import build_parser


class TestScreeningCLIParser:
    """Tests for screening CLI argument parser."""

    def test_parser_has_subcommands(self) -> None:
        """Parser should have run/list/status subcommands."""
        parser = build_parser()
        # Parse help output to verify subcommands exist
        # argparse stores subparsers in _subparsers
        subparsers_actions = [
            action for action in parser._subparsers._actions if hasattr(action, "_parser_class")
        ]
        assert len(subparsers_actions) > 0
        choices = subparsers_actions[0].choices
        assert "run" in choices
        assert "list" in choices
        assert "status" in choices

    def test_run_subcommand_requires_run_id(self) -> None:
        """run subcommand should require --run-id."""
        parser = build_parser()
        args = parser.parse_args(["run", "--run-id", "42"])
        assert args.run_id == 42
        assert args.subcommand == "run"

    def test_run_subcommand_accepts_optional_args(self) -> None:
        """run subcommand should accept optional strategy/symbols/timeframe."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--run-id",
                "42",
                "--strategy-id",
                "5",
                "--symbols",
                "BTCUSDT-PERP",
                "ETHUSDT-PERP",
                "--timeframe",
                "1h",
            ]
        )
        assert args.run_id == 42
        assert args.strategy_id == 5
        assert args.symbols == ["BTCUSDT-PERP", "ETHUSDT-PERP"]
        assert args.timeframe == "1h"

    def test_list_subcommand_defaults(self) -> None:
        """list subcommand should have default limit."""
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.subcommand == "list"
        assert args.limit == 20
        assert args.status is None

    def test_list_subcommand_with_filters(self) -> None:
        """list subcommand should accept --limit and --status."""
        parser = build_parser()
        args = parser.parse_args(["list", "--limit", "10", "--status", "completed"])
        assert args.limit == 10
        assert args.status == "completed"

    def test_status_subcommand_requires_run_id(self) -> None:
        """status subcommand should require --run-id."""
        parser = build_parser()
        args = parser.parse_args(["status", "--run-id", "42"])
        assert args.run_id == 42
        assert args.subcommand == "status"

    def test_no_subcommand_defaults_to_none(self) -> None:
        """No subcommand should set subcommand=None."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.subcommand is None

    def test_all_subcommands_have_func(self) -> None:
        """All subcommands should have func attribute."""
        parser = build_parser()
        for subcmd in ["run --run-id 1", "list", "status --run-id 1"]:
            args = parser.parse_args(subcmd.split())
            assert hasattr(args, "func"), f"Subcommand {subcmd} missing func"
