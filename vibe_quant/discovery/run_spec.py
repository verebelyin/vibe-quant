"""Discovery run-spec: the CLI parser and its launch helpers (bead vibe-quant-rabmo.5).

The parser, the window plan it derives and the launch-time bankroll/leverage
checks live here so adapters (the CLI entrypoint and the API launch precheck)
share one definition of a discovery run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vibe_quant.bankroll_grid import (
    BankrollCell,
    BankrollGridError,
    build_cells,
    reject_run_level_with_grid,
)
from vibe_quant.discovery.config_check import (
    WindowPlan,
    parse_cross_window_months,
    plan_windows,
)
from vibe_quant.fees import DEFAULT_TIER
from vibe_quant.liquidation_settings import (
    DEFAULT_LEVERAGE,
    LIQUIDATION_MODES,
    MISSING,
    LiquidationConfigError,
    LiquidationSettings,
    resolve_leverage,
)
from vibe_quant.symbol_leverage import SymbolLeverageError, check_leverage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.sizing_settings import SizingSettings

# Timeframe-aware bootstrap-CI floor defaults (vibe-quant-gds1c):
# - 1m: high-frequency noise widens Sharpe CIs; 1.0 rejects nearly everything.
# - 4h/1d: ~50-180 trades/yr makes the CI lower bound structurally < 1.0 for
#   ANY strategy (Batch 41: lower bounds -0.77..-1.39 with DSR p=0.0000), so
#   the gate degrades to "must exclude negative Sharpe" instead of blocking
#   the entire timeframe. Explicit --bootstrap-min-sharpe always wins.
_BOOTSTRAP_FLOOR_BY_TIMEFRAME: dict[str, float] = {"1m": 0.5, "4h": 0.0, "1d": 0.0}
_BOOTSTRAP_FLOOR_DEFAULT = 1.0


def _default_bootstrap_min_sharpe(timeframe: str) -> float:
    return _BOOTSTRAP_FLOOR_BY_TIMEFRAME.get(timeframe, _BOOTSTRAP_FLOOR_DEFAULT)


# Default TRAIN fraction: every discovery gets a 20% holdout that is used once,
# as the final pass/fail gate (vibe-quant-e70tl.5). Pass --train-test-split 0
# to opt out explicitly.
DEFAULT_TRAIN_TEST_SPLIT = 0.8


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for discovery jobs."""
    parser = argparse.ArgumentParser(
        prog="vibe-quant discovery",
        description="Run genetic strategy discovery",
    )
    parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-generations", type=int, default=15)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--convergence-generations", type=int, default=10)
    parser.add_argument(
        "--max-workers", type=int, default=0, help="Parallel workers (0=auto, -1=sequential)"
    )
    parser.add_argument("--symbols", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="4h")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2026-02-24")
    parser.add_argument(
        "--indicator-pool",
        type=str,
        default=None,
        help=(
            "Comma-separated indicator names to use (default: all). An MA name "
            "(KAMA/VIDYA/FRAMA) enables that MA's price-vs-MA genes; an explicit "
            "pool without MA names draws none."
        ),
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="Direction constraint: long, short, both, or omit for random",
    )
    parser.add_argument(
        "--eval-windows",
        type=int,
        default=3,
        help="Split the train range into N sub-windows; fitness is the WORST "
        "window (min Sharpe, min return, max drawdown, min PF; trades summed for "
        "the min-trades gate) and every window needs >= max(1, min_trades // (2N)) "
        "trades. Forces regime-robust strategies (default: 3; pass 1 for a "
        "single window).",
    )
    parser.add_argument(
        "--evaluator",
        choices=["nt", "fast"],
        default="nt",
        help="GA fitness evaluator: 'nt' = NautilusTrader screening (default); 'fast' = "
        "the fast tier (1m single-symbol replay, exact vs NT screening; other genomes "
        "run on NT), with the top-K re-run on NT before every gate.",
    )
    parser.add_argument(
        "--symbol-agg",
        choices=["portfolio", "worst"],
        default="portfolio",
        help="Multi-symbol scoring: 'portfolio' = one shared-account backtest "
        "over all symbols (default); 'worst' = each symbol backtested alone and "
        "the genome scored by its worst symbol (min Sharpe/return/PF, max DD, "
        "MIN trades per symbol) so no single symbol can carry the rest.",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=DEFAULT_TRAIN_TEST_SPLIT,
        help="TRAIN fraction of the date range (default: 0.8 = last 20%% is a "
        "holdout used once as the final pass/fail gate; 0 disables the holdout).",
    )
    parser.add_argument(
        "--cross-window-months",
        type=str,
        default=None,
        help="Comma-separated month offsets for cross-window validation (e.g. '1,2'). "
        "Re-runs top strategies on sub-windows of the TRAIN range starting N months "
        "later; every shifted window must pass (the in-sample window never counts).",
    )
    parser.add_argument(
        "--holdout-min-sharpe",
        type=float,
        default=0.0,
        help="Holdout gate: holdout Sharpe must exceed this (default 0.0).",
    )
    parser.add_argument(
        "--holdout-min-trades",
        type=int,
        default=None,
        help="Holdout gate trade floor (default: half the train-gate trade rate, "
        "max(1, min_trades * holdout_days / (2 * train_days))).",
    )
    parser.add_argument(
        "--cross-window-min-sharpe",
        type=float,
        default=0.5,
        help="Min Sharpe on each shifted window to count as a pass (default: 0.5)",
    )
    parser.add_argument(
        "--seed-from-run",
        type=int,
        default=None,
        help="Seed initial population with top chromosomes from a prior discovery run ID",
    )
    parser.add_argument(
        "--wfa-oos-step-days",
        type=int,
        default=0,
        help="Rolling-window stability check step in days (0=disabled). Tiles the "
        "TRAIN range (never the holdout) with N-day windows.",
    )
    parser.add_argument(
        "--wfa-min-consistency",
        type=float,
        default=0.75,
        help="Min fraction of profitable rolling windows (default: 0.75 = 3/4)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run. >1 enables multi-seed ensemble: "
        "runs GA N times, ranks by median Sharpe (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible GA runs. Default: random (recorded in "
        "notes). With --num-seeds >1 it is the base: seed i = base + i*7919.",
    )
    parser.add_argument(
        "--bootstrap-min-sharpe",
        type=float,
        default=None,
        help="Bootstrap CI lower bound threshold. Candidates whose bootstrap "
        "CI lower bound falls below this t-stat (mean/std*sqrt(n), not an "
        "annualized Sharpe) are rejected. Timeframe-aware "
        "default: 0.5 for 1m (wider CIs on high-freq noise), 0.0 for 4h/1d "
        "(low trade counts make the lower bound structurally < 1.0), "
        "1.0 otherwise.",
    )
    parser.add_argument(
        "--bootstrap-ci-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap t-stat CI (default: 0.95).",
    )
    parser.add_argument(
        "--no-bootstrap-ci",
        dest="require_bootstrap_ci",
        action="store_false",
        default=True,
        help="Disable the bootstrap t-stat CI hard guardrail.",
    )
    parser.add_argument(
        "--no-dsr",
        dest="require_dsr",
        action="store_false",
        default=True,
        help="Disable the Deflated Sharpe Ratio soft guardrail.",
    )
    parser.add_argument(
        "--dsr-rule",
        choices=["emax", "q95"],
        default="q95",
        help="DSR critical value: q95 (default, quantile-matched Monte-Carlo "
        "spread; design-dsr-q95) or emax (legacy E[maxZ] + 1.6449, reproduces "
        "earlier runs).",
    )
    parser.add_argument(
        "--no-buy-hold-gate",
        dest="require_buy_hold",
        action="store_false",
        default=True,
        help="Disable the fail-closed buy-and-hold champion gate.",
    )
    parser.add_argument(
        "--fee-tier",
        type=str,
        default=DEFAULT_TIER,
        help="Binance USDT-M VIP fee tier for every screening backtest of this "
        "run (default: VIP0).",
    )
    parser.add_argument(
        "--bnb-discount",
        action="store_true",
        help="Charge the BNB-discounted fee rate (x0.9: VIP0 maker 0.00018 / "
        "taker 0.00045) instead of the tier rate.",
    )
    parser.add_argument(
        "--liquidation-mode",
        choices=list(LIQUIDATION_MODES),
        default="off",
        help="Liquidation mode for every screening backtest of this run "
        "(off, isolated or cross; default: off). On, the venue runs at "
        "--leverage and the liquidation counts are reported "
        "(report-only: no score term, no gate).",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=None,
        help=f"Venue leverage for every screening backtest of this run, in every "
        f"mode (a whole number in [1, 125]; default: {DEFAULT_LEVERAGE}). It "
        f"changes results only while liquidation is on.",
    )
    parser.add_argument(
        "--liquidation-leverage",
        type=float,
        default=None,
        help="Deprecated alias for --leverage (given both, they must agree).",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=None,
        help="Exposure cap for every screening backtest of this run, in equity "
        "fractions (a number in (0, 125]; default: 0.5). Only a non-default "
        "value is recorded in notes.sizing (the API launch also stores it in the "
        "run parameters).",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=None,
        help="Risk budget per trade for every screening backtest of this run, in "
        "equity fractions (more than 0 and at most 0.5; default: 0.02). Only a "
        "non-default value is recorded in notes.sizing (the API launch also stores "
        "it in the run parameters).",
    )
    parser.add_argument(
        "--max-margin-pct",
        type=float,
        default=None,
        help="Margin limit for every screening backtest of this run: the margin in "
        "use across open positions plus pending entries may not exceed this "
        "fraction of equity (a number in (0, 1]; absent = OFF). Only a set value "
        "is recorded in notes.sizing (the API launch also stores it in the run "
        "parameters).",
    )
    parser.add_argument(
        "--bankroll-modes",
        type=str,
        default=None,
        help="Bankroll grid (give all three --bankroll-* flags): comma-separated margin modes, "
        "isolated and/or cross. Discovery then searches every mode x leverage x cushion cell "
        "(at most 12) and each champion records its cell. Excludes --liquidation-mode, "
        "--leverage and --max-margin-pct.",
    )
    parser.add_argument(
        "--bankroll-leverages",
        type=str,
        default=None,
        help="Bankroll grid: comma-separated leverages, whole numbers from 1 to 125.",
    )
    parser.add_argument(
        "--bankroll-max-margin-pcts",
        type=str,
        default=None,
        help="Bankroll grid: comma-separated margin limits (numbers in (0, 1]).",
    )
    parser.add_argument(
        "--immigrant-fraction",
        type=float,
        default=0.15,
        help=(
            "Fraction of population replaced when Shannon entropy drops below "
            "--entropy-threshold. 0 disables random-immigrant injection."
        ),
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.4,
        help="Population entropy below this triggers random-immigrant injection.",
    )
    parser.add_argument(
        "--no-crowding",
        dest="use_crowding",
        action="store_false",
        default=True,
        help="Disable deterministic crowding selection (falls back to classic tournament).",
    )
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Synthetic mock backtests (no NT). Results are flagged mock=true. "
        "Without --mock, missing catalog data is an error (never a silent mock).",
    )
    return parser


def plan_from_args(args: argparse.Namespace) -> WindowPlan:
    """Derive the train/holdout/eval/cross/WFA windows from parsed CLI args.

    Shared by ``main`` and the API launch precheck so a bad window config
    fails in seconds (CLI: run marked failed; API: 422 before a run row).

    Raises:
        DiscoveryConfigError: The window config cannot produce valid windows.
    """
    return plan_windows(
        args.start_date,
        args.end_date,
        train_test_split=args.train_test_split,
        eval_windows=args.eval_windows,
        cross_window_months=parse_cross_window_months(args.cross_window_months or ""),
        wfa_oos_step_days=args.wfa_oos_step_days,
    )


_UNSET = object()

# One spec field per build_parser() dest EXCEPT run_id, db and mock (process
# settings the CLI keeps), named exactly as the dest.
_SPEC_DESTS: tuple[str, ...] = (
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
    "indicator_pool",
    "direction",
    "eval_windows",
    "evaluator",
    "symbol_agg",
    "train_test_split",
    "cross_window_months",
    "holdout_min_sharpe",
    "holdout_min_trades",
    "cross_window_min_sharpe",
    "seed_from_run",
    "wfa_oos_step_days",
    "wfa_min_consistency",
    "num_seeds",
    "seed",
    "bootstrap_min_sharpe",
    "bootstrap_ci_level",
    "require_bootstrap_ci",
    "require_dsr",
    "dsr_rule",
    "require_buy_hold",
    "fee_tier",
    "bnb_discount",
    "liquidation_mode",
    "leverage",
    "liquidation_leverage",
    "max_position_pct",
    "risk_per_trade",
    "max_margin_pct",
    "bankroll_modes",
    "bankroll_leverages",
    "bankroll_max_margin_pcts",
    "immigrant_fraction",
    "entropy_threshold",
    "use_crowding",
)

# Canonical argv order (design 2.2.1): the router's order first, then the
# CLI-only flags in parser order. ``cross_window_months`` is emitted in the
# "--flag=value" form: a leading negative month ("-1,2") would otherwise parse
# as a flag.
_ARGV_FIELDS: tuple[tuple[str, str], ...] = (
    ("population_size", "--population-size"),
    ("max_generations", "--max-generations"),
    ("mutation_rate", "--mutation-rate"),
    ("crossover_rate", "--crossover-rate"),
    ("elite_count", "--elite-count"),
    ("tournament_size", "--tournament-size"),
    ("convergence_generations", "--convergence-generations"),
    ("max_workers", "--max-workers"),
    ("symbols", "--symbols"),
    ("timeframe", "--timeframe"),
    ("start_date", "--start-date"),
    ("end_date", "--end-date"),
    ("indicator_pool", "--indicator-pool"),
    ("direction", "--direction"),
    ("eval_windows", "--eval-windows"),
    ("symbol_agg", "--symbol-agg"),
    ("train_test_split", "--train-test-split"),
    ("cross_window_months", "--cross-window-months"),
    ("cross_window_min_sharpe", "--cross-window-min-sharpe"),
    ("num_seeds", "--num-seeds"),
    ("wfa_oos_step_days", "--wfa-oos-step-days"),
    ("wfa_min_consistency", "--wfa-min-consistency"),
    ("immigrant_fraction", "--immigrant-fraction"),
    ("entropy_threshold", "--entropy-threshold"),
    ("use_crowding", "--no-crowding"),
    ("seed_from_run", "--seed-from-run"),
    ("seed", "--seed"),
    ("require_bootstrap_ci", "--no-bootstrap-ci"),
    ("bootstrap_min_sharpe", "--bootstrap-min-sharpe"),
    ("bnb_discount", "--bnb-discount"),
    ("evaluator", "--evaluator"),
    ("liquidation_mode", "--liquidation-mode"),
    ("leverage", "--leverage"),
    ("max_position_pct", "--max-position-pct"),
    ("risk_per_trade", "--risk-per-trade"),
    ("max_margin_pct", "--max-margin-pct"),
    ("bankroll_modes", "--bankroll-modes"),
    ("bankroll_leverages", "--bankroll-leverages"),
    ("bankroll_max_margin_pcts", "--bankroll-max-margin-pcts"),
    ("holdout_min_sharpe", "--holdout-min-sharpe"),
    ("holdout_min_trades", "--holdout-min-trades"),
    ("bootstrap_ci_level", "--bootstrap-ci-level"),
    ("require_dsr", "--no-dsr"),
    ("dsr_rule", "--dsr-rule"),
    ("require_buy_hold", "--no-buy-hold-gate"),
    ("fee_tier", "--fee-tier"),
    ("liquidation_leverage", "--liquidation-leverage"),
)

# store_true/store_false dests: the bare flag carries the value.
_BARE_FLAGS: frozenset[str] = frozenset(
    {"use_crowding", "require_bootstrap_ci", "bnb_discount", "require_dsr", "require_buy_hold"}
)


@dataclass(frozen=True)
class DiscoveryRunSpec:
    """One discovery run as parsed from CLI argv.

    Every field is one ``build_parser()`` dest, typed as argparse produces it,
    with no default: the parser owns the defaults. ``given`` records which
    dests appeared on the argv, so ``to_argv`` re-emits exactly those in
    canonical order.
    """

    population_size: int
    max_generations: int
    mutation_rate: float
    crossover_rate: float
    elite_count: int
    tournament_size: int
    convergence_generations: int
    max_workers: int
    symbols: str
    timeframe: str
    start_date: str
    end_date: str
    indicator_pool: str | None
    direction: str | None
    eval_windows: int
    evaluator: str
    symbol_agg: str
    train_test_split: float
    cross_window_months: str | None
    holdout_min_sharpe: float
    holdout_min_trades: int | None
    cross_window_min_sharpe: float
    seed_from_run: int | None
    wfa_oos_step_days: int
    wfa_min_consistency: float
    num_seeds: int
    seed: int | None
    bootstrap_min_sharpe: float | None
    bootstrap_ci_level: float
    require_bootstrap_ci: bool
    require_dsr: bool
    dsr_rule: str
    require_buy_hold: bool
    fee_tier: str
    bnb_discount: bool
    liquidation_mode: str
    leverage: float | None
    liquidation_leverage: float | None
    max_position_pct: float | None
    risk_per_trade: float | None
    max_margin_pct: float | None
    bankroll_modes: str | None
    bankroll_leverages: str | None
    bankroll_max_margin_pcts: str | None
    immigrant_fraction: float
    entropy_threshold: float
    use_crowding: bool
    given: frozenset[str]

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, given: frozenset[str] = frozenset()
    ) -> DiscoveryRunSpec:
        """Read every spec dest from a parsed namespace."""
        values: dict[str, Any] = {dest: getattr(args, dest) for dest in _SPEC_DESTS}
        return cls(**values, given=given)

    @classmethod
    def from_argv(cls, argv: Sequence[str]) -> DiscoveryRunSpec:
        """Parse argv (without ``--run-id``) into a spec.

        The namespace is pre-filled with ``_UNSET`` for every spec dest, so a
        dest is ``given`` exactly when the argv set it; every other dest takes
        the parser default. Bad argv keeps argparse's own SystemExit(2).
        """
        parser = build_parser()
        namespace = argparse.Namespace(**dict.fromkeys(_SPEC_DESTS, _UNSET))
        parsed = parser.parse_args(["--run-id", "0", *argv], namespace=namespace)
        given = frozenset(dest for dest in _SPEC_DESTS if getattr(parsed, dest) is not _UNSET)
        for dest in _SPEC_DESTS:
            if getattr(parsed, dest) is _UNSET:
                setattr(parsed, dest, parser.get_default(dest))
        return cls.from_args(parsed, given)

    def to_argv(self) -> list[str]:
        """The CLI argv for the ``given`` dests, in canonical order."""
        argv: list[str] = []
        for dest, flag in _ARGV_FIELDS:
            if dest not in self.given:
                continue
            if dest in _BARE_FLAGS:
                argv.append(flag)
            elif dest == "cross_window_months":
                argv.append(f"{flag}={getattr(self, dest)}")
            else:
                argv.extend([flag, str(getattr(self, dest))])
        return argv

    def plan(self) -> WindowPlan:
        """The windows this run will use.

        Raises:
            DiscoveryConfigError: The window config cannot produce valid windows.
        """
        return plan_windows(
            self.start_date,
            self.end_date,
            train_test_split=self.train_test_split,
            eval_windows=self.eval_windows,
            cross_window_months=parse_cross_window_months(self.cross_window_months or ""),
            wfa_oos_step_days=self.wfa_oos_step_days,
        )


def _grid_flags_given(args: argparse.Namespace) -> bool:
    """Whether any ``--bankroll-*`` flag was passed (all three are then required)."""
    flags = (args.bankroll_modes, args.bankroll_leverages, args.bankroll_max_margin_pcts)
    if all(f is None for f in flags):
        return False
    if any(f is None for f in flags):
        msg = (
            "give all three of --bankroll-modes, --bankroll-leverages and "
            "--bankroll-max-margin-pcts together"
        )
        raise BankrollGridError(msg)
    reject_run_level_with_grid(
        liquidation_mode=args.liquidation_mode,
        leverage=args.leverage,
        liquidation_leverage=args.liquidation_leverage,
        max_margin_pct=args.max_margin_pct,
    )
    return True


def _grid_cells_from_args(args: argparse.Namespace, sizing: SizingSettings) -> list[BankrollCell]:
    """The cells of ``--bankroll-*`` (validated against the run's exposure cap)."""

    def _items(text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]

    def _numbers(name: str, text: str) -> list[object]:
        out: list[object] = []
        for item in _items(text):
            try:
                out.append(float(item))
            except ValueError as exc:
                msg = f"{name} must be comma-separated numbers, got {item!r}"
                raise BankrollGridError(msg) from exc
        return out

    return build_cells(
        _items(args.bankroll_modes),
        _numbers("--bankroll-leverages", args.bankroll_leverages),
        _numbers("--bankroll-max-margin-pcts", args.bankroll_max_margin_pcts),
        sizing,
    )


def check_launch_leverage(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a launch at a leverage above a symbol's Binance maximum (L5/L6).

    Runs right after parsing: ``--symbols`` against the resolved ``--leverage``
    (10 by default) and against every ``--bankroll-leverages`` value. The first
    refusal ends the launch through ``parser.error`` (usage + exit code 2) before
    any run row is read.

    A leverage the run-level settings already refuse - disagreeing aliases, or a
    value outside [1, 125] - is left to them (a failed run, exit 1), so this check
    only adds the venue cap.
    """
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    try:
        resolved = resolve_leverage(
            MISSING if args.leverage is None else args.leverage,
            MISSING if args.liquidation_leverage is None else args.liquidation_leverage,
        )
    except LiquidationConfigError:
        return
    if resolved is MISSING:
        resolved = DEFAULT_LEVERAGE
    leverages: list[object] = [resolved]
    for token in (args.bankroll_leverages or "").split(","):
        try:
            leverages.append(float(token))
        except ValueError:
            # A malformed token is the grid's own parse error (BankrollGridError -> exit 2).
            continue
    try:
        for leverage in leverages:
            try:
                checked = LiquidationSettings(leverage=cast("float", leverage)).leverage
            except LiquidationConfigError:
                continue
            check_leverage(symbols, checked)
    except SymbolLeverageError as exc:
        parser.error(str(exc))
