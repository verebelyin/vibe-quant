"""Discovery run-spec: the CLI parser and its launch helpers (bead vibe-quant-rabmo.5).

The parser, the window plan it derives and the launch-time bankroll/leverage
checks live here so adapters (the CLI entrypoint and the API launch precheck)
share one definition of a discovery run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from vibe_quant.bankroll_grid import (
    BankrollCell,
    BankrollGridError,
    bankroll_grid_note,
    build_cells,
    reject_run_level_with_grid,
)
from vibe_quant.discovery.config_check import (
    WindowPlan,
    parse_cross_window_months,
    plan_windows,
)
from vibe_quant.fees import DEFAULT_TIER, FeeSchedule
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import (
    DEFAULT_LEVERAGE,
    LIQUIDATION_MODES,
    MISSING,
    LiquidationConfigError,
    LiquidationSettings,
    resolve_leverage,
)
from vibe_quant.sizing_settings import SizingSettings
from vibe_quant.symbol_leverage import SymbolLeverageError, check_leverage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryResult
    from vibe_quant.liquidation_settings import LiquidationMode

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


def _csv_items(text: str) -> list[str]:
    """A comma-separated option split into its non-empty, stripped items."""
    return [item.strip() for item in text.split(",") if item.strip()]


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

    @classmethod
    def from_request(
        cls, body: DiscoveryLaunchRequest, *, start_date: str, end_date: str
    ) -> DiscoveryRunSpec:
        """The spec an API launch request runs (the router's adapter).

        The API's rules (formerly the router's argv and params ladders) decide
        which dests are ``given``; every other dest keeps the parser default.
        ``start_date`` / ``end_date`` are the dates the router resolved (it owns
        the "today" fallback). An empty indicator pool is forwarded as ``""``
        rather than refused here: the router owns that 422.
        """
        parser = build_parser()
        values: dict[str, Any] = {dest: parser.get_default(dest) for dest in _SPEC_DESTS}
        given: set[str] = set()

        def _give(dest: str, value: object) -> None:
            values[dest] = value
            given.add(dest)

        settings = body.liquidation_settings()
        _give("population_size", body.population)
        _give("max_generations", body.generations)
        _give("mutation_rate", body.mutation_rate)
        _give("crossover_rate", body.crossover_rate)
        _give("elite_count", body.elite_count)
        _give("tournament_size", body.tournament_size)
        _give("convergence_generations", body.convergence_generations)
        _give("max_workers", body.max_workers)
        _give("symbols", ",".join(body.symbols))
        _give("timeframe", body.timeframes[0] if body.timeframes else "4h")
        _give("start_date", start_date)
        _give("end_date", end_date)
        if body.indicator_pool is not None:
            _give("indicator_pool", ",".join(body.indicator_pool))
        if body.direction is not None:
            _give("direction", body.direction)
        if body.eval_windows >= 1:
            _give("eval_windows", body.eval_windows)
        if body.symbol_agg != "portfolio":
            _give("symbol_agg", body.symbol_agg)
        # Always passed: the CLI default is a holdout (0.8), so an explicit 0 must
        # reach the subprocess to disable it.
        _give("train_test_split", body.train_test_split)
        if body.cross_window_months:
            _give("cross_window_months", ",".join(str(m) for m in body.cross_window_months))
            _give("cross_window_min_sharpe", body.cross_window_min_sharpe)
        if body.num_seeds > 1:
            _give("num_seeds", body.num_seeds)
        if body.wfa_oos_step_days > 0:
            _give("wfa_oos_step_days", body.wfa_oos_step_days)
            _give("wfa_min_consistency", body.wfa_min_consistency)
        if body.immigrant_fraction != 0.15:
            _give("immigrant_fraction", body.immigrant_fraction)
        if body.entropy_threshold != 0.4:
            _give("entropy_threshold", body.entropy_threshold)
        if not body.crowding_enabled:
            _give("use_crowding", False)
        if body.seed_run_id is not None:
            _give("seed_from_run", body.seed_run_id)
        if body.seed is not None:
            _give("seed", body.seed)
        if body.no_bootstrap_ci:
            _give("require_bootstrap_ci", False)
        if body.bootstrap_min_sharpe is not None:
            _give("bootstrap_min_sharpe", body.bootstrap_min_sharpe)
        if body.bnb_discount:
            _give("bnb_discount", True)
        if body.evaluator != "nt":
            _give("evaluator", body.evaluator)
        # Run-level leverage settings are forwarded only as needed (vibe-quant-6wq6w):
        # the mode when on, and --leverage only when it is not the default, so a
        # default launch keeps its argv identical to today (C6).
        if settings.on:
            _give("liquidation_mode", settings.mode)
        if "leverage" in settings.to_params():
            _give("leverage", settings.leverage)
        # Run-level sizing is forwarded only when it is not the default, so a default
        # launch keeps its argv byte-identical to today (bead vibe-quant-2gyk1.2, S27).
        sizing_params = body.sizing_settings().to_params()
        for dest in ("max_position_pct", "risk_per_trade", "max_margin_pct"):
            if dest in sizing_params:
                _give(dest, sizing_params[dest])
        # A bankroll grid is forwarded as the three flags, exactly as the request gave it
        # (the CLI re-validates through the same rule).
        if body.bankroll_grid is not None:
            grid = body.bankroll_grid
            _give("bankroll_modes", ",".join(grid.modes))
            _give("bankroll_leverages", ",".join(str(x) for x in grid.leverages))
            _give("bankroll_max_margin_pcts", ",".join(str(x) for x in grid.max_margin_pcts))
        return cls(**values, given=frozenset(given))

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

    def to_params(self) -> dict[str, object]:
        """The API launch's stored parameters, in today's key order.

        The keys the router's params ladder wrote: the GA/worker block always,
        then each option exactly when ``from_request`` gave it, with the run-level
        liquidation and sizing settings last (``seed`` is never stored).
        """
        params: dict[str, object] = {
            "population": self.population_size,
            "generations": self.max_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_count": self.elite_count,
            "tournament_size": self.tournament_size,
            "convergence_generations": self.convergence_generations,
            "max_workers": self.max_workers,
        }
        if "indicator_pool" in self.given:
            params["indicator_pool"] = self.indicator_pool_list() or []
        if "direction" in self.given:
            params["direction"] = self.direction
        if "eval_windows" in self.given:
            params["eval_windows"] = self.eval_windows
        # Always recorded; train_test_split 0 is an explicit opt-out of the default holdout.
        params["symbol_agg"] = self.symbol_agg
        params["train_test_split"] = self.train_test_split
        if "cross_window_months" in self.given:
            params["cross_window_months"] = self.cross_window_month_list()
            params["cross_window_min_sharpe"] = self.cross_window_min_sharpe
        if "num_seeds" in self.given:
            params["num_seeds"] = self.num_seeds
        if "wfa_oos_step_days" in self.given:
            params["wfa_oos_step_days"] = self.wfa_oos_step_days
            params["wfa_min_consistency"] = self.wfa_min_consistency
        if "immigrant_fraction" in self.given:
            params["immigrant_fraction"] = self.immigrant_fraction
        if "entropy_threshold" in self.given:
            params["entropy_threshold"] = self.entropy_threshold
        if "use_crowding" in self.given:
            params["crowding_enabled"] = False
        if "seed_from_run" in self.given:
            params["seed_run_id"] = self.seed_from_run
        if "require_bootstrap_ci" in self.given:
            params["no_bootstrap_ci"] = True
        if "bootstrap_min_sharpe" in self.given:
            params["bootstrap_min_sharpe"] = self.bootstrap_min_sharpe
        if "bnb_discount" in self.given:
            params["bnb_discount"] = True
        if "evaluator" in self.given:
            params["evaluator"] = self.evaluator
        # The leverage settings the CLI will actually run with (vibe-quant-6wq6w):
        # nothing for a default launch, {"leverage": L} when off at a non-default L,
        # {"liquidation_mode", "leverage"} when on (C6/C8/C9). Promote/replay inherit them.
        params.update(self.liquidation_settings().to_params())
        # Run-level sizing (bead vibe-quant-2gyk1.2): only the keys moved off the default.
        params.update(self.sizing_settings().to_params())
        if "bankroll_modes" in self.given:
            params["bankroll_grid"] = {
                "modes": _csv_items(cast("str", self.bankroll_modes)),
                "leverages": [
                    float(item) for item in _csv_items(cast("str", self.bankroll_leverages))
                ],
                "max_margin_pcts": [
                    float(item)
                    for item in _csv_items(cast("str", self.bankroll_max_margin_pcts))
                ],
            }
        return params

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

    def to_notes(
        self,
        result: DiscoveryResult,
        *,
        symbols: list[str],
        use_mock: bool,
        seed: int,
        prior_run_ids: list[int],
        prior_legacy: dict[str, int],
        cells: list[BankrollCell] | None,
    ) -> dict[str, object]:
        """The persisted run notes for this spec (the CLI's one notes writer).

        Every spec-ownable field is re-derived from this spec, exactly as
        ``to_params`` stores it; the caller passes only what the run produced
        (``result``, the resolved ``symbols``, the effective ``seed``, the prior
        DSR trials tallied by the run and the grid ``cells``). ``symbols`` is the
        RESOLVED list the backtests ran on (``--symbols ""`` falls back to the
        run row), never re-derived from ``self.symbols``.
        """
        from vibe_quant.discovery import record

        notes = record.provenance_notes(
            result,
            use_mock=use_mock,
            eval_windows_count=self.eval_window_count(),
            eval_windows=self.plan().eval_windows,
            symbol_agg=self.symbol_agg,
            split_ratio=self.train_test_split,
            direction=self.direction,
            cross_window_months=self.cross_window_month_list(),
            cross_window_min_sharpe=self.cross_window_min_sharpe,
            wfa_oos_step_days=self.wfa_oos_step_days,
            wfa_min_consistency=self.wfa_min_consistency,
            num_seeds=max(1, self.num_seeds),
            seed=seed,
            bootstrap_min_sharpe=self.resolved_bootstrap_min_sharpe(),
            holdout_min_sharpe=self.holdout_min_sharpe,
            symbols=symbols,
            fee_schedule=self.fee_schedule(),
            prior_run_ids=prior_run_ids,
            prior_legacy=prior_legacy,
            liquidation=self.liquidation_settings(),
            sizing=self.sizing_settings(),
            indicator_pool=self.indicator_pool_list(),
        )
        if cells is not None:
            # A grid run has no run-level bankroll: every cell is a setup (R51).
            notes["bankroll_grid"] = bankroll_grid_note(cells)
        return notes

    def to_config(
        self,
        plan: WindowPlan,
        *,
        symbols: list[str],
        prior_trials_fn: Callable[[str, str], int | None] | None,
    ) -> DiscoveryConfig:
        """The ``DiscoveryConfig`` this run executes with.

        ``symbols`` is the RESOLVED list the caller runs the backtests on (the
        CLI falls back to the run row's symbols when ``--symbols ""``).
        """
        from vibe_quant.discovery.pipeline import DiscoveryConfig

        return DiscoveryConfig(
            prior_trials_fn=prior_trials_fn,
            population_size=self.population_size,
            max_generations=self.max_generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elite_count=self.elite_count,
            tournament_size=self.tournament_size,
            convergence_generations=self.convergence_generations,
            max_workers=self.worker_count(),
            symbols=symbols,
            timeframe=self.timeframe,
            start_date=plan.train_start,
            end_date=plan.train_end,
            indicator_pool=self.indicator_pool_list(),
            direction=self.direction,
            eval_windows=self.eval_window_count(),
            symbol_agg=self.symbol_agg,
            train_test_split=self.train_test_split,
            holdout_start_date=plan.holdout_start or "",
            holdout_end_date=plan.holdout_end or "",
            cross_window_months=self.cross_window_month_list(),
            cross_window_min_sharpe=self.cross_window_min_sharpe,
            wfa_oos_step_days=self.wfa_oos_step_days,
            wfa_min_consistency=self.wfa_min_consistency,
            holdout_min_sharpe=self.holdout_min_sharpe,
            holdout_min_trades=self.holdout_min_trades,
            require_bootstrap_ci=self.require_bootstrap_ci,
            bootstrap_min_sharpe=self.resolved_bootstrap_min_sharpe(),
            bootstrap_ci_level=self.bootstrap_ci_level,
            require_dsr=self.require_dsr,
            dsr_rule=cast("Literal['emax', 'q95']", self.dsr_rule),
            require_buy_hold=self.require_buy_hold,
            use_crowding=self.use_crowding,
            immigrant_fraction=self.immigrant_fraction,
            entropy_threshold=self.entropy_threshold,
        )

    def symbol_list(self) -> list[str]:
        """``--symbols`` split and stripped (empty when the flag is empty)."""
        return [item.strip() for item in self.symbols.split(",") if item.strip()]

    def indicator_pool_list(self) -> list[str] | None:
        """``--indicator-pool`` split and stripped; ``None`` when not given."""
        if not self.indicator_pool:
            return None
        return [item.strip() for item in self.indicator_pool.split(",") if item.strip()]

    def worker_count(self) -> int | None:
        """``--max-workers``: a negative value means sequential (``None``)."""
        return self.max_workers if self.max_workers >= 0 else None

    def eval_window_count(self) -> int:
        """The effective multi-window count (at least 1)."""
        return max(1, self.eval_windows)

    def cross_window_month_list(self) -> list[int]:
        """``--cross-window-months`` parsed into month offsets."""
        return parse_cross_window_months(self.cross_window_months or "")

    def resolved_bootstrap_min_sharpe(self) -> float:
        """``--bootstrap-min-sharpe``, or the timeframe-aware floor when not given."""
        if self.bootstrap_min_sharpe is None:
            return _default_bootstrap_min_sharpe(self.timeframe)
        return self.bootstrap_min_sharpe

    def fee_schedule(self) -> FeeSchedule:
        """The venue fee schedule every screening backtest of this run uses."""
        return FeeSchedule(tier=self.fee_tier, bnb_discount=self.bnb_discount)

    def liquidation_settings(self) -> LiquidationSettings:
        """The run-level liquidation settings (``--leverage`` vs its deprecated alias).

        Raises:
            LiquidationConfigError: The mode/leverage pair is invalid.
        """
        resolved = resolve_leverage(
            MISSING if self.leverage is None else self.leverage,
            MISSING if self.liquidation_leverage is None else self.liquidation_leverage,
        )
        if resolved is MISSING:
            resolved = DEFAULT_LEVERAGE
        return LiquidationSettings(
            mode=cast("LiquidationMode", self.liquidation_mode),
            leverage=cast("float", resolved),
        )

    def sizing_settings(self) -> SizingSettings:
        """The run-level exposure cap / risk / margin limit (only given keys)."""
        return SizingSettings.from_params(
            {
                key: value
                for key, value in (
                    ("max_position_pct", self.max_position_pct),
                    ("risk_per_trade", self.risk_per_trade),
                    ("max_margin_pct", self.max_margin_pct),
                )
                if value is not None
            }
        )

    def launch_settings(self, entry_order: str) -> LaunchSettings:
        """The one :class:`LaunchSettings` a non-grid run's engines are built from.

        The run's fee schedule, liquidation and sizing plus the caller's entry
        order (the run row's parameter); ``starting_balance`` stays the default
        because discovery always runs at 1000. A grid run must never call this:
        it has no run-level leverage, so ``__post_init__`` would check the cap
        against the default 10x that the grid path skips (R15b).
        """
        return LaunchSettings(
            fee_schedule=self.fee_schedule(),
            liquidation=self.liquidation_settings(),
            sizing=self.sizing_settings(),
            entry_order=entry_order,
        )

    def is_grid(self) -> bool:
        """Whether any ``--bankroll-*`` flag was passed (all three are then required).

        Raises:
            BankrollGridError: A partial grid, or a grid mixed with the run-level
                liquidation/leverage/margin flags.
        """
        flags = (self.bankroll_modes, self.bankroll_leverages, self.bankroll_max_margin_pcts)
        if all(f is None for f in flags):
            return False
        if any(f is None for f in flags):
            msg = (
                "give all three of --bankroll-modes, --bankroll-leverages and "
                "--bankroll-max-margin-pcts together"
            )
            raise BankrollGridError(msg)
        reject_run_level_with_grid(
            liquidation_mode=self.liquidation_mode,
            leverage=self.leverage,
            liquidation_leverage=self.liquidation_leverage,
            max_margin_pct=self.max_margin_pct,
        )
        return True

    def bankroll_cells(self, sizing: SizingSettings) -> list[BankrollCell]:
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
            _items(cast("str", self.bankroll_modes)),
            _numbers("--bankroll-leverages", cast("str", self.bankroll_leverages)),
            _numbers("--bankroll-max-margin-pcts", cast("str", self.bankroll_max_margin_pcts)),
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
