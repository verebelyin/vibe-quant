"""Run ONE fixed DSL strategy YAML on a symbol + window through the shared engine; print metrics as JSON.

The rule-test skill's runner (.claude/skills/rule-test/SKILL.md): a simple trading rule is
tested only through the engine's own path, never a private RSI, fill loop or cost model.
``--tier nt`` is the NT screening runner (any timeframe, ``max_bars_in_trade`` allowed);
``--tier fast`` is the 1m fast-tier replay (``FastBacktestFn``), exact against NT where
``scripts/agents/fast_tier_parity.py`` exits 0 and refused (exit 2) for a rule it cannot replay.
Costs, fills, funding and sizing are the screening defaults (taker fees, modeled slippage,
funding); ``--max-position-pct`` / ``--risk-per-trade`` / ``--leverage`` override the launch
settings. Scratch DB under /tmp; the real state DB is never opened. The last stdout line is JSON.

    PYTHONPATH=$PWD .venv/bin/python scripts/agents/rule_test.py rule.yaml \\
        --symbols BTCUSDT --start 2026-03-02 --end 2026-03-09 [--tier nt|fast] [--leverage 10]
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRIC_KEYS = (
    "total_trades",
    "total_return",
    "sharpe_ratio",
    "profit_factor",
    "max_drawdown",
    "total_fees",
    "gross_pnl",
)


class TierUnsupported(Exception):
    """The fast tier cannot replay this rule (run it with ``--tier nt``)."""


class RuleRunFailed(Exception):
    """The engine run failed; any metrics it returned are a failure sentinel, not a result."""


class _Rule:
    """Stands in for a chromosome: the backtest fns read only ``uid`` from it (logging)."""

    uid = "rule-test"


def _check_window(start: str, end: str) -> None:
    try:
        first, last = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    except ValueError as exc:
        raise RuleRunFailed(f"--start/--end must be YYYY-MM-DD: {exc}") from exc
    if last <= first:
        raise RuleRunFailed(f"--end {end} must be after --start {start} (end is exclusive)")


def run_rule(
    yaml_path: Path | str,
    symbols: list[str],
    start: str,
    end: str,
    *,
    tier: str = "nt",
    leverage: int = 10,
    max_position_pct: float | None = None,
    risk_per_trade: float | None = None,
) -> dict[str, Any]:
    """Metrics of the YAML strategy over ``[start, end)`` on ``symbols``, from the shared runner."""
    import yaml

    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.dsl.parser import parse_strategy
    from vibe_quant.fast_tier import FastTierUnsupported
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    _check_window(start, end)
    parse_strategy(str(yaml_path))  # fail closed on a YAML the engine would reject
    dsl = yaml.safe_load(Path(yaml_path).read_text())
    kwargs: dict[str, Any] = {
        "liquidation": LiquidationSettings(mode="off", leverage=leverage),
    }
    default = SizingSettings()
    kwargs["sizing"] = SizingSettings(
        max_position_pct=default.max_position_pct if max_position_pct is None else max_position_pct,
        risk_per_trade=default.risk_per_trade if risk_per_trade is None else risk_per_trade,
    )
    fn_cls = {"nt": NTBacktestFn, "fast": FastBacktestFn}[tier]
    fn = fn_cls(symbols, str(dsl["timeframe"]), start, end, **kwargs)
    out: dict[str, Any] = {"tier": tier, "symbols": symbols, "start": start, "end": end}
    if tier == "nt":
        # The runner NTBacktestFn builds for discovery (one launch-settings construction site).
        metrics = fn._make_runner(copy.deepcopy(dsl), start, end, symbols)({})
        if metrics.sharpe_ratio == float("-inf"):
            # NTScreeningRunner masks a failed run as sharpe -inf plus default zero metrics
            # (nt_runner.py __call__); a valid zero-trade run has sharpe 0.0, never -inf.
            raise RuleRunFailed(
                "engine run failed (the shared runner returned its failure sentinel, sharpe -inf); "
                "see the runner's 'NT screening backtest failed' warning above"
            )
        out.update({k: getattr(metrics, k) for k in METRIC_KEYS})
        out["win_rate"] = metrics.win_rate
        return out
    # The fast tier replays a chromosome; hand it this YAML's DSL where it would convert one.
    try:
        with mock.patch(
            "vibe_quant.discovery.genome.chromosome_to_dsl", lambda _c: copy.deepcopy(dsl)
        ):
            result = fn._run_single(_Rule(), start, end, symbols)  # type: ignore[arg-type]
    except FastTierUnsupported as exc:
        raise TierUnsupported(str(exc)) from exc
    out.update({k: result[k] for k in METRIC_KEYS})
    out["win_rate"] = None  # the fast tier's result dict carries no win rate; use --tier nt
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml", type=Path, help="strategy DSL YAML (vibe_quant/dsl/schema.py)")
    ap.add_argument("--symbols", required=True, help="comma list, e.g. BTCUSDT")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive)")
    ap.add_argument("--tier", choices=("nt", "fast"), default="nt")
    ap.add_argument("--leverage", type=int, default=10)
    ap.add_argument("--max-position-pct", type=float, default=None)
    ap.add_argument("--risk-per-trade", type=float, default=None)
    args = ap.parse_args(argv)

    yaml_path = args.yaml.resolve()  # before the chdir below
    tmp = Path(tempfile.mkdtemp(prefix="rule-test-", dir="/tmp"))
    os.environ["VIBE_QUANT_DB"] = str(tmp / "state.db")
    os.environ["VIBE_QUANT_LOG_DIR"] = str(tmp / "logs")
    os.chdir(REPO_ROOT)
    try:
        out = run_rule(
            yaml_path,
            [s.strip() for s in args.symbols.split(",") if s.strip()],
            args.start,
            args.end,
            tier=args.tier,
            leverage=args.leverage,
            max_position_pct=args.max_position_pct,
            risk_per_trade=args.risk_per_trade,
        )
    except RuleRunFailed as exc:
        print(f"rule_test failed, no metrics: {exc}", file=sys.stderr)
        return 1
    except TierUnsupported as exc:
        print(f"fast tier cannot replay this rule: {exc}; use --tier nt", file=sys.stderr)
        return 2
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
