"""Golden fixture for every champion gate stage (bead vibe-quant-rabmo.6, T5a).

Deterministic synthetic backtests with five tag classes: tag 2 is rejected by the
DSR guardrail, tag 3 by the shifted cross-window, tag 4 by the rolling WFA and
tag 5 by the holdout; tag 1 is the only champion. ``gate_golden.py`` digests the
outcomes and every rabmo.6 refactor step must reproduce them IDENTICALLY.
"""

from __future__ import annotations

import hashlib
import math
import random
from array import array
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.discovery.mock_backtest import _seeding_repr
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult
from vibe_quant.metrics import DailyReturns

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.discovery.operators import StrategyChromosome

PROGRESS_FILE = "/tmp/gate_golden_progress.json"
SERIES_DAYS = 181
SERIES_FIRST_DAY = 19_723

CONFIG = DiscoveryConfig(
    population_size=30,
    max_generations=2,
    top_k=10,
    elite_count=1,
    max_workers=1,
    min_trades=50,
    timeframe="4h",
    symbols=["BTCUSDT"],
    start_date="2024-01-01",
    end_date="2024-07-01",
    holdout_start_date="2024-07-01",
    holdout_end_date="2024-09-01",
    train_test_split=0.75,
    cross_window_months=[1],
    wfa_oos_step_days=30,
    require_dsr=True,
    require_bootstrap_ci=False,
    min_diversity_distance=0.0,
)
PLAIN_MOCK_CONFIG = replace(CONFIG, population_size=8, max_generations=3, top_k=5)


def tag(chromosome: StrategyChromosome) -> int:
    """The chromosome's stable 1..5 class (the hash ``mock_backtest`` seeds with)."""
    digest = hashlib.blake2b(_seeding_repr(chromosome).encode(), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % 5


def series(sr_ann: float, seed: int, n: int = SERIES_DAYS) -> DailyReturns:
    """A standardized-normal mark-to-market daily series with annual Sharpe ``sr_ann``."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    values = z * 0.01 + sr_ann / math.sqrt(252) * 0.01
    return DailyReturns(SERIES_FIRST_DAY, array("d", values.tobytes()))


def train(chromosome: StrategyChromosome) -> dict[str, float | int]:
    """Train metrics per tag; tag 2 carries the low (DSR-rejected) series."""
    t = tag(chromosome)
    return {
        "sharpe_ratio": 3.0 + t / 10,
        "max_drawdown": 0.1,
        "profit_factor": 2.0,
        "total_trades": 120 + t,
        "total_return": 0.4,
        "mtm_daily_returns": series(0.5 if t == 2 else 6.0, seed=t + 1),
    }


def factory(start: str, end: str) -> Callable[[StrategyChromosome], dict[str, float | int]]:
    """Window metrics: tag 3 fails a long window, tag 4 a short (WFA) one."""
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days

    def backtest(chromosome: StrategyChromosome) -> dict[str, float | int]:
        t = tag(chromosome)
        bad = (t == 3 and days > 60) or (t == 4 and days <= 60)
        return {
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.1,
            "profit_factor": 2.0,
            "total_trades": 60,
            "total_return": -0.1 if bad else 0.1,
        }

    return backtest


def holdout(chromosome: StrategyChromosome) -> dict[str, float | int]:
    """Holdout metrics: tag 5 loses money."""
    return {
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.1,
        "profit_factor": 2.0,
        "total_trades": 60,
        "total_return": -0.05 if tag(chromosome) == 5 else 0.1,
    }


def run_single() -> DiscoveryResult:
    """Section (a): one GA run of the fixture."""
    random.seed(7)
    pipeline = DiscoveryPipeline(
        config=CONFIG,
        backtest_fn=train,
        progress_file=PROGRESS_FILE,
        holdout_backtest_fn=holdout,
        backtest_fn_factory=factory,
    )
    return pipeline.run()


def run_multi_seed() -> DiscoveryResult:
    """Section (b): the fixture through the pooled multi-seed re-check (2 seeds)."""
    from vibe_quant.discovery.pooling import _run_multi_seed

    return _run_multi_seed(
        2,
        CONFIG,
        train,
        PROGRESS_FILE,
        holdout_backtest_fn=holdout,
        backtest_fn_factory=factory,
    )


def run_plain_mock() -> DiscoveryResult:
    """Section (a0): the stock mock GA (all rejected, legacy DSR input)."""
    from vibe_quant.discovery.mock_backtest import mock_backtest

    random.seed(7)
    pipeline = DiscoveryPipeline(
        config=PLAIN_MOCK_CONFIG,
        backtest_fn=mock_backtest,
        progress_file=PROGRESS_FILE,
    )
    return pipeline.run()


def stage_counts(result: DiscoveryResult) -> dict[str, int]:
    """Rejections per gate stage, sorted by stage name."""
    counts: dict[str, int] = {}
    for rejection in result.guardrail_rejections:
        stage = str(rejection.get("stage"))
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))
