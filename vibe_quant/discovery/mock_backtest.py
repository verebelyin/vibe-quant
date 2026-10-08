"""Deterministic pseudo-backtest metrics for testing the GA loop.

This lives outside ``vibe_quant.discovery.__main__`` so that
``ProcessPoolExecutor`` workers can unpickle it: functions defined in a
module executed as ``__main__`` (via ``python -m vibe_quant.discovery``)
pickle with ``__module__ == '__main__'``, which worker processes cannot
resolve.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome


def mock_backtest(chromosome: StrategyChromosome) -> dict[str, float | int]:
    """Generate deterministic pseudo-backtest metrics for testing GA loop."""
    seed_bytes = repr(chromosome).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big")
    rng = random.Random(seed)

    genes = len(chromosome.entry_genes) + len(chromosome.exit_genes)
    complexity = min(1.0, genes / 20.0)

    sharpe = max(0.05, 1.8 - (0.7 * complexity) + rng.uniform(-0.35, 0.35))
    max_drawdown = min(0.95, max(0.02, 0.14 + (0.2 * complexity) + rng.uniform(-0.05, 0.08)))
    profit_factor = max(0.2, 1.6 - (0.5 * complexity) + rng.uniform(-0.25, 0.35))
    total_trades = max(60, int(120 + rng.randint(-30, 90) - (genes * 2)))

    # Estimate total return from metrics
    total_return = max(-0.5, (sharpe * 0.15) - (max_drawdown * 0.3) + rng.uniform(-0.1, 0.2))

    # Generate synthetic per-trade returns for bootstrap CI guardrail.
    # Std must be small relative to mean for bootstrap CI to pass.
    mean_ret = total_return / max(1, total_trades)
    trade_returns = tuple(
        rng.gauss(mean_ret, abs(mean_ret) * 0.5 + 0.0001)
        for _ in range(total_trades)
    )

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "total_return": total_return,
        "trade_returns": trade_returns,  # type: ignore[dict-item]
    }
