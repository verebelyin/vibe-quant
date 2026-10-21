"""Fast tier: a NautilusTrader-free evaluator of discovery genomes on 1m bars (exploration only).

The screening path (``NTBacktestFn`` -> ``NTScreeningRunner``) is untouched; this is a
separate evaluator that must reproduce it, not replace it. ``FastBacktestFn`` takes the
same chromosome and returns the same metrics dict, built from three parts:

* signals: the GENERATED strategy's own indicator, readiness and condition code, driven
  bar by bar without an engine (:mod:`vibe_quant.fast_tier.signals`) -- exact by
  construction, no second implementation of the DSL;
* fills: an event-jumping replay of the screening venue (bar ticks O, H/L, C; one fill-RNG
  draw per fill; 1-tick adverse slip; at-limit fill draws) into real NT ``Position``
  objects (:mod:`vibe_quant.fast_tier.fills`);
* metrics: the screening runner's own ``_extract_metrics`` over a cache facade, so funding,
  mark-to-market drawdown, Sharpe, profit factor and the liquidation overlay are the
  existing code.

Results are never champions, never gate inputs and never written to the state DB.
Shapes the replay cannot reproduce exactly raise :class:`FastTierUnsupported`
(``NTBacktestFn.__call__`` turns it into an error row) instead of being approximated.
Parity with screening is checked by ``scripts/agents/fast_tier_parity.py``.
"""

from vibe_quant.fast_tier.errors import FastTierUnsupported

__all__ = ["FastTierUnsupported"]
