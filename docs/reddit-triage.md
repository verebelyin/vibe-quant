# Reddit research — extraction sweep triage (2026-07-11)

First credibility-ranked triage from the upgraded pipeline (wayfinder
vibe-quant-7go7m). Corpus: 378 items, 9 approved subs, 428 archived images.

## Sweep summary

- Items with ≥1 extraction attempt: 90 / 378 (prioritized: strategy-dense subs
  + image-bearing first). Remaining ~288 still `pending` — drainable via the
  extraction worker (low expected yield, see below).
- Findings: **6 parsed / 279 total = 2.2% parse rate** — matches the historical
  ~2/217 base rate.
- Vision confirmed working: the extractor reads equity-curve/PnL metrics off
  screenshots (e.g. "Sharpe 1.58, MaxDD 18.64%"). But top-voted posts are
  overwhelmingly memes/results-brags, not implementable rule-sets. Strategy
  density is uncorrelated with score.

## Credibility-ranked candidates

Ranked by evidence_level (live_traded > backtested > idea_only) then completeness.
Screen figures below are **isolated** where noted; the concurrent-sweep `screen_*`
values are contaminated (bead vibe-quant-7go7m.10) and excluded.

| Item | Sub | Evidence | Compl. | Strategy | Isolated screen (BTCUSDT 1m) |
|------|-----|----------|--------|----------|------------------------------|
| 184 | algotrading | backtested | 0.65 | `ema_crossover_adx_trend_1m` — 13/34 EMA × ADX>33, ATR 2.45/5.0 | **Sharpe −3.79, −1.00%, 16 trades, 30d** (first catch) |
| 462 | algotrading | live_traded | 0.60 | RSI-oversold alerts on momentum-ranked universe | needs isolated re-screen |
| 184 | algotrading | backtested | 0.40 | `vwap_range_rsi_reversion_1m` — VWAP+RSI mean-reversion, ADX 15-33 | needs isolated re-screen (concurrent run showed catastrophic over-trading) |
| 149 | algotrading | idea_only | 0.35 | Simple RSI mean-reversion | needs isolated re-screen (concurrent run showed marginal positive) |

## Recommendation

- **First catch taken**: item 184 EMA/ADX crossover — implemented + screened
  honestly (negative), journaled. Destination met.
- **Worth an isolated re-screen** before discarding: item 149 mean-reversion and
  item 462 RSI-alerts (both showed non-negative concurrent screens, but those are
  unreliable — see bead .10).
- **Full-corpus extraction** (remaining ~288 pending) is low priority: 2.2% parse
  rate means ~6 more strategies for hours of `claude -p` compute. Better ROI:
  drain incrementally via the worker, or narrow to algotradingcrypto +
  algorithmictrading text posts with explicit rule tables.
