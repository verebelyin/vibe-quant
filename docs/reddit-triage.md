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
Screen = auto_screen on BTCUSDT (the pipeline default); values verified reliable
(item 462 reproduces bit-identically in isolation — the earlier "contamination"
suspicion was a mis-diagnosis, closed bead vibe-quant-7go7m.10).

| Item | Evidence | Compl. | TF | Strategy | Screen (BTCUSDT) |
|------|----------|--------|----|----------|------------------|
| 184 | backtested | 0.65 | 1m | `ema_crossover_adx_trend_1m` — 13/34 EMA × ADX>33, ATR 2.45/5.0 | **Sharpe −3.79, −1.00%, 16 trades** (isolated 30d; auto_screen 180d timed out) — first catch |
| 462 | live_traded | 0.60 | 4h | `momentum_universe_rsi_oversold_swing` — RSI cross 30/70, 25% stop / 3% target | Sharpe −1.50, −3.22%, 15 trades |
| 184 | backtested | 0.40 | 1m | `vwap_range_rsi_reversion_1m` — VWAP+RSI reversion, ADX 15-33 | Sharpe −21.7, −93.7%, 5171 trades (catastrophic over-trading) |
| 149 | idea_only | 0.35 | 4h | `sma_mean_reversion` | **Sharpe +0.26, +2.19%, 99 trades** (only positive) |

## Recommendation

- **First catch taken**: item 184 EMA/ADX crossover — implemented + screened
  honestly (Sharpe −3.79, losing), journaled. Destination met.
- **Only non-losing screen**: item 149 `sma_mean_reversion` (Sharpe +0.26, +2.19%,
  99 trades) — but idea_only credibility and a weak +0.26 Sharpe that would almost
  certainly fail the bootstrap-CI / validation gates (every forced champion has
  collapsed). Not worth promoting without more evidence.
- **Full-corpus extraction** (remaining ~288 pending) is low priority: 2.2% parse
  rate means ~6 more strategies for hours of `claude -p` compute. Better ROI:
  drain incrementally via the worker, or narrow to algotradingcrypto +
  algorithmictrading text posts with explicit rule tables.
