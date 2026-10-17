# Research lane: the strategy factory

Turns ideas into validated strategies or documented rejections. Roles and contracts: [`README.md`](README.md). This file covers stages, gates, the hypothesis ticket and engine reach.

## Pipeline

| # | Stage | Maker | Checker / gate | Output |
|---|---|---|---|---|
| 1 | Source | `alpha-scout` (+ `cmd` triage for bulk sources) | scout's falsifier + prior-art check | `hypotheses/H-*.json`, ranked shortlist |
| 2 | **User gate** | orchestrator | user picks tickets | `brief.md` updated |
| 3 | Author | `strategy-author` (or a `cmd` draft from [`prompts/cmd/spec-to-dsl-draft.md`](prompts/cmd/spec-to-dsl-draft.md) + author review) | DSL parser; rule-mapping comment complete | `<ticket>_<slug>.yaml` |
| 4 | Screen / discover | `backtest-operator` | min-trades gate; screening sweep or GA seeded with the idea's indicators | run ids + raw metrics |
| 5 | Audit #1 | — | `overfit-auditor`: holdout, bootstrap CI, DSR, WFA / purged k-fold | PASS / REJECT / INSUFFICIENT |
| 6 | Validate | `backtest-operator` | validation (1m detail, latency, fees, funding) + consistency check | validation run id |
| 7 | Audit #2 | — | `overfit-auditor` on screening→validation consistency | PASS / REJECT |
| 8 | Risk | — | `risk-officer` | PASS / FAIL / BLOCK |
| 9 | **User gate** | orchestrator | user approves paper trading | paper session |
| 10 | Record | `cmd` draft ([`prompts/cmd/journal-entry.md`](prompts/cmd/journal-entry.md)), orchestrator edits | every number matches a handoff | journal entry; `bd remember` for durable lessons |

A REJECT at any stage ends the ticket. Journal the rejection and its decisive gate so the next scout does not re-propose the idea.

**Nightly loop (optional):** schedule stages 1→5 (T3 `schedule_task` or `/schedule`). The user gets the shortlist and verdicts each morning; nothing passes stage 5 without gate 9. Paper fills feed stage 1: an analyst compares `/api/paper` fills with validated expectations and files hypotheses about divergence, which re-enter the whole lane. Nothing is auto-deployed.

## Hypothesis ticket

`data/swarm/<job>/hypotheses/H-<YYYYMMDD>-<n>.json`:

```json
{
  "id": "H-20261004-1",
  "source": {"kind": "vault | paper | x | reddit | journal | paper-fills", "ref": "<path or url>"},
  "thesis": "<one sentence: who is wrong, and why price should move>",
  "mechanism": "<behavioural | structural | flow | funding | unknown>",
  "instrument": "BTCUSDT",
  "timeframe": "4h",
  "direction": "long | short | both",
  "indicators": ["RSI", "ATR"],
  "dsl_feasible": true,
  "missing": ["<indicator, data, or semantics the engine lacks>"],
  "falsifier": "<result that kills it, stated before testing>",
  "prior_art": ["<journal batch / bead id with a similar idea and its outcome>"],
  "rank_reason": "<why it is in the top N>"
}
```

## Engine reach

A hypothesis may assume: one Binance USDT-M perp per strategy, OHLCV at the strategy timeframe (+ `additional_timeframes`), funding, the indicator registry (built-ins + `vibe_quant/dsl/plugins/`), one position, SL/TP, time filters. Park these with their unlock (each is a separate engineering-lane job): order book / liquidation maps / CVD, multi-leg stat-arb and pairs, grids and pyramiding, market making, options, on-chain data.

## Source rules
- Social-media posts and vendor threads supply structural ideas only; ignore their performance claims.
- Our gates (holdout used once, bootstrap CI, DSR over real trial counts, WFA, purged k-fold, screening→validation consistency) replace any fixed Sharpe/DD/hit-rate thresholds quoted elsewhere.
- No LLM in the live decision loop: it cannot be backtested reproducibly, may know the future from training data, and breaks exactness rules. Claude thinks offline; compiled NT strategies react.
- The-Quant-Trading-Vault (MIT repo, scraped FMZ/TradingView scripts that may carry their authors' terms): ideas only. Re-express in the DSL, never copy source into the repo; clone outside the repo (e.g. `~/projects/The-Quant-Trading-Vault`). Expect a low `dsl_feasible` hit rate (mostly Pine indicators and FMZ utilities).
- LuxAlgo `luxalgo-mcp-server` and `edge-stats` (MIT) are optional aids for indicator formulas and quick conditional-probability checks. LuxAlgo `PineTS` is **AGPL-3.0**: keep it out of the codebase, never use it as a test oracle.
