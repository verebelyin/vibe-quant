# Research lane: the strategy factory

A continuous pipeline that turns ideas into either validated strategies or documented rejections, faster than edges decay. Roles and contracts are in [`README.md`](README.md). This file covers the lane's stages and gates, the hypothesis ticket, and where the ideas came from.

## Pipeline

| # | Stage | Maker | Checker / gate | Output |
|---|---|---|---|---|
| 1 | Source | `alpha-scout` (+ `cmd` triage for bulk sources) | scout's own falsifier + prior-art check | `hypotheses/H-*.json`, ranked shortlist |
| 2 | **User gate** | orchestrator | user picks which tickets proceed | `brief.md` updated |
| 3 | Author | `strategy-author` (or a `cmd` draft from [`prompts/cmd/spec-to-dsl-draft.md`](prompts/cmd/spec-to-dsl-draft.md) + author review) | DSL parser; rule-mapping comment complete | `<ticket>_<slug>.yaml` |
| 4 | Screen / discover | `backtest-operator` | min-trades gate; screening sweep or GA seeded with the idea's indicators | run ids + raw metrics |
| 5 | Audit #1 | — | `overfit-auditor`: holdout, bootstrap CI, DSR, WFA / purged k-fold | PASS / REJECT / INSUFFICIENT |
| 6 | Validate | `backtest-operator` | validation (1m detail, latency, fees, funding) + consistency check | validation run id |
| 7 | Audit #2 | — | `overfit-auditor` on screening→validation consistency | PASS / REJECT |
| 8 | Risk | — | `risk-officer` | PASS / FAIL / BLOCK |
| 9 | **User gate** | orchestrator | user approves paper trading | paper session |
| 10 | Record | `cmd` draft ([`prompts/cmd/journal-entry.md`](prompts/cmd/journal-entry.md)), orchestrator edits | every number matches a handoff | journal entry, `bd remember` for durable lessons |

A REJECT at any stage ends the ticket. The rejection and its decisive gate still go in the journal, because negative results stop the next scout from re-proposing the same idea.

**Nightly loop (optional):** schedule stages 1→5 (T3 `schedule_task` or `/schedule`). Each morning the user gets the shortlist and the audit verdicts, and nothing goes past stage 5 without the gate at step 9. Paper-trading fills feed back into stage 1 as a source: an analyst reads `/api/paper` fills against the validated expectations and files hypotheses about the divergence. Changes then go through the whole lane again. Nothing is auto-deployed.

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

What a hypothesis may assume today: one Binance USDT-M perp per strategy, OHLCV at the strategy timeframe (+ `additional_timeframes`), funding, the indicator registry (built-ins + `vibe_quant/dsl/plugins/`), one position, SL/TP, time filters. Unreachable today, so park these with their unlock: order book / liquidation maps / CVD, multi-leg stat-arb and pairs, grids and pyramiding, market making, options, on-chain data. Each unlock is a separate engineering-lane job.

## Where the ideas came from (2026-09 posts) and what we adopted

Inputs: Roan (@RohOnChain) threads/articles 2026-09-09 → 09-29 ("GPT-6 Astra" 8-bot fund, AgenKit 10-role harness, "Opus 5.5 + Jev" HFT stack, OpenMarket order-flow desk), @QuantIndicator on LuxAlgo's open-source repos, @RoundtableSpace on The-Quant-Trading-Vault. These posts are marketing for paid products (AgenKit, model subscriptions) and report results after "3–5 days" with no evidence. We took their structural ideas and none of their performance claims.

| Idea from the posts | Our version | Status |
|---|---|---|
| Maker-checker, "nothing grades its own output" | Hard rule; checker on a different model than the maker | adopted (README § Principles) |
| 8 bots, one per fund role; Chief of Staff is the only bot you talk to | Roster; orchestrator = Chief of Staff, one voice | adopted |
| AgenKit: 10 roles, 6 gated phases (brainstorm → architecture → plan → TDD build → two-verdict review → ship) | Engineering lane; `architect` / `implementer` / `reviewer` / `verifier` | adopted (merged roles: debugging → implementer + `diagnosing-bugs` skill; security / perf → reviewer checklist) |
| Risk Bot "zero negotiation authority" | `risk-officer` veto, user-only override | adopted |
| Continuous strategy discovery; alpha decays, so run a factory | This lane + optional nightly loop | adopted |
| Kimi swarm with 300 cheap monitors | Cheap `cmd` fan-out for bulk *reading* (vault triage), not live monitoring | adapted |
| Validation thresholds: Sharpe > 1.5, DD < 15%, hit rate > 55%, t > 2, 5-yr walk-forward | Our gates are stricter and already coded (holdout used once, bootstrap CI, DSR over real trial counts, WFA, purged k-fold, screening→validation consistency) | kept ours |
| 5,806-strategy vault "→ profitable bot in one command" | Idea source: `cmd` triage → `dsl_feasible` shortlist → author → full lane. Most entries are Pine indicators or FMZ utilities; expect a low hit rate | adopted as a source |
| "Opus thinks, Jev reacts": LLM decides buy/sell live in < 100 ms | Rejected for the trading path. A non-deterministic LLM in the decision loop can't be backtested reproducibly, may know the future from training data, and breaks our exactness rules. Our split is "Claude thinks offline, compiled NT strategies react" | rejected |
| Nightly "review the fills, ship a better version" | Paper fills as a hypothesis source; every change goes through the full lane | adapted |
| OpenMarket order flow (liquidation heatmap, depth, CVD) | Needs new data sources and DSL features | parked: engineering-lane job if wanted |
| Kelly-sized positions, Telegram alerts | `vibe_quant/risk/sizing.py`, `vibe_quant/alerts/telegram.py` already exist | existing |

### External resources (checked 2026-10-04)

| Resource | License | Use |
|---|---|---|
| [The-Quant-Trading-Vault](https://github.com/brainbrick-trades/The-Quant-Trading-Vault) | repo MIT; content scraped from FMZ/TradingView, so individual scripts may carry their authors' terms | ideas only. Re-express in the DSL, never copy source into the repo. Clone outside the repo (e.g. `~/projects/The-Quant-Trading-Vault`) |
| [LuxAlgo/luxalgo-mcp-server](https://github.com/LuxAlgo/luxalgo-mcp-server) | MIT | optional MCP for scout/author: indicator definitions and formulas |
| [LuxAlgo/edge-stats](https://github.com/LuxAlgo/edge-stats) | MIT | optional: quick conditional-probability checks on bars before writing a strategy |
| [LuxAlgo/PineTS](https://github.com/LuxAlgo/PineTS) | **AGPL-3.0** | keep out of the codebase (conventions: avoid AGPL). At most a throwaway external tool for eyeballing a Pine port. Never use it as a test oracle in the repo |
| LuxAlgo/market-trackers | MIT | US equities/congress data, not relevant to crypto perps |
