# Task: Subreddit shortlist research (beads ticket vibe-quant-7go7m.3)

Read AGENTS.md for project context. This is a WEB RESEARCH task — use web search extensively. Do NOT modify any code. Deliverable: `docs/reddit-subreddit-shortlist.md`. Do NOT commit.

## Context

Algorithmic trading engine for CRYPTO PERPETUAL FUTURES. We scrape Reddit for posts containing implementable strategy rules (entries/exits/indicators/params) and bankroll/position-sizing ideas. Current source: r/algotrading only. Extraction stats: 215 of 217 LLM extraction attempts skipped — much of the corpus is equities-focused chatter, jokes, or meta-discussion. We want subs with high density of CONCRETE strategy content translatable to crypto perps. Strategy posts often include screenshots (rule tables, TradingView setups, equity curves) — image-richness is a plus (we're adding vision extraction).

## What to research

Evaluate candidate subreddits (at minimum: algotrading, quant, quantfinance, Daytrading, swingtrading, Forex, CryptoMarkets, BitcoinMarkets, CryptoCurrencyTrading, algorithmictrading, thewallstreet, FuturesTrading, options — plus any better ones you find). For each:

1. **Strategy-post density**: how often do posts share concrete, implementable rules (not "what broker should I use")? Sample recent top posts via web search / site: queries to judge.
2. **Crypto-perp relevance**: native crypto content, or equities/forex needing translation (leverage/funding/24-7 markets)?
3. **Image usage**: do strategy posts commonly attach screenshots/charts?
4. **Volume + moderation**: posts/day, is self-promotion spam dominant, do mods allow strategy sharing (some subs ban "low-effort" strategy posts)?
5. **Bankroll/money-management content**: any sub notably good for position sizing/risk discussions?

## Deliverable format

`docs/reddit-subreddit-shortlist.md`:
- Scoring table: sub × (strategy density 1-5, crypto relevance 1-5, image richness 1-5, volume, notes)
- Proposed final list: 5-8 subs ranked, each with recommended listing config (top/all + top/year for backfill; new vs top/week for incremental) and expected signal type
- Explicitly rejected subs with one-line reasons
- 5-line executive summary at top

The user will approve/trim the final list — make the table easy to judge at a glance.
