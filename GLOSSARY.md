# vibe-quant

A research-to-trading pipeline for crypto perpetual futures. It searches for strategies with a repeatable edge, makes them survive honest evidence (backtests, overfitting gates, paper trading), and only then trades them live.

## Strategies

**Strategy**:
A declarative set of trading rules: indicators, entry and exit conditions, stop-loss and take-profit, and time filters.
_Avoid_: strategy config, template, DSL config

**Strategy DSL**:
The YAML language strategies are written in.
_Avoid_: strategy format, schema

**Condition**:
One boolean rule over indicators or prices (e.g. "RSI below 30", "fast MA crosses above slow MA") that opens or closes a long or short.
_Avoid_: signal, rule

**Indicator**:
A named computation over market data that conditions refer to.

**Indicator plugin**:
An indicator added as a drop-in file instead of being built in.
_Avoid_: custom indicator

**Context indicator**:
An indicator that reads data other than the traded bars, such as funding or the BTC trend.

**Timeframe**:
The bar size a strategy runs on, from 1m to 1d.
_Avoid_: TF, interval

**Sweep**:
A strategy's grid of parameter ranges, every combination of which is screened.
_Avoid_: parameter search (that is discovery)

## Backtesting

**Backtest run**:
One recorded execution of a strategy over a date range in a given run mode (screening, validation, discovery, paper).
_Avoid_: job (the background process that executes a run)

**Engine tier**:
Which of the two backtest fidelities a run uses: screening or validation.
_Avoid_: mode (alone), backtest type

**Screening**:
The fast engine tier with simplified fills; discovery always runs on it.
_Avoid_: quick backtest, NT simplified

**Validation**:
The full-fidelity engine tier that fills orders on 1-minute detail bars and charges full costs; it is the reference for how fills really happen.
_Avoid_: full backtest; never use bare "validation" for cross-window, cross-validation or DSL checks

**Win rate**:
Share of a run's trades whose net PnL (after fees, funding and modeled slippage) is above zero; trades still open at the end count at their mark.
_Avoid_: hit rate

**Fast tier**:
An exploration-only replay of screening for 1-minute, single-symbol genomes that needs no engine run; it must match screening fill for fill, and it never yields champions or gate inputs.
_Avoid_: fast screening, approximate backtest

**Burst**:
Many tiny seeded discoveries on a scratch database (indicator pair × direction × symbol × month × leverage) used to choose where a real discovery should search.

**Detail bars**:
The 1-minute bars validation uses to release and fill orders placed on a slower timeframe.

**Fill tick**:
One of two synthetic 1-minute prices per strategy bar on which screening releases queued orders.

**Outbox**:
The per-instrument queue that holds an order until that instrument's own next price update.
_Avoid_: latency model (which releases venue-wide)

**Fill model**:
The rule that turns a released order into a fill price and quantity, including slippage.

**Latency preset**:
A named network distance (co-located to retail) that delays order arrival.

**Data window**:
The date range a run actually covered after being clamped to the available data.
_Avoid_: window (alone)

**Replay**:
Re-running a discovery champion through screening to check that it reproduces its recorded metrics.

**Replay drift**:
A difference between a champion's recorded metrics and its replay.

**Screening–validation consistency**:
How much a strategy's results degrade from screening to validation; a large collapse marks it execution-sensitive.
_Avoid_: consistency (alone; WFA uses that word too)

**Exactness proof**:
A pinned strategy run whose metrics must reproduce bit for bit, proving a change did not move the numbers.
_Avoid_: parity test, fixed-strategy eval

**Comparability break**:
A dated change after which metrics from before and after can no longer be compared.

## Costs, sizing and risk

**Fees**:
Exchange commissions; market orders pay the taker rate, resting limit orders the maker rate.
_Avoid_: commission (in prose), costs (which also include funding and slippage)

**Funding**:
The periodic payment between longs and shorts on a perpetual, proportional to position notional.

**Funding interval**:
How often a symbol's funding settles, which differs by symbol and venue.

**Fallback funding rate**:
A flagged stand-in rate charged when a funding settlement is missing from the data.

**Mark price**:
The exchange's reference price for liquidation and funding, as opposed to the last traded price.
_Avoid_: index price

**Leverage**:
The multiple of equity a run's venue allows per position; a run-level setting, never searched.

**Liquidation overlay**:
An opt-in model of exchange liquidation, either off, isolated margin or cross margin.
_Avoid_: liquidation mode (when you mean on/off)

**Sizing**:
How large each entry is: the smaller of the risk-based size and the max exposure.
_Avoid_: position size gene (sizing is never searched)

**Risk per trade**:
The fraction of balance a trade loses if its stop is hit.

**Max exposure**:
The largest notional a single position may hold, as a multiple of equity.
_Avoid_: position cap, exposure cap

**Margin limit**:
A ceiling on margin in use as a fraction of equity; an entry that would cross it is refused, never shrunk.
_Avoid_: cushion, total exposure cap

**Starting balance**:
The account balance a run starts with.
_Avoid_: bankroll, initial balance, capital

## Discovery and gates

**Discovery run**:
A genetic search over genomes that ends with zero or more champions.
_Avoid_: GA run, search run

**Genome**:
The searchable encoding of a strategy: its genes, stop and take-profit, and direction.
_Avoid_: chromosome, individual

**Gene**:
One searchable piece of a genome, typically an indicator with a condition and threshold.

**Population**:
The set of genomes alive in one generation.

**Generation**:
One round of evaluating, selecting and breeding the population.

**Immigrant**:
A random genome injected into the population to restore diversity.

**Fitness**:
The single number discovery ranks genomes by, blending risk-adjusted return, drawdown and profit factor minus penalties.
_Avoid_: score

**Overtrade penalty**:
A fitness deduction for trading more often than the timeframe justifies.

**Eval windows**:
Sub-windows of the train range; a genome's fitness is its worst result across them.
_Avoid_: folds

**Train range**:
The part of the date range discovery may search on.
_Avoid_: in-sample (outside WFA)

**Holdout**:
The final slice of the date range, never searched, used once as the last gate.
_Avoid_: test set, out-of-sample (outside WFA)

**Gate**:
A pass/fail check a genome must clear to become a champion; gates fail closed.
_Avoid_: filter, criterion

**Guardrails**:
The first group of gates: minimum trades, complexity, bootstrap CI and DSR.

**Bootstrap CI**:
A resampled confidence bound on a strategy's trade-level edge.

**DSR (Deflated Sharpe Ratio)**:
A Sharpe ratio discounted for how many strategies were tried before finding it.

**Trial**:
One evaluated candidate that counts toward the DSR's multiple-testing discount.

**Effective trial count**:
The number of truly independent trials, after discounting correlated ones and adding overlapping earlier runs.
_Avoid_: N (alone), raw N (unless you mean the raw count)

**Cross-window gate**:
Re-runs a genome on train ranges shifted in time; every shift must pass.

**Walk-forward analysis (WFA)**:
Repeated optimise-then-test on rolling windows to measure how well in-sample results carry forward.

**Walk-forward efficiency (WFE)**:
Mean out-of-sample return per day over mean in-sample return per day, across WFA windows.

**Rolling-window check**:
Discovery's opt-in stability gate (`--wfa-oos-step-days`): back-to-back windows over the train range, no re-optimisation; a share of them must be profitable.
_Avoid_: WFA (alone; that is the overfitting pipeline's walk-forward analysis)

**Purged k-fold**:
Cross-validation with gaps between folds so overlapping trades cannot leak.

**Buy-and-hold gate**:
A champion must beat holding the same symbols in the same direction.

**Random baseline**:
A null distribution of random entries with the champion's exits, used to test that its entries add value.

**Symbol aggregation**:
How a multi-symbol genome is scored: as one shared portfolio account, or by its worst symbol.

**Champion**:
A genome that passed every gate.
_Avoid_: top strategy, survivor, winner

**Promote**:
Turn a discovery champion into a stored strategy with its own screening or validation run.

**Warm-start run**:
An earlier discovery run whose champions seed a new run's first population.
_Avoid_: seed run, seed strategy

**RNG seed**:
The random seed that makes a discovery run reproducible.
_Avoid_: seed (alone)

**Multi-seed run**:
One discovery repeated with several RNG seeds and ranked as an ensemble.

**Bankroll grid**:
A discovery repeated across margin mode, leverage and margin-limit combinations.
_Avoid_: bankroll (it varies no money)

**Regime-cross campaign**:
Discovery trained on each market-regime window, then validated out-of-sample on every other window.

## Paper and live trading

**Paper trading**:
Running a strategy against the exchange testnet with real-time data and simulated money.
_Avoid_: sandbox, demo

**Live trading**:
The same as paper trading against a real-money account; it requires the operator's explicit consent.

**Paper session**:
One continuous run of paper or live trading for a trader, from start to stop.
_Avoid_: paper node, paper run

**Trader**:
The persistent identity of a paper or live trading setup, which outlives sessions and API keys.
_Avoid_: bot, instance

**Account**:
One exchange API key on one environment (testnet or live).
_Avoid_: wallet, key

**Key rotation**:
Starting a trader on a new account; the trader keeps its risk state and halt.

**Checkpoint**:
A periodic snapshot of a session's positions, orders, balance and risk state.

**Risk carry-over**:
A new session inheriting the risk bookkeeping and halt of its trader's latest checkpoint.
_Avoid_: restore (alone)

**Start fresh**:
Starting a session that deliberately skips risk carry-over.

**Live consent**:
The explicit confirmation required to trade real money, and again to do so without carried-over risk.

**Account lock**:
The rule that only one session may trade an account at a time.

**Guard**:
The part of a session that owns its trading state and approves or blocks every order.
_Avoid_: circuit breaker, risk manager

**Halt**:
A trading state where only position-reducing orders pass, positions are flattened and strategies stop; it records a halt reason.
_Avoid_: stop, kill

**Halt reason**:
Why a session halted: max drawdown, daily loss, consecutive losses, manual, error, signal or kill switch.

**Pause**:
A trading state where new entries are blocked but existing stops and take-profits stay live.

**Resume**:
Returning a paused or operator-resumable halted session to running.

**Daily loss limit**:
The largest loss allowed within one UTC day before the session halts until midnight.

**Kill switch**:
A system-wide stop that halts every session; it fails closed and only the operator can release it.
_Avoid_: emergency stop, panic button

**Liquidation watch**:
A report-only monitor of exchange liquidations on the symbols a trader cares about.

**Event log**:
The append-only record of everything a session saw and did: signals, fills, risk checks, halts.
_Avoid_: trade log

**Alert**:
A message to the operator about errors, halts, trades or daily summaries.
_Avoid_: notification

**Reconciliation**:
Pairing a paper session's trades with a validation run of the same strategy to measure the gap.

## Data

**Archive**:
The append-only store of raw market downloads, the source of truth for market data.
_Avoid_: raw data, cache

**Catalog**:
The engine-ready market data, rebuildable from the archive at any time.
_Avoid_: data store, dataset

**Download session**:
One recorded ingest of market data into the archive, ending completed, partial or failed.

**Venue**:
An exchange a strategy can trade: Binance USDT-M perpetuals, or the Ethereal DEX.
_Avoid_: exchange (in code-facing prose)

## Funding carry

**Funding carry**:
A weekly cross-sectional book that goes long the lowest-funding symbols and short the highest, in one pooled account; it is not a DSL strategy.
_Avoid_: portfolio (alone), carry strategy

## Research

**Research item**:
One fetched external post or article that may contain a trading idea.
_Avoid_: post, source

**Extraction**:
Turning a research item into a candidate strategy, an evidence rating and any missing indicators.

**Auto-screen**:
The screening run that every successful extraction triggers.

**Indicator scaffold**:
A generated draft of a missing indicator, awaiting review before it becomes a plugin.
_Avoid_: scaffold (alone)

**Hypothesis**:
A falsifiable trading idea with a thesis, a mechanism and a falsifier, written before any backtest.
_Avoid_: idea, alpha (alone)

## Agent workflow

**Operator**:
The one person who runs the project and approves anything that touches money.
_Avoid_: user (in trading contexts), admin

**Chief**:
The orchestrating agent that plans, dispatches and verifies work.
_Avoid_: orchestrator, lead

**Maker**:
A coding agent that implements one task.
_Avoid_: worker, implementer (that is one maker role)

**Checker**:
A reviewing agent that can block a maker's change.
_Avoid_: reviewer (that is one checker role)

**Bead**:
One tracked issue.
_Avoid_: ticket, task, todo

**Swarm board**:
The shared, persistent board where agents post findings, gotchas and decisions.

**Pre-review gate**:
The automated checks a change must pass before review.
_Avoid_: gate (alone; that means a discovery gate)

## Overloaded words

Always qualify these; each means several different things here.

- **window**: eval window, cross-window shift, WFA window, holdout, data window, trading-hours filter, carry signal window.
- **run**: backtest run, discovery run, one seed of a multi-seed run, research scrape, carry backtest.
- **session**: paper session, trading-hours session in a strategy, download session, agent session.
- **seed**: RNG seed, warm-start run, multi-seed run.
- **gate**: discovery gate, pre-review gate, live-consent and start checks.
- **validation**: the validation engine tier, cross-window checks, purged k-fold cross-validation, strategy-file checks.
- **promote**: champion to strategy, extraction to strategy, indicator scaffold to plugin, paper to live.
- **pooled**: carry's shared account, pooled trial count across seeds, pooled daily returns across symbols.
- **portfolio**: funding carry, portfolio symbol aggregation, portfolio-level risk limits.
- **maker**: the maker fee rate, the coding agent.
- **equity**: balance before unrealised PnL (in sizing) vs balance plus unrealised PnL (in margin limit and guard). Say which.
- **consistency**: screening–validation consistency vs the share of profitable WFA windows.
- **archive**: the market-data archive vs storing research items.
