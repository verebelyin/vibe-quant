# vibe-quant

A research-to-trading pipeline for crypto perpetual futures. It searches for strategies with a repeatable edge, makes them survive honest evidence (backtests, overfitting gates, paper trading), and only then trades them live.

## Strategies

**Strategy**:
A declarative set of trading rules: indicators, entry and exit conditions, stop-loss and take-profit, and time filters.
_Avoid_: strategy config, template, DSL config

**Strategy DSL**:
The YAML language strategies are written in.
_Avoid_: strategy format, schema

**Time exit**:
The optional `max_bars_in_trade: N` DSL field: close a position N strategy-timeframe bars after its entry fill, unless a stop, take-profit or exit condition closes it first.
_Avoid_: timeout exit, holding period

**Condition**:
One boolean rule over indicators or prices (e.g. "RSI below 30", "fast MA crosses above slow MA") that opens or closes a long or short.
_Avoid_: signal, rule

**Indicator**:
A named computation over market data that conditions refer to.

**Indicator plugin**:
An indicator added as a drop-in file instead of being built in.
_Avoid_: custom indicator

**Context indicator**:
An indicator that reads data other than the traded bars, such as funding, the BTC trend or taker flow (`TAKER_IMBALANCE`). Its value uses only data closed by the bar close and is never carried over a gap.

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
An exploration-only replay of screening for 1-minute, single-symbol genomes that needs no engine run; it must match screening fill for fill, and it never yields champions or gate inputs. Context indicators flagged as-of (today `TAKER_IMBALANCE`) replay too; the others run on the engine.
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
The venue's margin ratio: a position's margin is its notional / leverage. A run-level setting (whole number 1-125, default 10), never searched.

**Liquidation overlay**:
An opt-in model of exchange liquidation, either off, isolated margin or cross margin.
_Avoid_: liquidation mode (when you mean on/off)

**Sizing**:
How large each entry is: the smaller of the risk-based size and the max exposure.
_Avoid_: position size gene (sizing is never searched)

**Sizing mode**:
What the risk budget divides by: the stop-loss distance (`risk`, default) or ATR × multiplier (`atr`, volatility sizing).
_Avoid_: confidence sizing (nothing sizes by signal strength)

**Risk per trade**:
The fraction of balance a trade loses if its stop is hit.

**Max exposure**:
The largest notional a single position may hold, as a multiple of balance.
_Avoid_: position cap, exposure cap

**Margin limit**:
A ceiling on margin in use as a fraction of equity; an entry that would cross it is refused, never shrunk.
_Avoid_: cushion, total exposure cap

**Balance**:
The account's wallet balance without unrealised PnL; sizing reads it.

**Equity**:
Balance plus the unrealised PnL of open positions; the margin limit and the paper guard read it.

**Costed balance**:
In a backtest, the balance minus the modeled funding and slippage the engine never booked, paid through that instant; backtest sizing and the margin limit read it (costed equity likewise). Paper and live use the real balance.
_Avoid_: ledger equity (the post-hoc series), net balance

**Starting balance**:
The account balance a run starts with.
_Avoid_: bankroll, initial balance, capital

## Discovery and gates

**Discovery run**:
A genetic search over genomes that ends with zero or more champions.
_Avoid_: GA run, search run

**Genome**:
The searchable encoding of a strategy: its genes, stop and take-profit (or, with `--exit-genes extended`, exit rules), and direction.
_Avoid_: chromosome, individual

**Gene**:
One searchable piece of a genome, typically an indicator with a condition and threshold.

**Stop cap**:
The widest stop a discovery run draws when liquidation is off: 95% of the worst isolated liquidation distance over the run's coins and both sides at the run leverage (BTC/ETH at 10x: 9.08%), so a stop is never booked past the point the exchange would have liquidated. Applies to any liquidation-off chromosome run whose cap falls below 10 (10x and up for BTC/ETH/SOL/BNB, 9x for ZEC/HYPE/NEAR/SUI); liquidation on, bankroll grids and `--port` runs keep the 0.5-10% range, and a leverage whose cap is at or below 0.5% is refused.
_Avoid_: max stop, stop limit

**Exit gene**:
A searchable piece of a genome's stop, target or time exit (ATR or trailing stop, 50% safety stop, risk-reward target, no target, bars-in-trade limit). Drawn only under `--exit-genes extended`; otherwise a genome keeps its fixed-% stop and target.

**Population**:
The set of genomes alive in one generation.

**Generation**:
One round of evaluating, selecting and breeding the population.

**Immigrant**:
A random genome injected into the population to restore diversity.

**Fitness**:
The single number discovery ranks genomes by, blending risk-adjusted return, drawdown and profit factor minus penalties.
_Avoid_: score

**Fitness v2**:
The opt-in fitness mode (`--fitness v2`) that scores a genome by its shrunk mark-to-market t-stat times a profitable-window factor, then averages that score with the median score of its one-step **neighbours**.
_Avoid_: new fitness, fitness 2

**Plateau**:
The v2 step that re-scores the top 5% of each generation on up to 4 neighbours, so a broad ridge outranks a sharp peak; its neighbours count as DSR trials but never enter the effective-N series pool.

**Neighbour**:
A genome one step away from another on a single numeric gene (parameter or threshold), the unit the plateau step scores alongside its parent.

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

**Display Sharpe**:
Every Sharpe or Sortino on screen, in texts and logs (`sharpe365=`) and in typed limits (`--*-min-sharpe-365`, `cross_window_min_sharpe_365`), annualised with √365: stored × √(365/252) ≈ 1.2035 (`metrics.sharpe_to_display`). Stored values (DB, notes, gates, DSR, exactness pins) stay NT's √252. Not converted: the bootstrap "Sharpe" CI (a t-stat), DSR (a probability), the carry engine's weekly Sharpe (√52) and the shadow forward Sharpe (already √365).
_Avoid_: unqualified "annual Sharpe" in a text without its unit

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
_Avoid_: timing null (that keeps the champion's own trades and only moves them)

**Timing null**:
Discovery's opt-in gate (`--null-gate`): the champion's as-traded return must beat a set share (95 % by default) of random re-timings of its own trades, which keep each trade's side, size and holding time (trades of several coins that overlap in time move together as one block). When the champion is in a position more than 90 % of the time, re-timing changes almost nothing, so the null refuses it: timing untestable, not certified (it never passes, and the buy-and-hold gate does not decide for it).
_Avoid_: random baseline, permutation test (alone)

**Symbol aggregation**:
How a multi-symbol genome is scored: as one shared portfolio account, by its worst symbol, or pooled (`--symbol-agg pooled`): each coin on its own account, scored as ONE hypothesis on the equal-weight average of their daily mark-to-market returns. A pooled champion also needs (3N + 4) // 5 profitable coins (the sign floor); its max drawdown is of daily closes, so it does not compare with an intrabar one. Its screening replay runs the pooled scoring again; its validation is one shared account, labelled and not compared.
_Avoid_: pooled (alone; see Overloaded words)

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
Starting a session that skips risk carry-over from another trader id on the account; the trader id's own saved state always carries.

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

**Shadow tracker**:
Scores frozen strategies on the validation tier on days they never saw, after each data update the operator requests; it places no orders and never starts paper by itself.
_Avoid_: forward test, paper shadow

**Frozen strategy**:
A strategy enrolled in the shadow tracker: its DSL, compiler hash, launch parameters, venue, instrument spec and funding fallback rate are fixed at enrolment, and its clock starts at the next UTC midnight (the first forward day). Any changed input or any changed stored day retires it; re-enrolment makes a new row with a fresh clock.
_Avoid_: shadow strategy, locked strategy

**Pre-registration**:
The required paragraph written before a test or enrolment that fixes its hypothesis, rule and stopping limits (a shadow row also fixes its drawdown limit); its outcome is judged only against it.
_Avoid_: plan, notes

**Shadow evidence**:
The e-value of a frozen strategy's forward returns (weekly block sums, half-normal mixture, `vibe_quant/shadow/estat.py`): 1.0 until 26 full weeks, then it grows while the forward mean return is positive and shrinks otherwise; a slot passes at 100.
_Avoid_: confidence, score, evidence rating (that is extraction's)

**Evidence slot**:
One of the 5 places in shadow family F1, each tested at alpha 0.01 so the family's false-pass chance stays at most 0.05; a slot is spent when its row ends, and a row without a slot can show evidence but never pass.
_Avoid_: seat, quota

**Futility rule**:
The shadow tracker's drop rule: after 90 forward days the one-sided 95 % upper bound of the forward Sharpe is below 0.5, or the forward drawdown breaks the pre-registered limit, or the account busts.
_Avoid_: stop-loss, kill rule

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

## Portfolio strategies

**Portfolio strategy**:
A book over several coins in one pooled account, run by `vibe_quant/portfolio/` from a signal, a schedule and a weighting; funding carry and the trend book (the multi-coin volatility-targeted trend strategy of `--signal trend`) are portfolio strategies. It is not a DSL strategy.
_Avoid_: portfolio (alone), basket

**Funding carry**:
A weekly cross-sectional book that goes long the lowest-funding symbols and short the highest, in one pooled account; it is not a DSL strategy.
_Avoid_: portfolio (alone), carry strategy

**Fill granularity**:
The bar size a portfolio strategy fills and marks on: `1m` (catalog 1m LAST and MARK) or `1h` (archive 1h klines, the research tier).
_Avoid_: tier (alone), timeframe

**Research tier**:
Portfolio-strategy runs at fill granularity 1h; their results are labelled research only, never a verdict or paper evidence, and a PASS needs a 1m re-run.
_Avoid_: screening (another engine tier)

**Rotation null**:
The trend evaluator's random baseline: each draw circularly shifts every coin's signal sequence and the regime by one shared fraction inside each common-eligibility stretch (the coins eligible on those days shift together), keeping values, persistence and co-movement but breaking timing. `null_percentile` ranks the strategy's Sharpe against these draws.
_Avoid_: random null, shuffle

**Frozen family**:
The pre-registered set of trend arms (by arm hash) plus prior trials, fixed before any run; its size is the DSR trial count N, so a result does not depend on run order.
_Avoid_: trial ledger (funding carry's count)

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
The local Vercel Board (`vb`, inspector http://127.0.0.1:4100) that holds every agent message: job channels, knowledge channels, `needs-user`, `chat` and `lobby`.
_Avoid_: bus (alone)

**Job channel**:
The board channel `job/<job>` of one swarm job; it holds one thread per task.

**Thread**:
One task's conversation in its job channel: the chief's root post, the worker's `[progress]`, `[question]` and `[done]` replies, the chief's answers; a new round is a `[round N]` reply.
_Avoid_: DM, inbox

**Knowledge channel**:
A board channel `kb/<topic>` (gotchas, findings, decisions, model-notes, thoughts, self-improvement) for cross-job rules; a visible post carries label `state=live`.
_Avoid_: board topic

**Bus**:
Only `bus.py`, the chief's tool over the board, and the file store it reads with `SWARM_STORE=jsonl` (unit tests, offline chief view); `SWARM_BUS` still names the job's `bus/` directory.

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
- **pooled**: carry's shared account, pooled trial count across seeds, pooled daily returns across symbols, pooled symbol aggregation.
- **portfolio**: portfolio strategy (funding carry, the trend book), portfolio symbol aggregation, portfolio-level risk limits.
- **maker**: the maker fee rate, the coding agent.
- **equity**: means balance plus unrealised PnL; sizing reads balance (no unrealised PnL), so say balance there. Older code comments and API error texts still say equity for balance.
- **consistency**: screening–validation consistency vs the share of profitable WFA windows.
- **archive**: the market-data archive vs storing research items.
- **evidence**: shadow evidence (an e-value) vs an extraction's evidence rating vs evidence in general (backtests, gates, paper).
