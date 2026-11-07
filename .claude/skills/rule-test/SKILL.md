---
name: rule-test
description: Test a simple trading rule (e.g. "15m RSI(14) crosses below 15, long next open, exit RSI >= 50") only through the shared engine path, and check a past journal rule against the engine. Use when the user asks "does the rule work", "does our backtester agree with this script", or wants a quick rule look.
---

Never write a private RSI, fill loop or cost model: the question is always "what does the ENGINE say". `scripts/research/*.py` are history, not tools.

## Steps

1. **Pre-register** in `docs/discovery-journal.md` before any run: rule, symbols, windows, the bar (trades, return, p), cost. Tune on one window, decide on a later one never seen; count every variant run (DSR N).
2. **Express the rule as a DSL YAML** (`.claude/agents/strategy-author.md`, `vibe_quant/dsl/schema.py`); put a `# rule → condition` comment on top. The DSL cannot omit a stop or target: use the widest (`stop_loss fixed_pct 50`, `take_profit fixed_pct 100`) and say so. Hold-N exit = `max_bars_in_trade`; cross = `crosses_below`/`crosses_above`.
3. **Run it:** `PYTHONPATH=$PWD .venv/bin/python scripts/agents/rule_test.py rule.yaml --symbols BTCUSDT --start 2020-01-01 --end 2026-10-21 --tier nt|fast [--leverage 1 --max-position-pct 1 --risk-per-trade 0.5]` (last line JSON; scratch DB; end date exclusive; exit 1, no JSON on a failed run or reversed dates). All-in 1× per trade = the three flags in brackets.
   - `nt` = screening runner, any timeframe (BTC 1m, 6.8 years: about 100 s MEASURED).
   - `fast` = 1m replay, about 10 s MEASURED, exact against `nt`. Use it only after `scripts/agents/fast_tier_parity.py --symbol BTCUSDT --start 2026-03-02 --end 2026-03-09 --n 12 --seed 1` exits 0 (paste its SUMMARY line); otherwise `nt` only. It refuses `max_bars_in_trade` (exit 2) and returns no win rate.
4. **Verdict** as a new dated journal entry: engine numbers copied verbatim and labelled "engine (rule_test, nt)"; never edit old entries.

## Engine facts that explain a mismatch with a private script

- **Trade count first.** If it differs, check indicator smoothing/warm-up, exits/occupancy (RSI14<15, BTC 2026-03-02..09: 37 trades, 43 with `max_bars_in_trade: 1` MEASURED), then execution/sizing.
- **RSI is Wilder since 2026-11-07** (= TradingView `ta.rsi`); older journal RSI used EMA.
- **No stop-and-reverse:** an entry is refused while a position is open; test a TradingView reversing rule as long half + short half.
- **Fills land at close[t+1]** (the signal bar's next bar close, ±1 tick), not open[t+1] (anti-look-ahead, `tests/unit/test_fill_timing.py`).
- Costs: taker 0.05%/side, modeled slippage and funding, not a flat 0.10%.
