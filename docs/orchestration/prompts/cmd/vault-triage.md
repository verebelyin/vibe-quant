<!--
cmd brief: bulk triage of The-Quant-Trading-Vault strategy specs into JSONL.
Tier: fast (deepseek-v4.1-flash-fast). Batch ~20 files per brief (per-call overhead ~18k tokens).
Orchestrator fills: <VAULT>, <FILE LIST>, <OUT>, <INDICATORS> (live registry, see alpha-scout.md).
Checker: the jq/python check at the bottom, then a hand spot-check of ~5% against the source files.
Run read-only (no worktree needed): --cwd <VAULT>.
-->
You are classifying trading-strategy description files. Read each file listed below and write one JSON object per file to the output file. Edit no other file.

VAULT DIRECTORY: <VAULT>
FILES (relative to the vault directory):
<FILE LIST, one per line>

OUTPUT: append to <OUT>, one JSON object per line (JSONL), in the same order as FILES.

Our engine can only use these indicators (exact names):
<INDICATORS, e.g. ADAPTIVE_RSI ADX ATR BBANDS CCI DEMA DONCHIAN EMA FRAMA ICHIMOKU KAMA KC MACD MFI NATR OBV PRICE_POSITION RAMS ROC RSI SMA STOCH TEMA VIDYA VOLSMA VWAP WILLR WMA>
It trades ONE perpetual futures symbol at a time, from OHLCV candles and funding rates only, one position at a time, with optional stop-loss / take-profit.

Each object has exactly these keys:
{
  "file": "<relative path>",
  "name": "<strategy name from the file>",
  "language": "pine" | "javascript" | "python" | "mylanguage" | "cpp" | "unknown",
  "is_strategy": true if it describes rules that open and close positions; false for tools, utilities, indicators-only, tutorials about APIs,
  "category": "trend" | "mean_reversion" | "breakout" | "momentum" | "grid" | "arbitrage" | "market_making" | "other",
  "timeframe": "<timeframe stated in the file, e.g. 4h>" or null if not stated,
  "indicators": ["<indicator names used, uppercase, mapped to the engine list where they are the same thing, e.g. 'Bollinger Bands' -> BBANDS, 'Stochastic' -> STOCH>"],
  "entry_long": "<one sentence>" or null,
  "entry_short": "<one sentence>" or null,
  "exit": "<one sentence covering exits, stops, take-profits>" or null,
  "needs": [any of "multi_symbol", "order_book", "pyramiding", "grid_orders", "custom_math", "external_data"; empty list if none],
  "missing_indicators": ["<indicators used that are NOT in the engine list>"],
  "dsl_feasible": true only if is_strategy is true AND needs is empty AND missing_indicators is empty,
  "evidence": "<short quote from the file that shows the entry rule>"
}

Rules:
- Use only what the file says. If a field is not stated, use null (or [] for lists). Leave guesses out.
- "evidence" must be copied word-for-word from the file.
- Keep every string under 200 characters.

EXAMPLE LINE:
{"file":"strategies/RSI-Mean-Reversion.md","name":"RSI Mean Reversion","language":"pine","is_strategy":true,"category":"mean_reversion","timeframe":null,"indicators":["RSI","EMA"],"entry_long":"Buy when RSI(14) crosses above 30 while close is above EMA(200).","entry_short":null,"exit":"Close when RSI crosses above 70 or 2% stop-loss.","needs":[],"missing_indicators":[],"dsl_feasible":true,"evidence":"if ta.crossover(rsi, 30) and close > ema200"}

Before replying, run this check and fix any line it reports:
python3 -c "import json,sys; keys={'file','name','language','is_strategy','category','timeframe','indicators','entry_long','entry_short','exit','needs','missing_indicators','dsl_feasible','evidence'}; [print('BAD', i, set(json.loads(l))^keys) for i,l in enumerate(open('<OUT>')) if set(json.loads(l))!=keys]; print('checked')"
Expected: only the word "checked".

Do not run git commands. Your final reply is exactly one line: `DONE <OUT>` or `FAILED <one-line reason>`.
