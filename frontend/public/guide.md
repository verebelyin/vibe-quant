# User Guide

Welcome to vibe-quant, a crypto perpetual futures backtesting and trading engine. This guide walks you through the main workflows.

---

## 1. Creating a Strategy

Go to **Strategy Management** in the sidebar.

### Quick Start (Wizard)

Click **New Strategy** to open the creation dialog. You can start from a **blank strategy** or pick a **template**.

The wizard walks you through 6 steps:

1. **Name & Type** — Give it a name and choose a type:
   - **Momentum** — trade in the direction of recent price movement
   - **Mean Reversion** — bet on prices returning to averages
   - **Breakout** — enter when price breaks support/resistance
   - **Trend Following** — follow established trends
   - **Arbitrage** — exploit price differences
   - **Volatility** — trade volatility expansion/contraction

2. **Markets** — Pick a primary timeframe (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`) and one or more symbols (e.g., `BTCUSDT`, `ETHUSDT`).

3. **Indicators** — Add technical indicators. Available types:
   | Category | Indicator | Key Parameters |
   |----------|-----------|---------------|
   | Trend | SMA, EMA | period |
   | Momentum | RSI | period, overbought, oversold |
   | Momentum | MACD | fast, slow, signal periods |
   | Momentum | Stochastic | k_period, d_period |
   | Volatility | Bollinger Bands | period, std_dev |
   | Volatility | ATR | period |
   | Volume | VWAP | (none) |

   Each indicator can optionally use a different timeframe than the primary one.

4. **Entry & Exit Conditions** — Define when to open and close positions. Each condition compares two values (price, volume, or any indicator) using an operator (`>`, `<`, `>=`, `<=`, `==`, `crosses_above`, `crosses_below`). Chain multiple conditions with AND/OR logic. You can optionally split into separate long/short conditions.

5. **Risk Management** — Set stop loss, take profit, trailing stop, and position sizing. Quick presets available:
   - **Conservative**: 1% stop loss / 2% take profit
   - **Moderate**: 2% SL / 4% TP
   - **Aggressive**: 3% SL / 6% TP

   Position sizing methods: **Fixed size**, **Percent of equity**, or **Kelly Criterion**.

6. **Review** — Confirm everything and create.

### Editing a Strategy

Click any strategy card to open the full editor with tabs: **General**, **Indicators**, **Conditions**, **Risk**, **Time**, and **YAML**. The YAML tab gives you raw access to the strategy definition and syncs bidirectionally with the visual editor.

The **Time** tab lets you restrict trading to specific hours (UTC), days of the week, market sessions (Asian/European/US), and avoid funding rate windows.

---

## 2. Downloading Data from Binance

Go to **Data Management** in the sidebar.

1. Select one or more **symbols** (e.g., BTCUSDT, ETHUSDT)
2. Pick a **date range** and **interval** (1m, 5m, 15m, 1h, 4h)
3. Click **Preview** to see how many months need downloading vs. already archived
4. Click **Start Download** to begin

Other actions:
- **Update All** — refresh all existing data to the latest
- **Rebuild Catalog** — rebuild the Parquet data catalog from the SQLite archive

The **Data Browser** tab shows what data you already have and its quality.

---

## 3. Running a Sweep (Parameter Scan)

Go to **Backtest Launch** in the sidebar.

1. Select your **strategy**
2. Set mode to **Screening** (faster, simplified fills — good for scanning many parameter combos)
3. Enable **Parameter Sweep**
4. Configure the sweep using the **Sweep Builder**:
   - **Quick Scan** — 3 values per parameter (fast overview)
   - **Fine Grid** — 10+ values per parameter (thorough scan)
   - **Custom** — manually set each parameter as Fixed, Range (min/max/step), or List
5. Set **date range** (presets: 1M, 3M, 6M, 1Y, 2Y), **timeframe**, **initial balance**, and **leverage** (1-125x)
6. Select **symbols** and run **Preflight Check** to verify data coverage
7. Click **Launch**

The sweep runs every combination of your parameter values. Results appear on the **Results Analysis** page under the **Sweep** tab, including a **3D Pareto surface** chart showing the best tradeoffs between Sharpe ratio, drawdown, and return.

---

## 4. Running a Backtest

Same page (**Backtest Launch**), but:

1. Set mode to **Validation** (full-fidelity simulation with realistic fills and latency)
2. Optionally select a **Latency Preset** and **Sizing Config** (configured in Settings)
3. Enable **overfitting filters** if desired:
   - **Deflated Sharpe Ratio (DSR)** — adjusts for multiple testing
   - **Walk-Forward Analysis (WFA)** — splits data into in/out-of-sample windows (configurable number of splits)
   - **Purged K-Fold CV** — cross-validation with configurable purge embargo %
4. Launch the backtest

Typical workflow: run a **screening sweep** first to find promising parameters, then run a **validation backtest** on the best ones.

---

## 5. Checking Results & Re-running

Go to **Results Analysis** in the sidebar.

### Viewing Results

Select a run from the dropdown. You'll see:

- **Metrics Panel** — 12 key metrics: Total Return, Sharpe, Max Drawdown, Win Rate, Profit Factor, Total Trades, Avg Trade Duration, Expectancy, Sortino, Calmar, CAGR, Annual Volatility
- **Cost Breakdown** — fees, slippage, funding costs vs. gross/net PnL
- **Overfitting Badges** — pass/fail for DSR, WFA, K-Fold (if enabled)
- **Charts** — equity curve, drawdown, trade distribution, performance radar, rolling Sharpe, monthly heatmap, and more
- **Trade Log** — every individual trade with details
- **Sweep Analysis** (if sweep run) — scatter plots, Pareto surface, filterable results table

### Comparing Runs

Switch to **Compare** view to see two runs side by side.

### Re-running with Different Parameters

From the **Sweep Analysis** tab, each parameter combo row has a **Validate** button that launches a new validation backtest with those specific parameters. You can also go back to **Backtest Launch** and adjust settings manually.

### Exporting

Use the **Export** panel to download results as CSV.

---

## 6. Setting Latency for Binance

Go to **Settings** in the sidebar, then the **Latency** tab.

Latency presets are configured on the backend and displayed as read-only cards showing:
- Preset name (e.g., "Binance Co-located", "Binance Retail")
- Description of the scenario
- Base latency in milliseconds

To use a latency preset: go to **Backtest Launch**, set mode to **Validation**, and select the preset from the **Latency Preset** dropdown.

---

## 7. Bankroll / Position Sizing

Go to **Settings** in the sidebar, then the **Sizing** tab.

Create a sizing configuration by choosing a method:

| Method | Parameters |
|--------|-----------|
| **Fixed Fractional** | Risk per trade (default 2%), max leverage, max position % |
| **Kelly Criterion** | Win rate, avg win, avg loss, Kelly fraction, max leverage, max position % |
| **ATR-based** | Risk per trade, ATR multiplier, max leverage, max position % |

After creating a config, select it in **Backtest Launch** (validation mode) under the **Sizing Config** dropdown.

The **Risk** tab in Settings lets you create risk configurations with portfolio-level controls: max total exposure, max correlated positions, daily loss limit, and drawdown halt threshold.
