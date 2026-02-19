# Feature Parity Audit: Streamlit vs React

Audit date: 2026-02-19

## Legend

- **Covered** = Feature exists with equivalent functionality
- **Partial** = Feature exists but missing sub-features
- **Missing** = Feature not implemented in React

---

## 1. Strategy Management

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Strategy list w/ search | Covered | `StrategyList` component |
| Show inactive toggle | Covered | Via `StrategyList` |
| Strategy cards (name, description, indicators) | Covered | `StrategyList` renders cards |
| New Strategy button | Covered | `StrategyCreateDialog` |
| Wizard mode | Covered | `StrategyWizard` component |
| Template selector (card grid) | Covered | Inside `StrategyCreateDialog` |
| Visual form editor (indicators, conditions, risk, time, sweep) | Covered | `StrategyEditor` |
| YAML editor w/ validation preview | Covered | `StrategyEditor` |
| Split editor (YAML + visual) | Covered | `StrategyEditor` |
| Delete/deactivate confirmation | Covered | `StrategyDeleteDialog` |
| File upload (YAML) | Covered | `StrategyEditor` |

## 2. Discovery

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Population size slider | Covered | `DiscoveryConfig` |
| Max generations slider | Covered | `DiscoveryConfig` |
| Mutation rate slider | Covered | `DiscoveryConfig` |
| Elite count slider | Covered | `DiscoveryConfig` |
| Tournament size | Covered | `DiscoveryConfig` |
| Convergence generations | Covered | `DiscoveryConfig` |
| Symbol multi-select | Covered | `DiscoveryConfig` |
| Timeframe select | Covered | `DiscoveryConfig` |
| Date range | Covered | `DiscoveryConfig` |
| Indicator pool display | Partial | Streamlit shows available indicators in expander; React may not |
| Start Discovery button | Covered | `DiscoveryConfig` |
| Active jobs w/ kill button | Covered | `DiscoveryJobList` |
| Generation progress bar | Covered | `DiscoveryProgress` |
| Fitness evolution chart (best/mean/worst) | Covered | `DiscoveryProgress` |
| Top strategies table (Rank, Sharpe, MaxDD, PF, etc.) | Covered | `DiscoveryResults` |
| Per-strategy detail expandable (metrics + YAML) | Covered | `DiscoveryResults` |
| Export to Strategy button | Covered | `DiscoveryResults` |
| DB fallback results display | Covered | `DiscoveryResults` |

## 3. Backtest Launch

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Strategy selector | Covered | `BacktestLaunchForm` |
| Symbol multi-select | Covered | `BacktestLaunchForm` |
| Timeframe selector | Covered | `BacktestLaunchForm` |
| Date range selector w/ presets | Partial | Date inputs exist, no presets (1M/3M/6M/1Y buttons) |
| Sweep params form (from DSL) | Covered | `SweepBuilder` component |
| **Overfitting filter toggles** | **Missing** | Streamlit has `render_overfitting_panel` with toggles for deflated sharpe, walk-forward, purged k-fold |
| Sizing config selector | Covered | Validation-only section |
| Risk config selector | Covered | Validation-only section |
| Latency preset selector | Covered | Validation-only section |
| Preflight summary | Covered | `PreflightStatus` |
| Run Screening / Run Validation buttons | Covered | `BacktestLaunchForm` |
| Active jobs list w/ status + kill | Covered | `ActiveJobsPanel` |
| Recent runs list | Covered | `ActiveJobsPanel` |
| Data coverage validation | Covered | Preflight check |

## 4. Results Analysis

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Run selector dropdown | Covered | `RunSelector` |
| Refresh runs button | Covered | `RunSelector` |
| Run details expander (strategy, mode, symbols, dates) | Partial | Run ID shown but no detail expander |
| Performance metrics (return, sharpe, sortino, calmar, etc.) | Covered | `MetricsPanel` - 12 metrics |
| **Win/Loss analysis panel** | **Missing** | Streamlit shows avg win, avg loss, largest win/loss, max consecutive W/L, payoff ratio |
| Cost breakdown (fees, funding, slippage, cost drag) | Covered | `CostBreakdown` |
| **Perpetual futures analytics (funding pie, gross/net P&L)** | **Missing** | Streamlit has dedicated `render_funding_impact` with pie chart |
| Overfitting filter results (badges) | Covered | `OverfittingBadges` |
| Equity curve chart | Covered | `ChartsPanel` > EquityCurveChart |
| Drawdown chart w/ top DD periods | Covered | `ChartsPanel` > DrawdownChart |
| **Rolling Sharpe chart** | **Missing** | Streamlit has `compute_rolling_sharpe` + chart |
| **Yearly returns bar chart** | **Missing** | Streamlit has `compute_yearly_returns` + chart |
| **Monthly returns heatmap** | **Missing** | Streamlit has `build_monthly_heatmap`. React has `HeatmapChart` component but not wired into results |
| **Long vs Short performance split** | **Missing** | Streamlit has `render_long_short_split` with trade count, win rate, P&L per direction |
| **Liquidation event summary** | **Missing** | Streamlit has `render_liquidation_summary` (count, total loss, % of trades) |
| Trade P&L distribution | Covered | `ChartsPanel` > TradeDistributionChart |
| Radar chart | Covered | `ChartsPanel` > PerformanceRadar |
| Trade scatter (ROI vs duration, size vs PnL) | **Missing** | Streamlit has two scatter plots |
| **Daily returns chart** | **Missing** | Streamlit has `compute_daily_returns` + chart |
| Trade log w/ filters (direction, symbol) | Covered | `TradeLog` with filters |
| Liquidation highlighting in trade log | Covered | `TradeLog` highlights liquidation rows |
| CSV export (trades) | Covered | `TradeLog` export button |
| CSV export (sweep results) | Partial | Sweep export not visible in `SweepAnalysis` |
| Sweep results view (Pareto, 3D, table, filters) | Covered | `SweepAnalysis` |
| Run validation from sweep candidate | **Missing** | Streamlit has button to save sweep params for validation |
| Raw NautilusTrader stats | **Missing** | Streamlit shows `raw_nt_stats` in expander |
| Tearsheet download (JSON) | Covered | `ExportPanel` |
| Notes section | Covered | `NotesPanel` |
| Comparison view (multi-run) | Covered | `ComparisonView` |

## 5. Paper Trading

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Trader ID input | Covered | `TraderInfo` shows trader ID |
| Start session (strategy selector) | Covered | `SessionControl` |
| Validated strategies filter (only completed validation) | Partial | React shows all strategies; Streamlit only shows those with completed validation backtests |
| Binance API key config | **Missing** | Streamlit has testnet toggle + API key/secret inputs in expander |
| Strategy validation metrics (Sharpe, MaxDD, Return) | Partial | React shows basic info; Streamlit shows metric cards for selected strategy |
| Testnet toggle | Covered | `SessionControl` checkbox |
| Sizing config (method, leverage, position %, risk/trade) | Covered | `SessionControl` form |
| Live P&L summary (balance, available, margin, total PnL) | Covered | `LiveDashboard` |
| Open positions table | Covered | `LiveDashboard` + `PositionsTable` |
| Pending orders table | Covered | `LiveDashboard` shows recent orders |
| Strategy status indicator (running/halted/paused) | Covered | `SessionControl` badge |
| HALT / RESUME / CLOSE ALL controls | Covered | `SessionControl` |
| Recent checkpoints timeline | Covered | `CheckpointsList` |
| Refresh button | Covered | Auto-refresh via polling |
| WebSocket live updates | Covered | `useWebSocket` hook in `LiveDashboard` |

## 6. Data Management

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Storage usage metrics (archive, catalog, total) | Covered | `DataStatusDashboard` |
| Data coverage table (symbols, dates, kline/bar counts) | Covered | `CoverageTable` |
| Symbol ingest form (multi-select, date range) | Covered | `IngestForm` |
| Download preview (archived vs new months) | Covered | `IngestForm` preview |
| Streaming download progress | Covered | `DownloadProgress` |
| Update recent data button | Covered | `IngestForm` update button |
| Rebuild catalog button | Covered | `IngestForm` rebuild button |
| **Data browser (OHLCV table + candlestick chart)** | **Missing from route** | Components exist (`DataBrowser`, `DataBrowserTab`) but NOT wired into `data.tsx` route |
| **Data quality verification** | **Missing from route** | Component exists (`DataQualityPanel`, `DataBrowserTab`) but NOT wired into `data.tsx` route |
| Download audit history | Covered | `DownloadHistory` |
| Supported symbols list | Partial | Shown in IngestForm; Streamlit has dedicated section |

## 7. Settings

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Sizing configs CRUD | Covered | `SizingTab` |
| Risk configs CRUD | Covered | `RiskTab` |
| Latency presets display | Covered | `LatencyTab` |
| Database path config | Covered | `DatabaseTab` |
| System info (NT version, Python, sizes) | Covered | `SystemTab` |

## 8. Navigation / Layout

| Streamlit Feature | React Status | Notes |
|---|---|---|
| Sidebar navigation grouped (Strategies, Backtesting, Trading, System) | Covered | Sidebar in `PageLayout` |
| Page title | Covered | Each route has title |
| Wide layout | Covered | Full-width layout |

---

## Summary

### Items needing fixes

1. **Data route missing DataBrowser + DataQuality** -- Components exist but aren't wired into `data.tsx`. Quick fix.
2. **Backtest overfitting filter toggles** -- Not in React launch form. Streamlit has toggles for deflated sharpe, walk-forward, purged k-fold with configurable params (WFE splits, purge embargo).
3. **Results: Win/Loss analysis panel** -- Avg win/loss, largest win/loss, consecutive W/L, payoff ratio, expectancy (expectancy exists in MetricsPanel but rest missing).
4. **Results: Long vs Short split** -- No direction-based performance breakdown.
5. **Results: Rolling Sharpe, Yearly Returns, Monthly Heatmap, Daily Returns charts** -- Four chart types missing from results.
6. **Results: Perpetual futures analytics** -- Funding pie chart, gross/net P&L breakdown.
7. **Results: Liquidation summary** -- Count, total loss, % of trades.
8. **Results: Trade scatter plots** -- ROI vs duration, size vs PnL.
9. **Results: Raw NT stats expander** -- JSON display of raw NautilusTrader statistics.
10. **Results: Run validation from sweep candidate** -- Button to promote sweep result to validation.
11. **Paper: Binance API key/secret inputs** -- Missing from React session start form.

### Quick fixes (can do now)

- Wire `DataBrowserTab` into `data.tsx` route (item 1)
