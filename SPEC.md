# SPEC.md -- vibe-quant Implementation Specification

> Multi-phase implementation plan for a high-performance algorithmic trading engine
> for cryptocurrency perpetual futures, using NautilusTrader as the single backtesting
> and execution engine with two-tier fidelity (screening + validation), automated
> strategy discovery, and comprehensive overfitting prevention.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [Key Design Decisions](#3-key-design-decisions)
4. [Data Architecture](#4-data-architecture)
5. [Strategy DSL](#5-strategy-dsl)
6. [Screening Pipeline (NautilusTrader)](#6-screening-pipeline-nautilustrader)
7. [Validation Backtesting (NautilusTrader)](#7-validation-backtesting-nautilustrader)
8. [Overfitting Prevention](#8-overfitting-prevention)
9. [Risk Management](#9-risk-management)
10. [Streamlit Dashboard](#10-streamlit-dashboard)
11. [Paper Trading & Live Execution](#11-paper-trading--live-execution)
12. [Observability & Alerts](#12-observability--alerts)
13. [Testing Strategy](#13-testing-strategy)
14. [Phase 1: Foundation & Data Layer](#phase-1-foundation--data-layer)
15. [Phase 2: Strategy DSL & Screening Pipeline](#phase-2-strategy-dsl--screening-pipeline)
16. [Phase 3: Validation Backtesting & Risk](#phase-3-validation-backtesting--risk)
17. [Phase 4: Overfitting Prevention Pipeline](#phase-4-overfitting-prevention-pipeline)
18. [Phase 5: Streamlit Dashboard](#phase-5-streamlit-dashboard)
19. [Phase 6: Paper Trading & Alerts](#phase-6-paper-trading--alerts)
20. [Phase 7: Ethereal DEX Integration](#phase-7-ethereal-dex-integration)
21. [Phase 8: Automated Strategy Discovery (Genetic Optimization)](#phase-8-automated-strategy-discovery-genetic-optimization)
22. [Risk Assessment](#risk-assessment)
23. [Appendix: Interview Decision Log](#appendix-interview-decision-log)

---

## 1. Architecture Overview

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT DASHBOARD                              │
│  Launch backtests | Configure params | View results | Manage lifecycle  │
└────────┬─────────────────────────┬──────────────────────┬───────────────┘
         │                         │                      │
         ▼                         ▼                      ▼
┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  STRATEGY DSL   │  │  SCREENING PIPELINE  │  │  RESULTS & ANALYTICS │
│  (YAML/JSON)    │──▶  (NT Screening Mode) │──▶  Multi-metric ranking │
│  Indicators +   │  │  Parallel param sweep│  │  Pareto front        │
│  Conditions +   │  │  Simplified fills    │  │  SQLite storage      │
│  Actions        │  │  Full margin/funding │  │                      │
│  Multi-TF +     │  └──────────┬───────────┘  └──────────────────────┘
│  Time filters   │             │
└────────┬────────┘             ▼
         │            ┌──────────────────────┐
         │            │  OVERFITTING FILTERS │
         │            │  Walk-Forward        │
         │            │  Deflated Sharpe     │
         │            │  Purged K-Fold CV    │
         │            │  (all toggleable)    │
         │            └──────────┬───────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    NAUTILUSTRADER ENGINE (Validation Mode)               │
│  Event-driven backtesting | Realistic execution simulation              │
│  Custom FillModel | LatencyModel | Fees + Funding | Liquidation        │
├─────────────────────────────────────────────────────────────────────────┤
│  Position Sizing Module  │  Risk Management Module  │  Event Logger     │
│  (Kelly/FF/ATR)          │  (Strategy + Portfolio)   │  (Structured JSON)│
└────────┬────────────────────────┬───────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐  ┌──────────────────────────────────────────────────┐
│  BINANCE ADAPTER │  │  DATA LAYER                                      │
│  (Phase 1)       │  │  NautilusTrader ParquetDataCatalog               │
├──────────────────┤  │  Multi-TF bars: 1m, 5m, 15m, 1h, 4h             │
│  ETHEREAL ADAPTER│  │  Instruments + Funding Rates                     │
│  (Phase 7)       │  │  Raw data archive (SQLite) for reproducibility   │
└──────────────────┘  │  DuckDB query layer for research                 │
                      └──────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Single-engine, two-tier fidelity**: NautilusTrader for both screening (simplified execution, parallel parameter sweeps) and validation (realistic fills, latency, full cost modeling). One engine means leverage, funding rates, and liquidation are always modeled -- even during screening.
2. **Single source of truth**: Strategy logic defined once in a declarative DSL, auto-translated to NautilusTrader Strategy subclasses. DSL supports multi-timeframe conditions, time-based filters, and position management.
3. **Separation of concerns**: Strategy signals are decoupled from position sizing, risk management, and execution. Each is a pluggable module.
4. **Incremental complexity**: Start with manual strategies and parameter sweeps. Build toward automated strategy discovery with genetic optimization.
5. **Defense in depth**: Multi-layer overfitting prevention with toggleable filters. Dual-level risk circuit breakers (strategy + portfolio).
6. **Data reproducibility**: Raw downloaded historical data is archived in SQLite so backtests can be replicated from the original source data at any time.

---

## 2. Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Language** | Python 3.13 (via `uv`) | NautilusTrader compatibility, ecosystem |
| **Package Manager** | `uv` | Fast, deterministic, project standard |
| **Backtesting Engine** | NautilusTrader ~1.222.x | Rust core, realistic execution simulation, backtest-live parity, leverage/funding/liquidation |
| **Technical Indicators** | pandas-ta-classic + NautilusTrader built-in | Community-maintained fork with CI/CD, NautilusTrader Rust indicators for core TA |
| **Market Data Storage** | NautilusTrader ParquetDataCatalog | Native integration, DataFusion queries, Parquet files |
| **Raw Data Archive** | SQLite (separate DB) | Immutable archive of downloaded CSV/API data for reproducibility |
| **State Storage** | SQLite (WAL mode) | Strategy configs, backtest results, trade logs. Zero infrastructure |
| **Query Layer** | DuckDB | Ad-hoc analytics on Parquet + SQLite. Research queries |
| **Dashboard** | Streamlit + Plotly | Interactive UI, auto-generated forms, backtest management |
| **Alerts** | Telegram Bot API | Mobile push notifications for errors and circuit breakers |
| **Logging** | Structured JSON (stdlib logging) | Queryable event logs with DuckDB |
| **Testing** | pytest + hypothesis (optional) | Unit + integration, 80% coverage on core |
| **Containerization** | Docker via devcontainer | Reproducible development environment |

### NautilusTrader Version Policy

- Pin to `major.minor` (e.g., `nautilus_trader>=1.222.0,<1.223.0`)
- Accept patch releases automatically for bugfixes
- Test on update before merging version bumps
- No modifications to NautilusTrader source (LGPL-3.0 compliance)

### License Compatibility

NautilusTrader is licensed under **LGPL-3.0-or-later**. This project is MIT-licensed. LGPL-3.0 permits using NautilusTrader as a library dependency without requiring relicensing of this project, provided:
- No modifications are made to NautilusTrader source code
- NautilusTrader is used only as an unmodified library (imported, not vendored/modified)
- All custom code resides in separate modules within this project

This is the standard and accepted way to use LGPL libraries from MIT-licensed projects.

### What NautilusTrader Provides

- Rust core: high-throughput event processing (up to millions of rows/sec streaming), nanosecond-resolution simulation
- Binance USDT-M perpetual futures adapter (leverage, funding rates, hedge mode)
- L2 order book simulation (fills walk through price levels)
- Built-in `RiskEngine` (max notional, order rate limits, trading state control)
- `PositionSizer` base class for custom sizing algorithms
- `FillModel` base class for custom slippage/execution models
- `LatencyModel` for network latency simulation in backtesting
- `ParquetDataCatalog` with Apache DataFusion query engine
- Identical code path for backtest, sandbox, and live trading
- `FundingRateUpdate` as first-class data type (since v1.220.0)
- Built-in indicators (RSI, EMA, SMA, MACD, Bollinger Bands, ATR, etc.) implemented in Rust

### What We Build On Top

- Strategy DSL (Indicator + Condition + Action, multi-TF, time filters, position management)
- DSL-to-NautilusTrader Strategy code generator
- Screening pipeline with parallel parameter sweeps (multiprocessing)
- Overfitting prevention pipeline (Walk-Forward, DSR, Purged K-Fold)
- Position sizing modules (Kelly Criterion, Fixed Fractional, ATR-based)
- Portfolio-level risk management and circuit breakers
- Streamlit dashboard (lifecycle management)
- Telegram alert system
- Data ingestion pipeline (Binance bulk download) with raw data archival
- Ethereal DEX adapter (Phase 7)
- Genetic strategy discovery engine (Phase 8)

---

## 3. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backtesting engine | NautilusTrader (single engine) | Rust performance, execution realism critical at 20x leverage, backtest-live parity, always models leverage/funding/liquidation |
| Parameter screening | NautilusTrader screening mode + multiprocessing | Simplified fills for speed, but still models margin/funding/liquidation unlike vectorized alternatives which lack these features |
| Strategy format | Declarative DSL | Single source auto-translates to NautilusTrader Strategy, UI form generation |
| DSL expressiveness | Indicator + Condition + Action + Multi-TF + Time Filters + Position Mgmt | Covers most crypto perp futures strategies; multi-timeframe confirmation is essential |
| Technical indicators | pandas-ta-classic + NautilusTrader built-in | pandas-ta-classic is community-maintained with CI/CD; NautilusTrader has Rust-native indicators for core TA |
| Position sizing | Separate pluggable modules | Same strategy can run with different sizing without code changes |
| Data storage | NautilusTrader ParquetDataCatalog + raw data archive | Native NT integration; raw CSV/API data archived in SQLite for reproducibility |
| State storage | SQLite (WAL mode) | Zero infrastructure for single-user MVP, WAL enables concurrent read/write from dashboard + backtests |
| Dashboard | Streamlit | Full lifecycle management (launch, configure, results), auto-generated forms |
| Async backtests | Background subprocess + PID tracking + polling | No extra infra (no Redis/Celery), Streamlit stays responsive, cleanup on restart |
| Overfitting | Full pipeline, toggleable | Walk-Forward + DSR + Purged K-Fold, each can be enabled/disabled independently |
| Screening output | Multi-metric Pareto front | No single metric dominates; user sees trade-offs between Sharpe, drawdown, etc. |
| Risk management | Strategy + portfolio level | Defense in depth; strategy-level DD limits AND global portfolio halt |
| Order types | Market entries + limit SL/TP | Realistic taker fills on entry, maker fees on SL/TP. Common real-world pattern |
| Latency simulation | NautilusTrader LatencyModelConfig | Configurable per venue with presets (co-located, domestic, international, retail) |
| Multi-strategy | Single strategy initially | Avoid portfolio orchestration complexity at MVP; add later |
| Timeframes | Pre-computed multi-TF | 1m, 5m, 15m, 1h, 4h pre-built in catalog. Strategies subscribe to what they need |
| Data history | 2 years | Sufficient for WFA with meaningful window counts; manageable data volume |
| Data updates | Manual CLI script | Run when needed. No always-on infrastructure at MVP |
| Symbols | Start with 3 (BTC, ETH, SOL) | Validate end-to-end first, easy to expand |
| Deployment | Local dev machine | Everything in devcontainer. No cloud infrastructure at MVP |
| Error handling | Halt + alert | Cancel orders, log error, Telegram notification, wait for manual intervention |
| Promotion | Manual gates | Human reviews and explicitly promotes: backtest -> paper -> live |
| Alerts | Telegram bot | Most common in crypto trading, mobile push, easy setup |
| Testing | pytest unit + integration | 80% coverage on core (sizing, fees, slippage, DSL translation) |
| Ethereal | Phase 7 | Binance first. Ethereal adapter after core engine proves value |
| Auto-discovery | Phase 8, genetic/evolutionary | Manual DSL first -> auto params -> auto indicator combinations |

---

## 4. Data Architecture

### Storage Layout

```
data/
├── catalog/                          # NautilusTrader ParquetDataCatalog
│   └── data/
│       ├── bar/
│       │   ├── btcusdt-perp.binance/
│       │   │   ├── 1-MINUTE-LAST/
│       │   │   │   └── {start}-{end}.parquet
│       │   │   ├── 5-MINUTE-LAST/
│       │   │   │   └── {start}-{end}.parquet
│       │   │   ├── 15-MINUTE-LAST/
│       │   │   ├── 1-HOUR-LAST/
│       │   │   └── 4-HOUR-LAST/
│       │   ├── ethusdt-perp.binance/
│       │   │   └── ...
│       │   └── solusdt-perp.binance/
│       │       └── ...
│       ├── funding_rate_update/
│       │   └── btcusdt-perp.binance/
│       │       └── ...
│       └── instrument/
│           └── ...
├── archive/
│   └── raw_data.db                  # SQLite: immutable archive of downloaded raw data
├── state/
│   └── vibe_quant.db                # SQLite (WAL mode): strategies, results, configs
└── logs/
    └── events/                       # Structured JSON event logs
        └── {backtest_id}.jsonl
```

### Raw Data Archive

All downloaded historical data is stored in an immutable SQLite archive before processing into the ParquetDataCatalog. This enables:
- **Reproducibility**: Rebuild the catalog from raw data at any time (e.g., after NautilusTrader version changes)
- **Auditing**: Verify data integrity by comparing catalog against original source
- **Re-ingestion**: Re-process with different aggregation logic or new timeframes without re-downloading

```sql
-- Raw data archive schema (data/archive/raw_data.db)
CREATE TABLE raw_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,              -- 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
    interval TEXT NOT NULL,            -- '1m'
    open_time INTEGER NOT NULL,        -- Unix timestamp ms
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time INTEGER NOT NULL,
    quote_volume REAL,
    trade_count INTEGER,
    taker_buy_volume REAL,
    taker_buy_quote_volume REAL,
    source TEXT NOT NULL,              -- 'binance_vision' or 'binance_api'
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, interval, open_time)
);

CREATE TABLE raw_funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    funding_time INTEGER NOT NULL,     -- Unix timestamp ms
    funding_rate REAL NOT NULL,
    mark_price REAL,
    source TEXT NOT NULL,
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, funding_time)
);

CREATE INDEX idx_raw_klines_symbol_time ON raw_klines(symbol, interval, open_time);
CREATE INDEX idx_raw_funding_symbol_time ON raw_funding_rates(symbol, funding_time);
```

### Data Ingestion Pipeline

**Initial bulk load (3 symbols, 2 years, 1-minute):**

```
data.binance.vision (monthly CSV archives)
    → Download in parallel (BTC, ETH, SOL USDT-M perpetual klines)
    → Store raw CSV data into SQLite archive (raw_klines table)
    → Parse from archive → DataWrangler → NautilusTrader Bar objects
    → Write instruments first, then bars to ParquetDataCatalog
    → Aggregate 1m bars to 5m, 15m, 1h, 4h
    → Write aggregated bars to catalog
    → Download funding rate history → store in archive → write to catalog
```

**Estimated data volume:**
- 3 symbols x 2 years x 365 days x 1440 bars/day = ~3.15 million 1-minute bars
- Plus aggregated timeframes: ~700K additional bars
- Total: ~3.85M bars, approximately 40-80 MB compressed Parquet
- Raw data archive: ~150-250 MB SQLite

**Ongoing updates (manual CLI):**

```bash
# Fetch and append new candles since last update
python -m vibe_quant.data update --symbol BTCUSDT-PERP --since auto
```

Uses Binance REST API (`GET /fapi/v1/klines`) with rate limit management. Archives raw data first, then appends to catalog. Rebuilds higher timeframe aggregations.

**Data validation:**
- Detect gaps > expected bar interval
- Attempt gap-fill via REST API for missing periods
- Flag remaining gaps as data quality issues
- OHLC consistency checks (high >= max(open, close), low <= min(open, close), high >= low)
- Instrument listing date validation: reject data before a symbol's actual listing date on Binance (e.g., SOL-PERP listed later than BTC-PERP) to avoid survivorship bias

### Data Access Patterns

| Consumer | Access Method | Notes |
|----------|--------------|-------|
| NT screening mode | `BacktestDataConfig` with `catalog_path` | Simplified execution, parallel runs |
| NT validation mode | `BacktestDataConfig` with `catalog_path` | Full execution simulation |
| Research/ad-hoc queries | DuckDB pointed at catalog Parquet files | Full SQL on the same data |
| Dashboard | SQLite for results, DuckDB for data analytics | Separate concerns |
| Data rebuild | Read from raw_data.db archive → re-ingest to catalog | Reproducibility |

### SQLite Schema (State Database)

**Important:** All SQLite connections MUST enable WAL mode on open:

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
```

```sql
-- Strategy definitions (DSL configs)
CREATE TABLE strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    dsl_config JSON NOT NULL,              -- The full DSL YAML as JSON
    strategy_type TEXT,                     -- 'technical', 'statistical', 'composite'
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    is_active BOOLEAN DEFAULT 1,
    version INTEGER DEFAULT 1
);

-- Position sizing configurations (separate from strategies)
CREATE TABLE sizing_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    method TEXT NOT NULL,                   -- 'fixed_fractional', 'kelly', 'atr'
    config JSON NOT NULL,                   -- Method-specific parameters
    created_at TEXT DEFAULT (datetime('now'))
);

-- Risk management configurations (separate from strategies)
CREATE TABLE risk_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    strategy_level JSON NOT NULL,           -- Per-strategy DD limits
    portfolio_level JSON NOT NULL,          -- Global DD halt threshold
    created_at TEXT DEFAULT (datetime('now'))
);

-- Backtest runs (both screening and validation)
CREATE TABLE backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER REFERENCES strategies(id),
    sizing_config_id INTEGER REFERENCES sizing_configs(id),
    risk_config_id INTEGER REFERENCES risk_configs(id),
    run_mode TEXT NOT NULL,                 -- 'screening' or 'validation'
    symbols JSON NOT NULL,                  -- ["BTCUSDT-PERP", "ETHUSDT-PERP"]
    timeframe TEXT NOT NULL,                -- '1m', '5m', '1h'
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    parameters JSON NOT NULL,               -- Actual params used for this run
    latency_preset TEXT,                    -- 'co_located', 'domestic', 'international', 'retail', 'custom'
    status TEXT DEFAULT 'pending',           -- pending, running, completed, failed
    pid INTEGER,                            -- OS process ID for job management
    heartbeat_at TEXT,                      -- Last heartbeat timestamp
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Backtest results (one row per completed run)
CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,

    -- Performance metrics
    total_return REAL,
    cagr REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    max_drawdown REAL,
    max_drawdown_duration_days INTEGER,
    volatility_annual REAL,

    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate REAL,
    profit_factor REAL,
    avg_win REAL,
    avg_loss REAL,
    largest_win REAL,
    largest_loss REAL,
    avg_trade_duration_hours REAL,
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,

    -- Cost breakdown
    total_fees REAL,
    total_funding REAL,
    total_slippage REAL,

    -- Overfitting metrics
    deflated_sharpe REAL,
    walk_forward_efficiency REAL,          -- OOS return / IS return ratio
    purged_kfold_mean_sharpe REAL,

    -- Execution metadata
    execution_time_seconds REAL,
    starting_balance REAL,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Individual trades (for detailed analysis)
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,                -- 'LONG' or 'SHORT'
    leverage INTEGER DEFAULT 1,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    entry_fee REAL,
    exit_fee REAL,
    funding_fees REAL,
    slippage_cost REAL,
    gross_pnl REAL,
    net_pnl REAL,
    roi_percent REAL,
    exit_reason TEXT                        -- 'signal', 'stop_loss', 'take_profit', 'liquidation'
);

-- Sweep results (bulk storage for parameter sweeps from screening)
CREATE TABLE sweep_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    parameters JSON NOT NULL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    total_return REAL,
    profit_factor REAL,
    win_rate REAL,
    total_trades INTEGER,
    total_fees REAL,
    total_funding REAL,
    execution_time_seconds REAL,
    is_pareto_optimal BOOLEAN DEFAULT 0,
    passed_deflated_sharpe BOOLEAN,
    passed_walk_forward BOOLEAN,
    passed_purged_kfold BOOLEAN
);

-- Background job tracking for process management
CREATE TABLE background_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER UNIQUE REFERENCES backtest_runs(id) ON DELETE CASCADE,
    pid INTEGER NOT NULL,
    job_type TEXT NOT NULL,                 -- 'screening', 'validation', 'data_update'
    status TEXT DEFAULT 'running',          -- 'running', 'completed', 'failed', 'killed'
    heartbeat_at TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    log_file TEXT,                          -- Path to log file
    error_message TEXT
);

-- Indexes
CREATE INDEX idx_backtest_runs_strategy ON backtest_runs(strategy_id);
CREATE INDEX idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX idx_backtest_results_run ON backtest_results(run_id);
CREATE INDEX idx_trades_run ON trades(run_id);
CREATE INDEX idx_sweep_results_run ON sweep_results(run_id);
CREATE INDEX idx_sweep_results_pareto ON sweep_results(is_pareto_optimal);
CREATE INDEX idx_background_jobs_status ON background_jobs(status);
```

---

## 5. Strategy DSL

### Format: Indicator + Condition + Action + Multi-TF + Time Filters

Strategies are defined in YAML as a composition of layers with support for multi-timeframe conditions, time-based filters, and position management:

```yaml
# Example: RSI Mean Reversion Strategy with Multi-TF Confirmation
name: rsi_mean_reversion_mtf
description: "Enter on RSI extremes with EMA trend confirmation on higher TF"
version: 1

# Primary execution timeframe
timeframe: 5m

# Additional timeframes for cross-TF conditions
additional_timeframes: [1h, 4h]

indicators:
  # Primary timeframe (5m) indicators
  rsi:
    type: RSI
    period: 14
    source: close
    timeframe: 5m                # explicit, defaults to primary
  ema_fast:
    type: EMA
    period: 20
    source: close
  atr:
    type: ATR
    period: 14

  # Higher timeframe indicators
  rsi_1h:
    type: RSI
    period: 14
    source: close
    timeframe: 1h               # cross-timeframe indicator
  ema_trend_4h:
    type: EMA
    period: 50
    source: close
    timeframe: 4h               # higher TF trend

entry_conditions:
  long:
    - rsi < 30                    # 5m RSI oversold
    - rsi_1h > 40                 # 1h RSI not deeply oversold (confirmation)
    - close > ema_trend_4h        # Price above 4h EMA (uptrend)
  short:
    - rsi > 70                    # 5m RSI overbought
    - rsi_1h < 60                 # 1h RSI not deeply overbought
    - close < ema_trend_4h        # Price below 4h EMA (downtrend)

exit_conditions:
  long:
    - rsi > 50                    # RSI returned to neutral
  short:
    - rsi < 50

# Position management
position_management:
  scale_in:
    enabled: false                # MVP: no scaling in
  partial_exit:
    enabled: false                # MVP: full position exits only
  # Future: scale_in rules, partial_exit at targets

# Time-based filters
time_filters:
  allowed_sessions:
    - { start: "08:00", end: "20:00", timezone: "UTC" }  # Avoid low-volume overnight
  blocked_days: []                # e.g., ["Saturday", "Sunday"] for non-24/7 markets
  avoid_around_funding:
    enabled: true
    minutes_before: 5             # Don't enter 5 min before funding settlement
    minutes_after: 5              # Don't enter 5 min after funding settlement

stop_loss:
  type: atr_trailing              # 'fixed_pct', 'atr_fixed', 'atr_trailing'
  atr_multiplier: 2.0
  indicator: atr

take_profit:
  type: fixed_pct
  percent: 3.0                    # 3% take profit

# Parameter sweep ranges (for screening pipeline)
sweep:
  rsi.period: [7, 10, 14, 21, 28]
  rsi_oversold_threshold: [20, 25, 30, 35]
  rsi_overbought_threshold: [65, 70, 75, 80]
  ema_fast.period: [10, 15, 20, 25]
  ema_trend_4h.period: [30, 50, 80]
  stop_loss.atr_multiplier: [1.5, 2.0, 2.5, 3.0]
  take_profit.percent: [2.0, 3.0, 4.0, 5.0]
```

### Supported Indicator Types (MVP)

| Category | Indicators |
|----------|-----------|
| **Trend** | EMA, SMA, WMA, DEMA, TEMA, Ichimoku Cloud |
| **Momentum** | RSI, MACD, Stochastic, CCI, Williams %R, ROC |
| **Volatility** | ATR, Bollinger Bands, Keltner Channels, Donchian Channels |
| **Volume** | OBV, VWAP, Volume SMA, Money Flow Index |
| **Custom** | Extensible registry for user-defined indicators |

**Indicator mapping**: Each DSL indicator type maps to both a NautilusTrader built-in indicator class (Rust, used at runtime) and a pandas-ta-classic function (used for ad-hoc research). The NautilusTrader indicators are preferred at runtime for performance.

### Condition Operators

```
<indicator> > <value|indicator>    # Greater than
<indicator> < <value|indicator>    # Less than
<indicator> >= <value|indicator>   # Greater than or equal
<indicator> <= <value|indicator>   # Less than or equal
<indicator> crosses_above <value|indicator>  # Crossover
<indicator> crosses_below <value|indicator>  # Crossunder
<indicator> between <low> <high>   # Range check
close > <indicator>                # Price vs indicator (cross-TF)
close < <indicator>                # Price vs indicator (cross-TF)
```

Cross-timeframe conditions reference indicators by their declared name, regardless of timeframe. The code generator handles subscribing to the appropriate `BarType` for each timeframe.

### Multi-Timeframe Support

The DSL compiler generates a NautilusTrader Strategy that:
1. Subscribes to bars on all required timeframes (`subscribe_bars()` for each `BarType`)
2. Registers indicators on their respective bar types
3. On each primary-timeframe bar, evaluates all conditions using the latest values from all timeframes
4. Higher-timeframe indicator values update less frequently but are always available via `indicator.value`

### Time-Based Filter Implementation

Time filters are evaluated before condition checks in the `on_bar()` handler:
- **Session filters**: Compare current bar timestamp against allowed UTC windows
- **Day filters**: Skip signals on blocked days of the week
- **Funding avoidance**: Skip entry signals within N minutes of known funding settlement times (Binance: 00:00, 08:00, 16:00 UTC; Ethereal: every hour)

### Position Management (Future Extension)

The DSL schema includes `position_management` fields for forward-compatibility:
- **Scale-in**: Add to winning positions at defined levels (e.g., after N% move in favor)
- **Partial exit**: Close portion of position at intermediate targets before final TP
- MVP: Both disabled, full position entry and exit only. Infrastructure is in place for Phase 3+.

### Auto-Translation

The DSL compiler generates a NautilusTrader `Strategy` subclass with:
- `on_start()` that subscribes to the required `BarType`s (multi-TF) and registers indicators
- `on_bar()` that evaluates time filters, then conditions against current indicator values
- Order submission via `submit_order()` for entries and SL/TP
- Indicator calculations use NautilusTrader's built-in Rust indicators when available, with pandas-ta-classic as a fallback for exotic indicators

### Schema for UI Form Generation

Each strategy YAML includes type information for its parameters, enabling the Streamlit dashboard to auto-generate input forms:

```yaml
parameters_schema:
  rsi.period:
    type: integer
    min: 2
    max: 100
    default: 14
    description: "RSI lookback period"
  stop_loss.atr_multiplier:
    type: float
    min: 0.5
    max: 5.0
    step: 0.5
    default: 2.0
    description: "ATR multiplier for stop loss distance"
  time_filters.allowed_sessions:
    type: session_list
    default: [{ start: "00:00", end: "23:59", timezone: "UTC" }]
    description: "Allowed trading sessions (UTC)"
```

---

## 6. Screening Pipeline (NautilusTrader)

### Architecture

The screening pipeline uses NautilusTrader in a **simplified execution mode** for fast parameter sweeps. Unlike vectorized alternatives, this approach correctly models leverage, margin, funding rates, and liquidation even during screening -- critical for perpetual futures at 20x.

**Screening mode simplifications** (vs full validation):
- Basic `FillModel` with simple probabilistic slippage (no custom volume-based model)
- No `LatencyModel` (zero latency)
- Standard fee model (not custom)
- Bar-level data only (no order book)

**What screening mode still models** (the key advantage):
- Leverage and margin mechanics
- Funding rate payments (8-hourly for Binance)
- Liquidation risk at configured leverage
- Position sizing with configured method
- All DSL conditions including multi-TF and time filters

### Workflow

```
1. Load strategy DSL YAML
2. Compile DSL to NautilusTrader Strategy subclass
3. Build parameter grid from DSL sweep section
4. For each parameter combination:
   a. Configure BacktestRunConfig with screening-mode venue
   b. Run via BacktestNode
   c. Extract performance metrics
5. Parallelize across CPU cores (multiprocessing)
6. Compute Pareto front ranking
7. Apply overfitting filters (toggleable)
8. Store results in SQLite (sweep_results table)
9. Output top N candidates for validation
```

### Parallel Execution

```python
# Pseudo-code for the screening engine
from multiprocessing import Pool
from nautilus_trader.backtest.node import BacktestNode, BacktestRunConfig

def screen_strategy(dsl_config: dict, symbols: list, date_range: tuple) -> pd.DataFrame:
    # 1. Compile DSL to Strategy class
    strategy_cls, strategy_config_cls = dsl_compiler.to_nautilus(dsl_config)

    # 2. Build parameter grid
    param_grid = build_param_grid(dsl_config['sweep'])  # Cartesian product

    # 3. Create run configs for each parameter combo
    run_configs = []
    for params in param_grid:
        config = BacktestRunConfig(
            engine=BacktestEngineConfig(
                strategies=[ImportableStrategyConfig(
                    strategy_path=strategy_cls.__module__ + ":" + strategy_cls.__name__,
                    config_path=strategy_config_cls.__module__ + ":" + strategy_config_cls.__name__,
                    config=params,
                )],
            ),
            data=data_configs,
            venues=[screening_venue_config],  # Simplified execution
        )
        run_configs.append((params, config))

    # 4. Run in parallel across CPU cores
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(run_single_backtest, run_configs)

    # 5. Aggregate and rank
    results_df = pd.DataFrame(results)
    return results_df

def run_single_backtest(args):
    """Run one parameter combination. Executed in worker process."""
    params, config = args
    node = BacktestNode(configs=[config])
    raw_results = node.run()
    return extract_metrics(params, raw_results)
```

### Screening Venue Configuration

```python
screening_venue_config = BacktestVenueConfig(
    name="BINANCE",
    oms_type="NETTING",
    account_type="MARGIN",
    base_currency=None,
    starting_balances=["100_000 USDT"],
    default_leverage=Decimal("10"),
    leverages={"BTCUSDT-PERP.BINANCE": Decimal("20")},
    fill_model=ImportableFillModelConfig(
        fill_model_path="vibe_quant.backtesting.fill_models:ScreeningFillModel",
        config_path="vibe_quant.backtesting.fill_models:ScreeningFillModelConfig",
        config={"prob_fill_on_limit": 0.8, "prob_slippage": 0.5},
    ),
    # No latency model for screening (speed)
    # Standard fee model
)
```

### Performance Expectations

NautilusTrader's Rust core provides high-throughput event processing. For a 2-year, 5-minute backtest:
- ~210k bars per symbol → completes in ~2-3 seconds per run
- With multiprocessing on 8 cores: ~500-1000 parameter combos in 10-15 minutes
- Recommended sweep size: 100-500 combinations per screening run
- Larger explorations use the genetic optimizer (Phase 8) instead of exhaustive grid search

### Multi-Metric Pareto Front Ranking

After screening, candidates are ranked using Pareto optimality across multiple objectives:

**Objectives (all maximized, limited to 3 for tractable Pareto front):**
- Sharpe Ratio
- 1 - Max Drawdown (inverted so higher = better)
- Profit Factor

**Constraints (hard filters before ranking):**
- Minimum 50 trades (avoids low-sample-size flukes)
- Max drawdown < 30% (survivability threshold at configured leverage)
- Profit factor > 1.0 (must be net profitable)

**Pareto front**: A candidate is Pareto-optimal if no other candidate is better in ALL objectives simultaneously. With 3 objectives, the Pareto front is typically ~10-20% of candidates for populations of 100-500; for larger sweeps (1000+), the fraction shrinks. Monitor front size and tighten hard filters if it grows too large.

The dashboard visualizes the Pareto front as a scatter plot matrix (e.g., Sharpe vs Max DD, with color = Profit Factor).

---

## 7. Validation Backtesting (NautilusTrader)

### Configuration

Top candidates from screening are validated through NautilusTrader's full-fidelity execution simulation with realistic fills, latency, and cost modeling.

**Venue configuration:**

```python
venue_config = BacktestVenueConfig(
    name="BINANCE",
    oms_type="NETTING",
    account_type="MARGIN",
    base_currency=None,
    starting_balances=["100_000 USDT"],
    default_leverage=Decimal("10"),
    leverages={"BTCUSDT-PERP.BINANCE": Decimal("20")},
    fill_model=custom_fill_model,          # Custom volume-based slippage
    fee_model=BinanceFeeModel(),           # Maker 0.02%, Taker 0.04% (note: verify current rates; Binance adjusts fees by VIP tier and promotions -- consider making configurable in DSL)
    latency_model=latency_model,           # Network latency simulation
)
```

### Latency Model Configuration

NautilusTrader's `LatencyModelConfig` simulates realistic network delays for order operations. Configurable per venue with presets:

```python
# Latency presets
LATENCY_PRESETS = {
    "co_located": LatencyModelConfig(
        base_latency_nanos=1_000_000,          # 1ms
        insert_latency_nanos=500_000,          # 0.5ms
        update_latency_nanos=500_000,          # 0.5ms
        cancel_latency_nanos=500_000,          # 0.5ms
    ),
    "domestic": LatencyModelConfig(
        base_latency_nanos=20_000_000,         # 20ms
        insert_latency_nanos=10_000_000,       # 10ms
        update_latency_nanos=10_000_000,       # 10ms
        cancel_latency_nanos=10_000_000,       # 10ms
    ),
    "international": LatencyModelConfig(
        base_latency_nanos=100_000_000,        # 100ms
        insert_latency_nanos=50_000_000,       # 50ms
        update_latency_nanos=50_000_000,       # 50ms
        cancel_latency_nanos=50_000_000,       # 50ms
    ),
    "retail": LatencyModelConfig(
        base_latency_nanos=200_000_000,        # 200ms
        insert_latency_nanos=100_000_000,      # 100ms
        update_latency_nanos=100_000_000,      # 100ms
        cancel_latency_nanos=100_000_000,      # 100ms
    ),
}
```

The dashboard exposes latency preset selection as a dropdown, with a "custom" option for manual nanosecond configuration.

### Execution Simulation

**Order types used:**
- **Entry**: Market orders (taker fee, filled at next bar open + slippage + latency delay)
- **Stop Loss**: Stop-market orders (triggered at mark price, filled with slippage)
- **Take Profit**: Limit orders (maker fee, filled at limit price with queue position probability)

**Fill model (custom):**
Custom fill model loaded via `ImportableFillModelConfig` (specifying `fill_model_path` and `config_path` to a subclass of `FillModel`):
- Market orders: Fill at bar open price + volume-based slippage
- Slippage formula: `spread/2 + 0.1 * volatility * sqrt(order_size / avg_volume)`
- Stop orders: Fill at stop price + 1-tick slippage (pessimistic)
- Limit orders: Fill at limit price if price touches (probabilistic fill for queue position)
- OCO handling: When both SL and TP could trigger on same bar, SL triggers first (pessimistic)

**Funding rate simulation:**
- Subscribe to `FundingRateUpdate` data (stored in catalog)
- Binance: Applied every 8 hours (00:00, 08:00, 16:00 UTC)
- Positive rate: long positions pay, short positions receive
- Negative rate: short positions pay, long positions receive
- Accrual: `position_notional * funding_rate` per period

**Liquidation simulation:**
- Isolated margin: `Liq_Price(long) = entry * (1 - 1/leverage + MMR)`
- Maintenance margin rate: Tiered by position size (Binance schedule)
- When mark price crosses liquidation price: position force-closed, remaining margin lost

### Strategy Code Generation

The DSL compiler generates a NautilusTrader `Strategy` subclass:

```python
# Auto-generated from DSL YAML
class RSIMeanReversionMTFStrategy(Strategy):
    def __init__(self, config: RSIMeanReversionMTFConfig):
        super().__init__(config)
        self.rsi_period = config.rsi_period
        self.ema_fast_period = config.ema_fast_period
        # ... etc

    def on_start(self):
        # Primary timeframe bars
        self.bar_type_5m = BarType.from_str(
            f"{self.config.instrument_id}-5-MINUTE-LAST-EXTERNAL"
        )
        self.subscribe_bars(self.bar_type_5m)

        # Higher timeframe bars
        self.bar_type_1h = BarType.from_str(
            f"{self.config.instrument_id}-1-HOUR-LAST-EXTERNAL"
        )
        self.subscribe_bars(self.bar_type_1h)

        self.bar_type_4h = BarType.from_str(
            f"{self.config.instrument_id}-4-HOUR-LAST-EXTERNAL"
        )
        self.subscribe_bars(self.bar_type_4h)

        # Register indicators on their respective bar types
        self.rsi = RSI(self.rsi_period)
        self.register_indicator_for_bars(self.bar_type_5m, self.rsi)

        self.rsi_1h = RSI(14)
        self.register_indicator_for_bars(self.bar_type_1h, self.rsi_1h)

        self.ema_trend_4h = ExponentialMovingAverage(self.ema_trend_period)
        self.register_indicator_for_bars(self.bar_type_4h, self.ema_trend_4h)

        # Subscribe to funding rates
        self.subscribe_funding_rates(self.config.instrument_id)

    def on_bar(self, bar: Bar):
        if not self.indicators_initialized():
            return

        # Only act on primary timeframe bars
        if bar.bar_type != self.bar_type_5m:
            return

        # Time filter: check allowed sessions
        if not self._is_in_allowed_session(bar.ts_event):
            return

        # Time filter: avoid around funding settlement
        if self._is_near_funding_settlement(bar.ts_event):
            return

        # Entry conditions (from DSL, multi-TF)
        if (self.rsi.value < 30
            and self.rsi_1h.value > 40
            and bar.close > self.ema_trend_4h.value):
            if not self.has_open_position():
                self._enter_long(bar)

        # Exit conditions (from DSL)
        elif self.rsi.value > 50 and self.has_long_position():
            self._exit_position(bar)

        # ... short conditions similarly
```

---

## 8. Overfitting Prevention

### Filter Pipeline (All Toggleable)

```
Screening sweep results (100-500 candidates)
    │
    ├── [Toggle] Deflated Sharpe Ratio Filter
    │   Removes candidates whose Sharpe is explainable by number of trials
    │
    ├── [Toggle] Walk-Forward Analysis
    │   Validates out-of-sample performance across sliding windows
    │
    ├── [Toggle] Purged K-Fold Cross-Validation
    │   Cross-validates with embargo periods to prevent leakage
    │
    ▼
Validated candidates → Full Validation Backtesting
```

### Deflated Sharpe Ratio (DSR)

Corrects the Sharpe ratio for multiple testing bias. When you test N parameter combinations, the expected maximum Sharpe ratio by pure chance increases with N.

```
DSR = Φ[(SR_hat - SR_0) * sqrt(T-1) / sqrt(1 - γ₃·SR_hat + ((γ₄-1)/4)·SR_hat²)]
```
Where `SR_hat` = observed Sharpe, `SR_0` = expected max Sharpe under null (depends on N_trials), `γ₃` = skewness, `γ₄` = kurtosis, `T` = sample size. See Bailey & Lopez de Prado (2014).

**Implementation:**
- Input: observed Sharpe ratio, number of trials (from screening sweep), return distribution moments (skew, kurtosis), sample size
- Output: probability that the observed Sharpe is genuinely positive after accounting for multiple testing
- Filter threshold: DSR > 0.95 (95% confidence the Sharpe is not a fluke)

### Walk-Forward Analysis (WFA)

Sliding window train/test validation:

```
|----Train----|--Test--|
      |----Train----|--Test--|
            |----Train----|--Test--|
```

**Configuration (optimized for 2-year data):**
- Training window: 9 months (configurable)
- Test window: 3 months (configurable)
- Step size: 1 month (configurable)
- Minimum windows: 8
- With 2 years of data (24 months): produces ~13 windows (floor((24-9-3)/1) + 1 = 13)

**Metrics:**
- Walk-forward efficiency: `mean(OOS_return) / mean(IS_return)` -- should be > 0.5
- OOS Sharpe: Sharpe ratio computed only on out-of-sample periods
- Consistency: percentage of OOS windows that are profitable

**Filter threshold:** Walk-forward efficiency > 0.5 AND > 50% of OOS windows profitable

### Purged K-Fold Cross-Validation

Standard K-Fold CV with embargo (purge) periods between train and test sets to prevent information leakage through overlapping indicator lookback windows.

**Configuration:**
- K = 5 folds (configurable)
- Purge period: max indicator lookback period (e.g., 50 bars for EMA-50)
- Embargo: additional gap after test set before next train set

**Metrics:**
- Mean OOS Sharpe across all folds
- Standard deviation of OOS Sharpe (consistency)

**Filter threshold:** Mean OOS Sharpe > 0.5 AND std(OOS Sharpe) < 1.0

### Pipeline Configuration

```yaml
overfitting:
  deflated_sharpe:
    enabled: true
    confidence_threshold: 0.95
  walk_forward:
    enabled: true
    train_window_days: 270            # 9 months
    test_window_days: 90              # 3 months
    step_days: 30                     # 1 month
    min_efficiency: 0.5
    min_profitable_windows_pct: 0.5
  purged_kfold:
    enabled: true
    n_splits: 5
    purge_bars: 50                    # auto-detected from strategy indicators
    min_mean_sharpe: 0.5
    max_sharpe_std: 1.0
```

---

## 9. Risk Management

### Architecture: Separate Pluggable Modules

Risk management is decoupled from strategy logic. Three module types attach to any strategy at deployment time:

#### 1. Position Sizing Module

```yaml
# sizing_config.yaml
method: fixed_fractional
config:
  risk_per_trade: 0.02            # 2% of equity per trade
  max_position_pct: 0.10          # Max 10% in single position
  max_leverage: 20
```

**Available methods:**

| Method | Formula | Use Case |
|--------|---------|----------|
| Fixed Fractional | `position = (equity * risk_pct) / (entry - stop_loss)` | Conservative, consistent risk |
| Kelly Criterion | `f* = (W - (1-W)/R) * kelly_fraction` | Optimal growth, uses half-Kelly (0.5) |
| ATR-Based | `position = (equity * risk_pct) / (ATR * multiplier)` | Volatility-adaptive sizing |

All methods enforce:
- Maximum position size as % of equity
- Maximum leverage limit
- Maximum portfolio heat (total risk across all positions)

#### 2. Strategy-Level Risk Module

```yaml
# strategy_risk_config.yaml
strategy_level:
  max_drawdown_scale_pct: 0.10    # Scale to 50% size at 10% DD
  max_drawdown_halt_pct: 0.15     # Halt strategy at 15% DD
  max_daily_loss_pct: 0.02        # 2% daily loss limit
  max_consecutive_losses: 10      # Halt after 10 consecutive losses
  cooldown_after_halt_hours: 24   # Wait 24h before resuming
```

#### 3. Portfolio-Level Risk Module

```yaml
# portfolio_risk_config.yaml
portfolio_level:
  max_portfolio_drawdown_pct: 0.20    # Global halt at 20% portfolio DD
  max_portfolio_daily_loss_pct: 0.05  # 5% daily portfolio loss limit
  max_total_exposure_pct: 0.50        # Max 50% of equity in positions
  max_correlated_exposure_pct: 0.30   # Max 30% in correlated positions
```

### Circuit Breaker Flow

```
New order signal from strategy
    │
    ├── Strategy-level check: Is this strategy halted? Daily loss exceeded?
    │   → If halted: reject order, log reason
    │
    ├── Position sizing: Calculate size based on configured method
    │   → Apply drawdown scaling factor
    │
    ├── Portfolio-level check: Total exposure OK? Portfolio DD OK?
    │   → If breached: reject order, send Telegram alert
    │
    ├── NautilusTrader RiskEngine: Max notional, order rate limits
    │   → Built-in checks
    │
    └── Submit order to execution engine
```

---

## 10. Streamlit Dashboard

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  vibe-quant    [Strategies] [Backtests] [Results] [Data] [Settings]│
├─────────────────────────────────────────────────────────────────────┤

Tab 1: Strategies
├── Strategy list (from SQLite)
├── Create/edit strategy (auto-generated form from DSL schema)
├── DSL YAML editor (raw mode toggle)
├── Multi-TF indicator configuration
├── Time filter configuration
└── Strategy version history

Tab 2: Backtests
├── Launch screening sweep
│   ├── Select strategy
│   ├── Select symbols (BTC, ETH, SOL)
│   ├── Select timeframe
│   ├── Configure parameter sweep ranges (auto-form)
│   ├── Toggle overfitting filters (checkboxes)
│   └── [Run Screening] button → background subprocess
├── Launch validation backtest
│   ├── Select candidates from sweep results
│   ├── Configure sizing module
│   ├── Configure risk module
│   ├── Select latency preset (co-located / domestic / international / retail / custom)
│   └── [Run Validation] button → background subprocess
└── Active jobs (status polling, progress bars, kill button)

Tab 3: Results
├── Sweep results table (sortable, filterable)
├── Pareto front scatter plot (Sharpe vs MaxDD, color=PF)
├── Individual backtest detail
│   ├── Equity curve (Plotly line chart)
│   ├── Drawdown chart (filled area, red)
│   ├── Monthly returns heatmap
│   ├── Trade distribution histogram
│   ├── Trade log table (entry, exit, P&L, fees, funding, reason)
│   └── Key metrics: Return, Sharpe, Sortino, Calmar, MDD, WR, PF
├── Strategy comparison (side-by-side metrics)
└── Overfitting filter results (which filters passed/failed)

Tab 4: Data
├── Catalog status (symbols, date ranges, row counts)
├── Raw data archive status (row counts, date ranges, last download)
├── Data update trigger (manual fetch button)
├── Data rebuild trigger (rebuild catalog from raw archive)
└── Data quality checks (gaps, anomalies)

Tab 5: Settings
├── Sizing module presets (create/edit)
├── Risk module presets (create/edit)
├── Latency model presets (view/customize)
├── Overfitting filter configuration
├── Telegram alert configuration
└── NautilusTrader version info
```

### Background Job Management

Backtests run as background subprocesses with PID tracking and heartbeat monitoring:

```python
import subprocess
import os
import signal

class BacktestJobManager:
    def __init__(self, state_manager: StateManager):
        self.state = state_manager

    def launch_screening(self, config: dict) -> int:
        """Launch screening sweep as subprocess. Returns run ID."""
        run_id = self.state.create_backtest_run(config)
        log_path = f"data/logs/{run_id}.log"
        proc = subprocess.Popen(
            ["python", "-m", "vibe_quant.screening.run", str(run_id)],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        self.state.register_job(run_id, proc.pid, "screening", log_path)
        return run_id

    def launch_validation(self, config: dict) -> int:
        """Launch validation backtest as subprocess. Returns run ID."""
        run_id = self.state.create_backtest_run(config)
        log_path = f"data/logs/{run_id}.log"
        proc = subprocess.Popen(
            ["python", "-m", "vibe_quant.validation.run", str(run_id)],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        self.state.register_job(run_id, proc.pid, "validation", log_path)
        return run_id

    def get_job_status(self, run_id: int) -> dict:
        """Poll job status from SQLite."""
        return self.state.get_backtest_run(run_id)

    def kill_job(self, run_id: int):
        """Kill a running job by PID."""
        job = self.state.get_job(run_id)
        if job and job['status'] == 'running':
            try:
                os.kill(job['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.state.update_job_status(run_id, 'killed')

    def cleanup_stale_jobs(self):
        """Called on dashboard startup. Marks jobs with dead PIDs as failed."""
        running_jobs = self.state.get_running_jobs()
        for job in running_jobs:
            try:
                os.kill(job['pid'], 0)  # Check if process exists
            except ProcessLookupError:
                self.state.update_job_status(job['run_id'], 'failed',
                    error='Process died unexpectedly (detected on startup)')
```

**Heartbeat protocol**: Background processes update `heartbeat_at` in SQLite every 30 seconds. Dashboard marks jobs as stale if no heartbeat for 120 seconds.

Streamlit polls SQLite every 2 seconds for status updates. The subprocess updates SQLite rows as it progresses.

---

## 11. Paper Trading & Live Execution

### Paper Trading Architecture (Phase 6)

NautilusTrader's `TradingNode` connects to live Binance WebSocket feeds but executes against a simulated matching engine:

```python
node = TradingNode(config=TradingNodeConfig(
    trader_id="PAPER-001",
    data_clients={
        "BINANCE": BinanceDataClientConfig(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
            account_type=BinanceAccountType.USDT_FUTURES,
            testnet=True,  # Use Binance testnet
        ),
    },
    exec_clients={
        "BINANCE": BinanceExecClientConfig(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
            account_type=BinanceAccountType.USDT_FUTURES,
            testnet=True,  # Simulated execution via testnet
        ),
    },
    strategies=[strategy_config],
))
```

**Key differences from backtesting:**
- Real-time WebSocket data (not historical replay)
- Execution against Binance testnet (simulated fills with real order book)
- State persisted to SQLite (positions, orders, balance) for crash recovery
- Structured event log written in real-time
- Telegram alerts active for errors and circuit breakers

### Live Execution (Phase 6+)

Same code as paper trading with `testnet=False` and real API credentials. Promotion is manual:

```
Backtest (pass overfitting filters)
    → Manual review of results
        → Paper trading (N days/weeks)
            → Manual review of paper vs backtest divergence
                → Live trading (small size first)
```

### Error Handling

On any unexpected error during paper/live:
1. Cancel all open orders for the affected strategy
2. Log the full error with stack trace (structured JSON)
3. Send Telegram alert with error summary
4. Set strategy state to `HALTED`
5. Wait for manual intervention (operator reviews, fixes, resumes)

Transient errors (WebSocket disconnect, API timeout) are handled by NautilusTrader's built-in reconnection with exponential backoff. Strategy-level errors (invalid signal, sizing calculation failure) trigger the halt-and-alert flow.

---

## 12. Observability & Alerts

### Structured Event Logging

Every significant event is logged as a JSON line to `logs/events/{run_id}.jsonl`:

```json
{"ts": "2025-01-15T10:30:00.123Z", "event": "SIGNAL", "strategy": "rsi_mean_reversion_mtf", "symbol": "BTCUSDT-PERP", "direction": "LONG", "context": {"rsi": 28.5, "rsi_1h": 45.2, "ema_trend_4h": 42100}}
{"ts": "2025-01-15T10:30:00.125Z", "event": "TIME_FILTER", "result": "PASS", "session": "08:00-20:00 UTC", "funding_check": "OK"}
{"ts": "2025-01-15T10:30:00.126Z", "event": "ORDER", "order_id": "O-001", "type": "MARKET", "side": "BUY", "quantity": 0.05, "sizing_method": "fixed_fractional", "risk_pct": 0.02}
{"ts": "2025-01-15T10:30:00.200Z", "event": "FILL", "order_id": "O-001", "fill_price": 42150.5, "slippage_bps": 1.2, "fee": 0.842, "latency_ms": 20}
{"ts": "2025-01-15T10:30:00.201Z", "event": "POSITION_OPEN", "symbol": "BTCUSDT-PERP", "side": "LONG", "entry_price": 42150.5, "quantity": 0.05, "leverage": 10, "initial_margin": 210.75, "liquidation_price": 38150.2}
{"ts": "2025-01-15T10:30:00.202Z", "event": "RISK_CHECK", "check": "portfolio_exposure", "current_pct": 0.12, "limit_pct": 0.50, "result": "PASS"}
```

Event logs are queryable with DuckDB:

```sql
SELECT * FROM read_json_auto('logs/events/42.jsonl')
WHERE event = 'FILL'
ORDER BY ts;
```

### Telegram Alerts

**Alert types:**
- Circuit breaker triggered (strategy halt, portfolio halt)
- Unexpected error (API failure, execution error)
- Paper/live trade executed (optional, configurable)
- Daily P&L summary (optional)

**Implementation:** Python `telegram` library, bot token stored in environment variable.

---

## 13. Testing Strategy

### Test Pyramid

```
Unit Tests (fast, many)
├── DSL parser and validator
├── DSL multi-TF condition parser
├── DSL time filter parser
├── DSL-to-NautilusTrader code generator
├── Position sizing calculations (Kelly, FF, ATR)
├── Fee calculations (maker/taker with various tiers)
├── Slippage model calculations
├── Funding rate accrual
├── Liquidation price calculations
├── Pareto front ranking algorithm
├── Deflated Sharpe Ratio calculation
├── Walk-Forward window generation
├── Purged K-Fold split generation
├── Risk circuit breaker logic
├── Latency preset configuration
└── Raw data archive CRUD

Integration Tests (slower, fewer)
├── DSL YAML → NautilusTrader Strategy → screening backtest (end-to-end)
├── DSL YAML → NautilusTrader Strategy → validation backtest (end-to-end)
├── Screening vs validation results directional consistency (>90% same signals)
├── Data ingestion → raw archive → catalog write → catalog read round-trip
├── SQLite operations (CRUD strategies, results, trades)
├── Background subprocess launch, heartbeat, and status polling
├── Process cleanup on stale job detection
└── Overfitting pipeline (all filters) on synthetic data

Property-Based Tests (optional, high value)
├── Position size never exceeds max allocation (any random input)
├── Liquidation price always between entry and zero (any leverage)
├── Drawdown calculation never exceeds 100%
└── Fee calculation always non-negative
```

**Target: 80% coverage on core modules** (sizing, fees, slippage, DSL translation, risk checks).

### Known-Result Fixtures

Maintain a small set of backtest runs with pre-computed expected results for regression testing. Any code change must reproduce these exact numbers (within floating-point tolerance).

---

## Phase 1: Foundation & Data Layer

**Goal:** Project skeleton, data ingestion with raw archival, NautilusTrader ParquetDataCatalog populated with 3 symbols across 2 years and 5 timeframes.

### Deliverables

1. **Project structure and dependencies**
   - `pyproject.toml` with `uv`: NautilusTrader, pandas-ta-classic, Streamlit, Plotly, DuckDB
   - Package structure: `vibe_quant/` with submodules (data, dsl, screening, validation, risk, dashboard)
   - Development tooling: pytest, ruff, mypy
   - Update devcontainer if needed

2. **Data ingestion CLI with raw archival**
   - Download 1-minute OHLCV from `data.binance.vision` for BTCUSDT, ETHUSDT, SOLUSDT (USDT-M perpetual)
   - Store raw CSV data in SQLite archive (`raw_klines` table) before any processing
   - Parse from archive → `BarDataWrangler` → NautilusTrader `Bar` objects
   - Write `CryptoPerpetual` instrument definitions to catalog
   - Write 1-minute bars to catalog
   - Aggregate and write 5m, 15m, 1h, 4h bars to catalog
   - Download and store funding rate history (`FundingRateUpdate`) in both archive and catalog
   - Data validation: gap detection, OHLC consistency, gap-fill via REST API

3. **Data update CLI**
   - Detect last timestamp in catalog per symbol
   - Fetch missing candles via Binance REST API
   - Archive raw data first, then append to catalog, rebuild aggregations

4. **Data rebuild CLI**
   - Rebuild entire ParquetDataCatalog from raw SQLite archive
   - Useful after NautilusTrader version updates or schema changes
   - `python -m vibe_quant.data rebuild --from-archive`

5. **SQLite state database**
   - Create all tables (strategies, sizing_configs, risk_configs, backtest_runs, backtest_results, trades, sweep_results, background_jobs)
   - WAL mode enabled by default on all connections
   - Basic CRUD operations via a `StateManager` class
   - Connection factory with WAL mode and busy timeout configured

6. **Verification**
   - Data quality checks: no gaps > 5 minutes, OHLC consistency (high >= max(open, close), low <= min(open, close), high >= low)
   - Row counts match expected (bars per day * days * symbols)
   - DuckDB can query the catalog Parquet files directly
   - Raw archive matches catalog data (spot-check validation)

### Acceptance Criteria

- `python -m vibe_quant.data ingest --symbols BTCUSDT,ETHUSDT,SOLUSDT --years 2` downloads, archives, and populates catalog
- `python -m vibe_quant.data status` shows symbol, date range, bar count, timeframes, archive stats
- `python -m vibe_quant.data rebuild --from-archive` rebuilds catalog from raw data
- NautilusTrader `BacktestEngine` can load data from catalog and run a trivial strategy
- Unit tests pass for data wrangling, aggregation, archival, and SQLite CRUD

---

## Phase 2: Strategy DSL & Screening Pipeline

**Goal:** Define strategies in YAML with multi-TF support and time filters, auto-translate to NautilusTrader, run parallel parameter sweeps with Pareto ranking.

### Deliverables

1. **DSL parser and validator**
   - YAML schema definition (JSON Schema for validation)
   - Parser that loads YAML into typed Python objects
   - Validation: indicator types exist, conditions reference valid indicators, sweep ranges are valid
   - Multi-timeframe validation: referenced timeframes are available in catalog
   - Time filter validation: valid session format, valid timezone
   - Error messages with line numbers for invalid DSL

2. **Indicator registry**
   - Pluggable indicator system: `@indicator_registry.register("RSI")`
   - Built-in indicators: RSI, MACD, EMA, SMA, BB, ATR, Stochastic, OBV, VWAP, CCI
   - Each indicator provides: NautilusTrader indicator class mapping, pandas-ta-classic function reference, parameter schema
   - Multi-timeframe indicator support: same indicator type on different bar types

3. **DSL-to-NautilusTrader compiler**
   - Takes parsed DSL → generates `Strategy` subclass (Python source code)
   - Maps DSL indicators to NautilusTrader built-in indicator classes
   - Generates `on_start()` with multi-TF subscriptions and indicator registration
   - Generates `on_bar()` with time filter evaluation, condition evaluation, and order submission
   - Generates strategy config class with all parameterizable fields

4. **Screening pipeline**
   - Compiles DSL to NautilusTrader Strategy
   - Builds parameter grid from DSL sweep section
   - Configures screening-mode venue (simplified fills, no latency)
   - Runs all parameter combos in parallel via multiprocessing
   - Computes metrics: Sharpe, Sortino, Max DD, Total Return, Profit Factor, Win Rate, Trade Count, Fees, Funding
   - Applies hard filters (min trades, max DD, min PF)
   - Computes Pareto front (3 objectives)
   - Stores all results in SQLite `sweep_results` table

5. **Example strategies**
   - RSI Mean Reversion with 1h/4h trend confirmation (multi-TF)
   - MACD Crossover with session filter (time-based)
   - Bollinger Band Squeeze (single TF, volatility breakout)

### Acceptance Criteria

- 3 example DSL YAMLs parse and validate correctly (including multi-TF and time filters)
- `python -m vibe_quant.screening run --strategy rsi_mean_reversion_mtf --symbols BTCUSDT-PERP` completes
- 100+ parameter combinations screen within reasonable time using multiprocessing
- Screening correctly models leverage and funding rates (unlike vectorized alternatives)
- Pareto front correctly identified in results
- Unit tests: DSL parser, multi-TF conditions, time filters, indicator mappings, Pareto ranking

---

## Phase 3: Validation Backtesting & Risk

**Goal:** Run full-fidelity NautilusTrader backtests with custom fills, latency simulation, position sizing, and risk management.

### Deliverables

1. **Validation venue configuration**
   - Custom `FillModel`: volume-based slippage with square-root market impact
   - Fee model: Binance maker 0.02% / taker 0.04%
   - `LatencyModelConfig` with presets (co-located, domestic, international, retail, custom)
   - Venue config: BINANCE, NETTING, MARGIN, starting balance, leverage
   - Funding rate integration: subscribe to `FundingRateUpdate`, accrue per 8h period

2. **Position sizing modules**
   - `FixedFractionalSizer(PositionSizer)`: risk_per_trade, max_position_pct
   - `KellySizer(PositionSizer)`: win_rate, avg_win, avg_loss, kelly_fraction
   - `ATRSizer(PositionSizer)`: risk_per_trade, atr_multiplier
   - All enforce max leverage and max position limits

3. **Risk management modules**
   - Strategy-level `RiskActor(Actor)`: monitors per-strategy equity, drawdown, daily loss
   - Portfolio-level `PortfolioRiskActor(Actor)`: monitors total equity, total exposure
   - Circuit breaker actions: cancel orders, set trading state to HALTED, log event

4. **Validation runner**
   - CLI: `python -m vibe_quant.validation run --run-id 42`
   - Loads strategy from SQLite, compiles to NautilusTrader Strategy
   - Configures BacktestEngine with venue (including latency model), data, strategy
   - Runs backtest, extracts results
   - Stores results in SQLite (backtest_results + trades tables)
   - Writes structured event log

5. **Screening-to-validation consistency check**
   - Compare screening results vs validation results for same parameters
   - Report: which candidates improved/degraded when moving to full-fidelity execution
   - Expected: validation results generally worse due to realistic fills, latency, and slippage
   - Flag candidates that degrade > 50% as "execution-sensitive"

### Acceptance Criteria

- DSL compiles to valid NautilusTrader Strategy that runs without errors
- Validation backtest produces realistic P&L including fees, slippage, funding, and latency effects
- Position sizing correctly limits risk per trade and total exposure
- Risk circuit breakers trigger at configured thresholds
- Latency model visibly affects results (compare same strategy with zero vs retail latency)
- Structured event log captures all signals, orders, fills, latency, and risk checks

---

## Phase 4: Overfitting Prevention Pipeline

**Goal:** Implement all three overfitting filters as a toggleable pipeline between screening and validation.

### Deliverables

1. **Deflated Sharpe Ratio**
   - Implementation of Bailey & Lopez de Prado DSR formula
   - Input: observed Sharpe, number of trials, return skewness, kurtosis
   - Output: p-value (probability Sharpe is not a fluke)
   - Filter: configurable confidence threshold (default 0.95)

2. **Walk-Forward Analysis**
   - Configurable window generator (train_days, test_days, step_days)
   - For each window: run NautilusTrader screening-mode backtest on train set, validate on test set
   - Compute WF efficiency: `mean(OOS_return) / mean(IS_return)`
   - Compute consistency: % of OOS windows profitable
   - Filter: min efficiency, min profitable windows
   - Default windows sized for 2-year data: 9m train / 3m test / 1m step → ~12 windows

3. **Purged K-Fold Cross-Validation**
   - K-fold time-series splitter with purge gap
   - Auto-detect purge size from strategy indicator lookback
   - For each fold: run NautilusTrader screening-mode backtest, compute OOS metrics
   - Filter: min mean OOS Sharpe, max Sharpe std deviation

4. **Pipeline orchestrator**
   - Toggleable filter chain (each enabled/disabled independently)
   - Reads sweep results from SQLite
   - Applies filters sequentially, tagging each result with pass/fail per filter
   - Updates `sweep_results` table with filter outcomes
   - Outputs filtered candidate list for validation

5. **CLI and reporting**
   - `python -m vibe_quant.overfitting run --run-id 42 --filters wfa,dsr,pkfold`
   - Summary report: N candidates in → N passed DSR → N passed WFA → N passed PKFOLD → N final

### Acceptance Criteria

- DSR correctly rejects high-Sharpe strategies when N_trials is large
- WFA correctly identifies strategies that degrade out-of-sample
- WFA produces ~12 windows from 2 years of data with default config
- Purged K-Fold correctly prevents data leakage
- Each filter can be independently toggled on/off
- Filter results stored in SQLite and visible in dashboard (Phase 5)
- Unit tests with synthetic data (known overfit vs robust strategies)

---

## Phase 5: Streamlit Dashboard

**Goal:** Full lifecycle UI for strategy management, backtest launching, and result analysis.

### Deliverables

1. **Strategy management tab**
   - List strategies from SQLite with status indicators
   - Create new strategy: auto-generated form from DSL parameter schema
   - Edit strategy: form mode + raw YAML editor toggle
   - Multi-TF indicator configuration UI
   - Time filter configuration UI
   - Strategy versioning (display history)

2. **Backtest launch tab**
   - Strategy selector (dropdown from SQLite)
   - Symbol selector (multi-select: BTC, ETH, SOL)
   - Timeframe selector (1m, 5m, 15m, 1h, 4h)
   - Parameter sweep range configuration (auto-generated sliders/inputs from DSL schema)
   - Overfitting filter toggles (checkboxes for DSR, WFA, PKFOLD)
   - Sizing module selector (dropdown of presets)
   - Risk module selector (dropdown of presets)
   - Latency preset selector (co-located / domestic / international / retail / custom)
   - [Run Screening] and [Run Validation] buttons
   - Active jobs list with status, progress, heartbeat indicator, and [Kill] button

3. **Results analysis tab**
   - Sweep results table: sortable by any metric, filterable
   - Pareto front scatter plot (Plotly): Sharpe vs MaxDD, color = PF, size = trades
   - Click-through to individual result detail:
     - Equity curve (Plotly line)
     - Drawdown chart (Plotly filled area, red)
     - Monthly returns heatmap (Plotly heatmap)
     - Trade distribution histogram
     - Trade log table (sortable, includes fees, funding, slippage per trade)
     - Key metrics panel (Return, Sharpe, Sortino, Calmar, MDD, WR, PF)
     - Cost breakdown (total fees, funding, slippage)
     - Overfitting filter results (pass/fail badges per filter)
   - Strategy comparison: select 2-3 results, side-by-side metrics

4. **Data management tab**
   - Catalog status: symbols, date ranges, bar counts per timeframe
   - Raw archive status: row counts, date ranges, last download timestamp
   - [Update Data] button triggers manual data fetch subprocess
   - [Rebuild Catalog] button rebuilds from raw archive
   - Data quality dashboard: gap detection, OHLC anomalies

5. **Settings tab**
   - Sizing module CRUD (create/edit presets)
   - Risk module CRUD (create/edit presets)
   - Latency model presets (view defaults, create custom)
   - Overfitting filter default configuration
   - Telegram bot token configuration
   - System info (NautilusTrader version, catalog size, SQLite stats)

### Acceptance Criteria

- Dashboard launches: `streamlit run vibe_quant/dashboard/app.py`
- Can create a strategy (including multi-TF, time filters), launch a screening sweep, view results -- full workflow in UI
- Background backtests don't freeze the UI
- Stale jobs detected and cleaned up on dashboard startup
- Pareto front visualization correctly highlights non-dominated solutions
- Individual backtest detail shows all charts and metrics including cost breakdown
- Latency preset selection works and affects validation results
- Settings persist across dashboard restarts (SQLite-backed)

---

## Phase 6: Paper Trading & Alerts

**Goal:** Connect to live Binance data for paper trading, with Telegram alerts for errors and circuit breakers.

### Deliverables

1. **NautilusTrader TradingNode setup**
   - Configuration for Binance testnet (WebSocket data, simulated execution)
   - Strategy deployment from DSL (same compilation as backtest)
   - Attached position sizing and risk modules

2. **State persistence**
   - Save/restore positions, orders, balance to SQLite on shutdown/crash
   - Periodic checkpoint every 60 seconds
   - Recovery: load last checkpoint, reconcile with exchange state

3. **Telegram bot**
   - Bot creation and token management
   - Alert types: error, circuit breaker, trade executed (configurable)
   - Rate limiting (max 1 alert per type per minute)
   - Daily summary message (P&L, open positions, metrics)

4. **Error handling**
   - Halt-and-alert flow for unexpected errors
   - NautilusTrader reconnection for transient errors
   - Strategy state machine: ACTIVE → HALTED → manual RESUME

5. **Paper trading dashboard tab**
   - Live P&L display
   - Open positions table
   - Recent trades list
   - Strategy status (ACTIVE/HALTED)
   - Manual controls: HALT, RESUME, CLOSE ALL

6. **Manual promotion workflow**
   - Dashboard UI: select validated strategy → [Start Paper Trading] button
   - Paper trading runs as long-running subprocess (tracked in background_jobs)
   - Results accumulate in SQLite for review
   - Compare paper results vs backtest expectations

### Acceptance Criteria

- Paper trading connects to Binance testnet and receives live data
- Strategy executes simulated trades based on real-time signals
- Telegram alerts fire on errors and circuit breakers
- State persists through process restart
- Dashboard shows live paper trading status

---

## Phase 7: Ethereal DEX Integration

**Goal:** Build custom NautilusTrader adapter for Ethereal exchange.

### Deliverables

1. **EtherealDataClient (LiveDataClient)**
   - Connect to Ethereal WebSocket (`wss://ws.etherealtest.net`)
   - Subscribe to: BookDepth, MarketPrice, FundingRate
   - Normalize Ethereal data to NautilusTrader types
   - Handle Socket.IO protocol

2. **EtherealExecutionClient (LiveExecutionClient)**
   - EIP-712 signature generation for order submission
   - REST API integration: place order, cancel order, get fills
   - Map Ethereal order types to NautilusTrader order types
   - Handle non-custodial model (wallet key management)

3. **Ethereal instrument definitions**
   - `CryptoPerpetual` configs for BTCUSD, ETHUSD, SOLUSD
   - Max leverage: BTC/ETH 20x, SOL 10x
   - Fee structure: maker 0%, taker 0.03%
   - Funding interval: 1 hour (vs Binance 8 hours)
   - Settlement: USDe (yield-bearing)

4. **Data ingestion for Ethereal**
   - Historical data from `archive.ethereal.trade`
   - Funding rate history
   - Store in raw archive, then write to ParquetDataCatalog

5. **Venue configuration for backtesting**
   - Ethereal venue config with correct fee/funding/leverage parameters
   - Latency presets adjusted for Ethereal (blockchain settlement adds latency)
   - Testnet integration for paper trading

### Acceptance Criteria

- Ethereal adapter connects to testnet and receives market data
- Orders can be placed/cancelled via EIP-712 signed requests
- Backtesting works with Ethereal venue configuration
- Funding rates apply hourly (not 8-hourly)
- Fee structure correctly applies 0%/0.03% maker/taker

---

## Phase 8: Automated Strategy Discovery (Genetic Optimization)

**Goal:** Genetic/evolutionary algorithm that automatically combines indicators and discovers profitable strategies.

### Deliverables

1. **Strategy genome representation**
   - Encode a strategy as a "chromosome": list of (indicator, parameter, condition) genes
   - Each gene represents one component: e.g., `(RSI, period=14, crosses_below, 30)`
   - Chromosome = entry conditions + exit conditions + stop/take profit config + time filters
   - Variable-length chromosomes (1-5 indicators per strategy)

2. **Genetic operators**
   - **Crossover**: Combine genes from two parent strategies
   - **Mutation**: Randomly change one gene (swap indicator, adjust parameter, change threshold)
   - **Selection**: Tournament selection based on fitness (Pareto ranking)
   - **Elitism**: Top N strategies survive unchanged to next generation

3. **Fitness function**
   - NautilusTrader screening-mode backtest of each candidate (parallel via multiprocessing)
   - Multi-objective fitness: Sharpe, MaxDD, ProfitFactor
   - Penalty for complexity (more indicators = slight fitness penalty, Occam's razor)
   - Overfitting filter integration: candidates must pass enabled filters

4. **Discovery pipeline**
   - Initialize population (N random valid strategies from grammar rules)
   - For each generation:
     - Evaluate fitness via NautilusTrader screening mode (parallel)
     - Apply overfitting filters
     - Select parents (tournament)
     - Generate offspring (crossover + mutation)
     - Replace weakest with offspring
   - Terminate after M generations or convergence
   - Output: top K strategies as DSL YAML for validation

5. **Dashboard integration**
   - Discovery tab in Streamlit
   - Configure: population size, generations, indicator pool, symbols, timeframe
   - Live generation-by-generation progress visualization
   - Fitness evolution chart (best/mean/worst per generation)
   - Final results: top discovered strategies with full metrics

6. **Guard rails**
   - Minimum 50 trades per candidate (prevents overfitting to few trades)
   - Complexity penalty prevents bloated strategies
   - DSR applied to entire discovery run (accounts for total candidates tested)
   - Walk-Forward required for final candidates before promotion

### Acceptance Criteria

- Genetic algorithm discovers strategies that pass overfitting filters
- Discovered strategies outperform random baselines on out-of-sample data
- Population evolves toward better fitness over generations
- Dashboard shows real-time evolution progress
- Top discovered strategies can be exported as standard DSL YAML for manual review

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| NautilusTrader breaking changes | High | Medium | Pin major.minor version, test before updates, maintain adapter layer |
| DSL expressiveness insufficient for advanced strategies | Medium | Medium | Hybrid approach: DSL + escape hatch for custom Python (Strategy subclass override) |
| Genetic discovery produces overfit strategies | High | High | Multi-layer overfitting prevention, DSR correction for total trials |
| Binance API rate limits during data ingestion | Low | Medium | Use data.binance.vision for bulk, REST API only for recent gaps |
| Binance Vision data gaps | Medium | Medium | Gap detection, REST API gap-fill, data quality flags in dashboard |
| Streamlit performance with large result sets | Medium | Medium | Pagination, lazy loading, DuckDB for aggregation queries |
| ParquetDataCatalog format changes | Medium | Medium | Raw data archive enables full catalog rebuild from original data |
| Slippage model inaccuracy at high leverage | High | Medium | Conservative defaults, validate against paper trading results |
| SQLite concurrent access | Low | Medium | WAL mode, busy timeout, connection factory pattern |
| Screening parameter sweep memory usage | Medium | Medium | Multiprocessing (separate process memory), chunked sweeps, memory estimation |
| pandas-ta-classic maintenance | Medium | Low | Prefer NautilusTrader built-in indicators (Rust); pandas-ta-classic only for exotic indicators |
| LGPL-3.0 compliance | Low | Low | No modifications to NautilusTrader source, all custom code in separate modules |

---

## Appendix: Interview Decision Log

All architecture decisions were made through a structured interview process and subsequently refined through technical review. Key decisions and rationale:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Core engine | NautilusTrader (single engine) | Execution realism critical at 20x leverage; Rust performance; backtest-live parity; correctly models funding/liquidation even during screening |
| 2 | Parameter screening | NautilusTrader screening mode + multiprocessing | Removed VectorBT: open-source lacks leverage/funding/liquidation support. Single engine ensures screening results are directionally consistent with validation |
| 3 | Data storage | NautilusTrader ParquetDataCatalog + raw archive | Native NT integration; raw data archived in SQLite for reproducibility and rebuild |
| 4 | State storage | SQLite (WAL mode) | Zero infrastructure; WAL enables concurrent dashboard + backtest access |
| 5 | Dashboard | Streamlit (full lifecycle) | Launch backtests, configure params, view results -- all in UI |
| 6 | Ethereal timing | Phase 7 (after Binance) | De-risk by proving core engine first; Ethereal only has 3 products |
| 7 | Overfitting rigor | Full pipeline, toggleable | Walk-Forward + DSR + Purged K-Fold; each independently controllable |
| 8 | Strategy system | Flexible, composable, multi-TF | Support multi-timeframe conditions, time filters, and position management |
| 9 | Symbol count | Start with 3 | BTC, ETH, SOL; easy to expand |
| 10 | Data history | 2 years | Sufficient for WFA (~12 windows); manageable data volume |
| 11 | Data pipeline | Manual CLI script with raw archival | No always-on infrastructure at MVP; raw archive enables reproducibility |
| 12 | Multi-strategy | Single at a time initially | Avoid portfolio orchestration complexity |
| 13 | Risk management | Strategy + portfolio level | Defense in depth; dual circuit breakers |
| 14 | Promotion | Manual gates | Human reviews every stage: backtest → paper → live |
| 15 | Async backtests | Background subprocess + PID tracking + heartbeat | No extra infra (no Redis/Celery); cleanup on dashboard restart |
| 16 | Parameter config UI | Auto-generated from schema | Strategy DSL defines param types/ranges; UI auto-builds forms |
| 17 | Timeframes | Pre-computed multi-TF | 1m, 5m, 15m, 1h, 4h in catalog; strategies subscribe as needed |
| 18 | Observability | Structured JSON event log | Queryable with DuckDB; strategy context included |
| 19 | DSL expressiveness | Indicator + Condition + Action + Multi-TF + Time Filters | Multi-TF confirmation and session filters essential for crypto perp futures |
| 20 | Position sizing | Separate pluggable modules | Same strategy, different sizing; decoupled |
| 21 | Screening ranking | Multi-metric Pareto front (3 objectives) | Tractable frontier size; shows meaningful trade-offs |
| 22 | Order types | Market entries + limit SL/TP | Taker fills on entry, maker on SL/TP; common pattern |
| 23 | Version management | Pin major.minor | Accept patches for bugfixes; test before upgrades |
| 24 | Deployment | Local dev machine | Everything in devcontainer; no cloud at MVP |
| 25 | Error handling | Halt + alert | Cancel orders, Telegram notification, wait for manual fix |
| 26 | Alerts | Telegram bot | Mobile push, common in crypto; easy setup |
| 27 | Testing | pytest unit + integration | 80% coverage on core; known-result fixtures |
| 28 | End goal | Automated strategy discovery | Genetic/evolutionary optimization; discover indicator combos |
| 29 | Technical indicators | pandas-ta-classic + NautilusTrader built-in | pandas-ta-classic is community-maintained fork; NT Rust indicators preferred at runtime |
| 30 | Latency simulation | NautilusTrader LatencyModelConfig | Native support with presets: co-located (1ms), domestic (20ms), international (100ms), retail (200ms) |
| 31 | License | MIT project + LGPL-3.0 dependency | LGPL permits unmodified library usage; no NautilusTrader source modifications |
| 32 | Strategy grammar | Phase later | Manual DSL first; add auto-generation in Phase 8 |
