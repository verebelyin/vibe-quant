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
│  Custom FillModel | LatencyModel (ticks) | Fees + Funding | Liquidation │
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

1. **Single-engine, two-tier fidelity**: NautilusTrader for both screening (simplified execution, parallel parameter sweeps) and validation (realistic fills, own-1m-detail-bar order release, full cost modeling; latency presets apply to tick data only). One engine means leverage and funding rates are always modeled; liquidation is an opt-in per-run overlay for screening, validation and discovery (OFF by default, §7).
2. **Single source of truth**: Strategy logic defined once in a declarative DSL, auto-translated to NautilusTrader Strategy subclasses. DSL supports multi-timeframe conditions, time-based filters, and position management.
3. **Separation of concerns**: Strategy signals are decoupled from position sizing, risk management, and execution. Each is a pluggable module.
4. **Incremental complexity**: Start with manual strategies and parameter sweeps. Build toward automated strategy discovery with genetic optimization.
5. **Defense in depth**: Multi-layer overfitting prevention with toggleable filters. Dual-level risk circuit breakers (strategy + portfolio).
6. **Data reproducibility**: Raw downloaded historical data is archived in SQLite so backtests can be replicated from the original source data at any time.

### Portfolio strategies: funding carry (backtest only)

A separate, self-contained strategy family (`vibe_quant/portfolio/`) — not a DSL strategy and not part of screening, validation or discovery. One pooled `OmsType.NETTING` account trades a weekly cross-sectional funding-carry book, generic in N symbols; the shipped run uses `BTCUSDT`/`ETHUSDT`/`SOLUSDT`/`BNBUSDT`. Backtest only.

```bash
python -m vibe_quant.portfolio run --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --start 2022-06-01 --end 2026-09-28 [--engine nt|reference]
python -m vibe_quant.portfolio evaluate --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --start 2022-06-01 --end 2026-09-28 [--draws 200] [--seed 0]
```

- **Decision:** every Monday 00:05:00 UTC (`D`); `L = D - 60 s`.
- **Signal:** the sum of settled funding rates over the trailing 7 calendar days `(L - 7d, L]`. A symbol whose expected settlement (on its real interval grid — Binance Vision `funding_interval_hours`) in that window has no archived row within ±60 s is excluded that week (`funding_hole`) — no forward-fill. Rates are summed raw; Binance already scales each settlement by `8/N`, so a 4h symbol's 42 half-size rates and an 8h symbol's 21 rates give comparable 7-day totals. A missing NON-8h row (e.g. a 4h symbol's 04:00) is detected too since 2026-10-17 (the archive's `raw_funding_intervals` table holds each row's Vision interval); before that it was silently free.
- **Ranking / weights:** total order by the key `(signal, symbol)`; `k = max(1, n // legs_divisor)` legs per side; long the `k` lowest signals, short the `k` highest; each leg `±gross/(2k)` (equal-dollar). `n < 2` → the book is flat (held legs closed).
- **Sizing:** the floor and the band compare are exact in `Decimal` — `target_steps = sign(w) * floor(Decimal(abs(w) * E_D) / (Decimal(price) * step))`, never float division (the `abs(w) * E_D` product itself is formed in float). The band never skips an open, a close or a flip; a same-side resize is skipped when `Decimal(abs(target - current)) < Decimal(str(band)) * abs(target)` (default band `0.10`). Binance min-quantity and MIN_NOTIONAL are enforced in code and can skip a non-reduce order — an open, a same-side increase, or a flip's open leg (a flip's close, and every other reduce-only close, is always sent). A book below the minimum tradable size fails closed.
- **Fills:** an order queued at `D` is released and filled on that symbol's OWN next 1m close (per-instrument outbox, strict `<`; a release more than 60 min stale, or an order still queued at the run end, fails the run). Orders are **never** submitted from the decision timer callback — a fill from the callback lands while the clock is at `D`, at the last datum's price, i.e. a look-ahead fill (MEASURED). Open positions are not closed at the run end (no fee; marked at the final price).
- **Funding:** signed quantity held strictly before each settlement × the MARK 1m open at the settlement minute × rate, via a windowed, seeded accrual (each window is seeded with the exact held quantity, so there is no float residue across windows). A missing MARK bar fails the run closed. A funding row missing while a position is held charges the flagged fallback rate and is counted.
- **Metrics:** non-overlapping weekly returns Sharpe ×√52 (headline) with the Lo standard error (`sqrt((1 + SR²/2) / years)`, `years = weeks/52`); daily ×√365 informational; decomposition of price P&L vs funding vs fees per leg, per side and per year (per-year price P&L is mark-to-market at the Jan-1 boundary prices); turnover; rounding drift; funding dispersion vs fees. A Sharpe is `None` — never a huge number — when `n < 3` or the stdev is within `1e-12 * max(1, |mean|)`. Liquidation is **not** modelled (`summary["liquidation"] == "not_modelled"`).
- **Outputs:** `data/portfolio/<engine>-<start>-<end>-<sha8 of the spec>/` (`summary.json` plus daily/weekly/fills/funding/decisions parquet ledgers; gitignored), and every run appends its `spec_hash` to the `data/portfolio/trials.jsonl` DSR trial ledger. The same rule modules (signal, sizing, ledger, ticks) drive both the NT engine and a no-NT reference simulator; `scripts/agents/exactness_carry.py` proves the two agree **exactly** (`parity.compare_runs` over every dataclass field) and pins the headline numbers.
- **Out of scope:** paper/live portfolio mode, `TradingGuard` portfolio caps, live funding signal, new symbols / data download, UI, DB schema, liquidation overlay, vol-scaled weights, liquidity filter.

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
| Backtesting engine | NautilusTrader (single engine) | Rust performance, execution realism critical for leveraged perps (run leverage 1-125x, default 10x), backtest-live parity, models leverage/funding always and liquidation as an opt-in per-run overlay (off by default) |
| Parameter screening | NautilusTrader screening mode + multiprocessing | Simplified fills for speed, but still models margin/funding, and liquidation as an opt-in overlay (off by default), unlike vectorized alternatives which lack these features |
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

-- Binance Vision funding_interval_hours; append-only, refreshed by `python -m vibe_quant.data funding-intervals`
CREATE TABLE raw_funding_intervals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    funding_time INTEGER NOT NULL,     -- Unix timestamp ms
    interval_hours INTEGER NOT NULL,   -- settlement interval in effect at funding_time
    source TEXT NOT NULL,
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, funding_time)
);

CREATE INDEX idx_raw_klines_symbol_time ON raw_klines(symbol, interval, open_time);
CREATE INDEX idx_raw_funding_symbol_time ON raw_funding_rates(symbol, funding_time);
```

### Data Ingestion Pipeline

**Supported symbols (8):** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `SUIUSDT` (Vision data from 2023-05), `ZECUSDT` (from 2020-02), `NEARUSDT` (from 2020-10) and `HYPEUSDT` (from 2025-05); HYPE settles funding every 4h, the others every 8h. A new coin's first ingest goes through the CLI or `POST /api/data/ingest`; the UI ingest form lists archive symbols only.

**Initial bulk load (example: 3 symbols, 2 years, 1-minute):**

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

**Declared params only**: an indicator accepts only the params its `indicator_registry` spec's `param_schema` declares (in the frontend's `IndicatorConfig` field names, e.g. STOCH `period`/`d_period` for the registry's `period_k`/`period_d`). An undeclared builtin param is rejected with `Indicator '<TYPE>' does not use param '<key>'. Valid params: [...]`; an explicit `null` is treated as absent. It is checked when a strategy is saved — **indicator-only**, so an unrelated condition or risk error does not block a save, and a config with no indicators is a no-op — and again at launch, so a launched run carrying an unused param fails with that message instead of silently ignoring it.

**Plugin extension**: Custom indicators can be added by dropping a `.py` file in `vibe_quant/dsl/plugins/`. Each plugin registers an `IndicatorSpec` declaring its `compute_fn`, parameter metadata, and GA enrollment fields. The compiler, schema validator, GA pool, and frontend catalog all read from the live `indicator_registry` — no per-indicator edits required. See `vibe_quant/dsl/plugins/README.md` for the full API.

**Context indicators (`needs_context`)**: a spec with `needs_context=True` reads non-OHLCV data instead of only the bar buffer. The compiler then also buffers each bar's close time (`ts_init` rounded to the timeframe boundary) and hands `compute_fn` a DataFrame with `attrs["bar_close_ns"]` and `attrs["symbol"]`; strategies without such an indicator compile to byte-identical source. Data is looked up as-of the bar close through `vibe_quant/dsl/aux_data.py`, which screening/validation configure in-process (archive path, no StrategyConfig fields). Built-ins:
- `FUNDING` (`period`, default 1): last settled perpetual funding rate in bps (mean of the last `period` settlements if > 1).
- `FUNDING_Z` (`period`, default 30): z-score of the rate over the last `period` 8h settlements, computed on the full settlement series (not the bar buffer).
- `BTC_TREND` (`period`, default 200) / `OWN_TREND`: BTCUSDT (resp. the strategy's own symbol) daily `close / EMA(close, span=period) - 1`. `BTC_ROC` (`period`, default 20): BTCUSDT daily `close[i] / close[i-period] - 1`. Computed once on the full daily history from the catalog and looked up as-of the strategy bar close (a bar sees only daily bars already closed: a 4h bar closing 20:00 on day D sees day D-1); NaN when the as-of daily bar is > 2 days old. Preflight requires the daily series to cover `[start - 3d, end - 2d]`, else `AuxDataUnavailableError`.

Archived settlement times are snapped to the 8h boundary (ms jitter) before the as-of lookup: the 08:00 bar sees the 08:00 settlement, a bar closing 07:59:59.999 sees 00:00. Coverage is enforced by a run-level preflight (screening and validation, with the run window) that raises `AuxDataUnavailableError` (a `DataUnavailableError`, never masked as a -inf result) when the archive is missing, has no funding for the symbol, starts more than 3 days after the window start (warmup allowance) or ends before `end - 16h`. Inside the run, per-bar lookups return NaN (never raise, since NT swallows `on_bar` errors) before the first settlement and when the last settlement is more than two funding periods old (archive holes are not forward-filled). Context indicators never forward-fill: a NaN from the lookup (including FUNDING/FUNDING_Z across an archive hole) is stored as the current value, so conditions on it are False instead of using the last valid value. Context indicators are excluded from the default GA pool, and paper/live trading refuses strategies using them (no live feed).

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
- **Funding avoidance**: Skip entry signals within N minutes of the symbol's own funding settlement times: the archived per-symbol interval decides, causally (most Binance USDT-M perps settle every 8h at 00:00, 08:00, 16:00 UTC; HYPEUSDT settles every 4h; Ethereal: every hour). A settlement archived later never changes an earlier decision: across an archive hole wider than the next row's own interval the previous row's interval applies, and before the first archived row (or with no archived interval at all) the 8h grid applies with a WARNING.

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

The screening pipeline uses NautilusTrader in a **simplified execution mode** for fast parameter sweeps. Unlike vectorized alternatives, this approach correctly models leverage, margin and funding rates even during screening -- critical for leveraged perpetual futures (run leverage 1-125x, default 10x) -- with liquidation available as an opt-in per-run overlay (OFF by default; see §7).

**Screening mode simplifications** (vs full validation):
- Basic `FillModel` with simple probabilistic slippage (no custom volume-based model)
- No `LatencyModel` (zero latency; latency presets apply only to tick data)
- Standard fee model (not custom)
- Bar-level data only (no order book)

**Fill timing:** each compiled strategy queues its orders in a per-instrument outbox and releases them on that instrument's own next price update. Screening synthesizes two 1m "fill ticks" per strategy bar (`vibe_quant/data/fill_ticks.py`, `command_release="trade_tick"`), so an order fills at the close of the first 1m bar after the signal — the same release validation uses, not the next strategy-timeframe bar. Two caveats: drawdown is still marked on strategy-timeframe bars (so screening max DD can differ from 1m-detail/validation DD), and with gaps in the 1m data, release waits for the next own tick.

**What screening mode still models** (the key advantage):
- Leverage and margin mechanics
- Funding rate payments (8-hourly for Binance)
- Liquidation risk at the configured leverage -- only when the opt-in overlay is on (off by default)
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
    starting_balances=[f"{parameters.initial_balance} USDT"],  # parameters.initial_balance, default 1000
    default_leverage=parameters.leverage,  # run setting: whole number in [1, 125], default 10; no per-symbol overrides
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

Top candidates from screening are validated through NautilusTrader's full-fidelity execution simulation with realistic fills, own-1m-detail-bar order release, and cost modeling (latency presets apply to tick data only).

**Venue configuration:**

```python
venue_config = BacktestVenueConfig(
    name="BINANCE",
    oms_type="NETTING",
    account_type="MARGIN",
    base_currency=None,
    starting_balances=[f"{parameters.initial_balance} USDT"],  # parameters.initial_balance, default 1000
    default_leverage=parameters.leverage,  # run setting: whole number in [1, 125], default 10; no per-symbol overrides
    fill_model=custom_fill_model,          # Custom volume-based slippage
    fee_model=fee_model,                   # VIP0 maker 0.02% / taker 0.05% (vibe_quant/fees.py); run-level fee tier and BNB discount (x0.9)
    latency_model=latency_model,           # Network latency simulation (tick data only; bars release via the per-instrument outbox)
)
```

### Latency Model Configuration

NautilusTrader's `LatencyModelConfig` simulates realistic network delays for order operations. Configurable per venue with presets (presets apply only to tick data — bar backtests release orders through the per-instrument outbox described below instead):

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

**Order release (2026-10-10):** bar backtests do not use `LatencyModel` — it releases queued orders venue-wide, so in a multi-symbol run a later symbol filled at the signal bar close and inflated the result. Each compiled strategy instead queues orders in a per-instrument outbox and releases them on its own next price update: validation on the instrument's next 1m detail bar (`command_release="bar"`), screening on the instrument's own fill ticks (§6). With gaps in the 1m data, release waits for the next own tick.

### Execution Simulation

**Order types used:**
- **Entry**: Market orders (taker fee, filled on the instrument's next own fill tick -- validation's next 1m detail bar -- + slippage)
- **Stop Loss**: Stop-market orders (triggered at mark price, filled with slippage)
- **Take Profit**: Limit orders (maker fee, filled at limit price with queue position probability)

**Fill model (custom):**
Custom fill model loaded via `ImportableFillModelConfig` (specifying `fill_model_path` and `config_path` to a subclass of `FillModel`):
- Market orders: Fill at bar open price + volume-based slippage
- Slippage formula: `spread/2 + 0.1 * volatility * sqrt(order_size / avg_volume)`
- Stop orders: Fill at stop price + 1-tick slippage (pessimistic)
- Limit orders: Fill at limit price if price touches (probabilistic fill for queue position)
- OCO handling: When both SL and TP could trigger on same bar, SL triggers first (pessimistic)
  - **Implementation deviation (2026-10-03, vibe-quant-e70tl.12):** NautilusTrader picks the intrabar high/low order once per bar inside its compiled matching engine, so per-order SL-first is not expressible. Both tiers use NT `bar_adaptive_high_low_ordering=True` (extreme nearer the open first) — symmetric for longs/shorts. Validation uses 1m detail bars, where both-levels-in-one-bar is rare.

**Funding rate simulation:**
- Subscribe to `FundingRateUpdate` data (stored in catalog)
- Binance: applied at each settlement of the symbol's own interval (8h at 00:00, 08:00, 16:00 UTC for most symbols; the archive's `funding_interval_hours` decides, e.g. HYPEUSDT every 4h)
- Positive rate: long positions pay, short positions receive
- Negative rate: short positions pay, long positions receive
- Accrual: `position_notional * funding_rate` per period
- **Screening/discovery with an execution bar longer than 8h** funds a position closed by a boundary STOP_MARKET/LIMIT exit up to its refined 1m bar (the same refinement the liquidation overlay uses); shorter bars use the NT fill time (vibe-quant-6osqd).
- **Holes are detected on each symbol's real settlement interval** (Binance Vision `funding_interval_hours`, archive table `raw_funding_intervals`; refresh with `python -m vibe_quant.data funding-intervals`). A missing settlement is charged the fallback 0.01% per 8h scaled to its interval/8 (user decision 2026-10-17) and counted in `funding_fallback_settlements`. Rows without a Vision interval (current month) inherit the previous row's. Between two archived rows the expected interval is `H = max(h_prev, h_next)`; settlements on that grid missing from the archive are holes.

**Liquidation simulation (opt-in, OFF by default):**
- A per-run overlay available to screening, validation and discovery: mode `off` (default), `isolated` or `cross`; leverage a whole number in `[1, 125]` (default 10). With the mode on, every metric is recomputed from the truncated ledgers (trades, return, win rate, PF, return moments, Sharpe/Sortino and the daily series, MTM max DD), not NT's raw path.
- The venue leverage is `parameters.leverage` (default 10) in **every tier and mode**: screening and discovery share one path (`screening/nt_runner.py`) and validation uses `validation/runner.py`, and both set the venue `default_leverage` to `parameters.leverage` whether the overlay is on or off. The liquidation distance itself uses the same leverage. NT's own margin checks and rejections therefore run at that venue leverage in every mode, so a mode-on run at `L != 10` can differ from mode off only through the liquidation itself (with the overlay off, leverage practically never changes results).
- The scan walks Binance **MARK-price 1m klines** over the half-open run window `[start, end)`: a position is liquidated when the mark crosses its liquidation price. Only MARK bars opening inside the window are seen (plus one carry bar before `start`), so a position still open at the run end is never liquidated by a later price, and a fill, funding or other cash flow at `ts >= end` is refused.
- Liquidation prices, isolated/cross margin math and the maintenance-margin rate use a versioned per-symbol **snapshot of the Binance USDT-M maintenance-margin tier tables** (`vibe_quant/liquidation/binance_um_brackets.json`, version `2026-10-12`, from the Binance futures `brackets` endpoint). Historical bracket changes are not modelled; a notional at or beyond the last bracket is an error.
- The **clearance fee is a cost**: `total_fees` includes it and `fee_drag` counts it. `liquidation_forfeited` (lost margin) overlaps `total_fees` and must never be added to it.
- **Integrity errors fail the run (fail closed):** missing or incomplete MARK 1m coverage, an unpriceable open survivor, or a fill/funding/cash flow at `ts >= end` ends the run `failed` with `LiquidationIntegrityError: ...` (never a `-999`/`-inf` row). This holds for screening, validation, discovery (every stage) and the overfitting runners.

### How to turn liquidation on

OFF by default: a run without `liquidation_mode` is off and, at the default leverage 10, stays byte-identical to before at execution bars <= 8h (import-isolated at execution bars <= 8h: a mode-off run at or below 8h never imports `vibe_quant.liquidation`, while on a longer bar -- today 1d -- it imports the exit-timing refinement for funding, vibe-quant-6osqd, so 1d mode-off funding follows the refined exit since 2026-10-17). At a mode-off leverage `L != 10` the screening/discovery venue runs at `L` (`screening/nt_runner.py`), so the numbers differ only where NT's margin checks bind (strategy 239 at 1x/125x measured identical to 10x; not guaranteed for every strategy). It needs MARK 1m klines in the archive (and, for the engine, in the catalog):

```bash
python -m vibe_quant.data download-mark --start 2024-01-01 --end 2026-03-17 --catalog
# --start is required; --end defaults to today; --symbols defaults to all supported symbols
# --catalog also rebuilds the MARK 1m bars in the ParquetDataCatalog from the archive
```

- **Screening** -- API/UI: set the top-level `liquidation_mode` (`off|isolated|cross`) on the launch request. The launch form has a **Liquidation** select (Off/Isolated/Cross) and reuses the existing **Leverage** input as the venue leverage; `Off` sends no `liquidation_mode` key, but `parameters.leverage` is always sent and the screening venue runs at it in every mode (default 10; a whole number in [1,125]). There is no screening CLI flag.
- **Validation** -- API/UI: the same top-level `liquidation_mode` field; the launch form has the same **Liquidation** select (Off/Isolated/Cross) reusing the existing **Leverage** input. Turning the mode on does not change validation's venue leverage (it is `parameters.leverage` in both modes).
- **Discovery** -- CLI: `--liquidation-mode isolated --leverage 20` (no penalty term, no gate; fitness reads the liquidation-adjusted metrics; the deprecated `--liquidation-leverage` spelling is an alias). API/UI: `liquidation_mode` + `leverage` (deprecated `liquidation_leverage` alias; a conflict is a 422); the launch form has the same **Liquidation** select plus an always-visible **Leverage (1-125)** input.
- **Discovery bankroll grid (optional)** -- search a grid of margin mode × leverage × cushion instead of one run-level setup. CLI: all three of `--bankroll-modes` (comma-separated; `isolated`/`cross` only — a grid never runs `off`), `--bankroll-leverages` (comma-separated whole numbers 1–125) and `--bankroll-max-margin-pcts` (comma-separated numbers in `(0, 1]`); API `bankroll_grid = {modes, leverages, max_margin_pcts}`; the discovery form has matching grid fields. Cells = product(modes in `(isolated, cross)` order, sorted leverages, sorted cushions), deduplicated, **at most 12**; every cell must be able to margin the run's exposure cap at its own leverage. A grid is **exclusive** with the run-level `--liquidation-mode` / `--leverage` / `--max-margin-pct` (a conflict is rejected before any run). Discovery runs the search once per cell (× seeds) and picks the best strategy and cell together; every cell's evaluations count as trials for the DSR gate (**N = Σ over cells × seeds**), so a larger grid makes passing harder and costs proportionally longer. Each champion records its cell (`bankroll = {liquidation_mode, leverage, max_margin_pct}`, plus `notes.bankroll_grid = {cells, n_cells}`, and no run-level `liquidation`/`leverage`/`sizing.max_margin_pct`), promote/replay inherit exactly that cell (a missing/invalid cell is a 422), and the full-range headline is measured at the champion's own cell. **No grid** = today's single-setup path, unchanged.
- **Promote / replay** inherit the parent discovery run's `notes.liquidation` into the child validation run.
- **Results:** a mode-on run's results show a liquidation chip and counts.
- **Paper trading:** the paper node reads the validated run's leverage and validates it through the shared rule, derives its margin type from the validated run, and runs a Binance `forceOrders` detector. It is **report only** — no orders, no halt (user decision 2026-10-15). Live trading is not supported.
  - **Leverage:** a whole number in [1,125]; a stored `0.5`/`2.5` or garbage stops the paper start with a clear error, and a config-file override is validated the same way.
  - **Margin type:** derived from the validated run (isolated → `ISOLATED`; cross/off → `CROSSED`; an explicit config value wins and shows a `config_diff` entry). A validated run with liquidation off adds a `config_diff` `liquidation` entry ('paper liquidation parity is unverified').
  - **On position open:** one RISK_CHECK `liquidation_estimate` (isolated: the estimated LP; cross: `null` with a reason).
  - **Startup:** if the venue is not connected 120 s after start, one RISK_CHECK plus a critical Telegram, and strategies are never started.
  - **Liquidation detector:** polls every 60 s per symbol (30 s request timeout, per-symbol backoff 60/120/240/480/600 s, a degraded warning after 3 failures, 7-day lookback cap) and writes one LIQUIDATION event plus a critical Telegram per exchange liquidation/ADL; delivery is at most once per process and at least once across restarts.

### Position sizing

Every entry is sized by the rule below (settings validated by `vibe_quant/sizing_settings.py`; sizing code in the generated strategy, `vibe_quant/dsl/templates.py`). It reads the **balance**: the account's wallet balance (NT `balance_total`), without unrealised PnL. (**Equity** = balance + unrealised PnL is what the margin limit and the paper guard read.)

```
size = min(balance * risk_per_trade / stop_distance, balance * max_position_pct / price)
```

Then: a size below the instrument's min quantity is raised to the min quantity when that still fits under the `max_position_pct` cap, and the entry is skipped when it does not (so at small balance a trade can risk more than `risk_per_trade`, never more than the cap); the size is rounded down to the size increment (skipped if that leaves nothing); an entry below the min notional is skipped.

Two **run-level** settings drive it:

- `max_position_pct` — "Max exposure (× balance)", default **0.5**, range `(0, 125]`. It is a **cap, not a target** (the risk rule can size smaller), and it is **per symbol**: N symbols can hold up to N × cap in total.
- `risk_per_trade` — "Risk per trade", default **0.02** (the UI shows percent: default 2, max 50), range `(0, 0.5]`.

A third run-level setting is a **limit on margin in use**, not a sizing rule:

- `max_margin_pct` — "Max margin in use (% of equity)", a fraction in `(0, 1]`, **absent = OFF**. It caps the sum of `notional / leverage` over every open position, every pending entry order and the candidate entry, against `max_margin_pct × equity`, where `equity` is wallet + unrealised PnL (NT's portfolio equity in a backtest). Landing **exactly** on the limit is allowed (the sum and the limit are compared exactly, so a non-terminating division like `1000/3 + 500/3` still fits `500`). An entry that would go over is **refused, never shrunk** (backtests log a warning and count the refusal; paper also alerts once per symbol, see § Paper); exits and stop-losses are never blocked.

Off (absent) the limit reads nothing, so a run without it is byte-identical to before. In **backtests** every item uses the run leverage (`parameters.leverage`). In **paper/live** each item uses the leverage the venue **confirmed** for its symbol (`account.leverage(instrument_id)`, from the venue's `symbolConfig`); an entry whose symbol has no confirmed leverage is **refused** (fail closed), and the requested leverage is not a substitute. Pending orders count **once**, as the union by `client_order_id` of NT's open and inflight orders (Nautilus reports `PENDING_CANCEL`/`PENDING_UPDATE` in both), including entries the guard already admitted that are not yet open. Paper always runs with a limit (default **0.5**); a backtest or discovery run without `max_margin_pct` has none.

**Margin-limit accounting:** validation records `notes.margin_limit = {max_margin_pct, refused_entries}` (a walk-forward run sums its windows' counts); screening applies the same rule but only logs the refused count (no per-row column); the results chip shows `· Max margin 50%` for every run type, plus `· N entries refused` for a validation run.

The launch check only requires `max_position_pct <= leverage` (the run's `parameters.leverage`, default 10). Binance notional brackets are **not** enforced: equity 10,000 × cap 125 = 1.25M BTC notional sits in the 800k–3M bracket (max 75x) while the run's leverage is 125 — a backtest fills it, Binance would reject.

- **Where it is set:** the Backtest Launch form and the Discovery form (two inputs; Launch is disabled while either is invalid); API `parameters.max_position_pct` / `parameters.risk_per_trade` (backtest launch) and top-level `max_position_pct` / `risk_per_trade` (discovery launch); discovery CLI `--max-position-pct` / `--risk-per-trade`. The margin limit has its own launch input, **Max margin in use (% of equity)** (entered as a percent, carried as the fraction; empty = off, a value outside (0, 100] is refused): API `parameters.max_margin_pct` (backtest launch) and top-level `max_margin_pct` (discovery launch); discovery CLI `--max-margin-pct`.
- **Stored only when non-default** — a default launch stores exactly what it did before. A discovery run records `notes.sizing`; promote/replay inherit it (a present-but-unreadable `notes.sizing` is a 409 and writes no rows). A direct CLI discovery run records `notes.sizing` only, while the API launch also stores the run parameters.
- **Not sweepable:** a sweep key naming a sizing field (including dotted spellings like `max.position.pct`) is rejected.
- **Validation consistency** compares only against a screening reference with the same sizing. The sweep-row **Validate** button forwards the parent run's leverage, `initial_balance`, `liquidation_mode`, `bnb_discount`, `fee_tier` and sizing, and refuses a parent with a DSL override.
- **Results:** the results page shows a sizing chip, e.g. `Max exposure 0.5× · Risk 2% (default)`; a multi-symbol run appends "per symbol, up to N × … total", and a run with a margin limit appends `· Max margin 50%`.
- **Comparability:** defaults are byte-identical to before (the exactness proofs are unchanged), so a non-default sizing run is a new run, not a comparability break; a run without `max_margin_pct` is likewise byte-identical (exactness_239 IDENTICAL), so the margin limit is opt-in and not a break either.

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
- Filter rule (one helper, `discovery/guardrails.py::q95_dsr_decision`, used by the discovery gate and the promote-time re-check in `overfitting/pipeline.py`): the statistic is the z-score `(SR_hat - SR_0) / sqrt(Var(SR_hat))` of the champion's **mark-to-market daily** returns; a missing series fails closed.
  - **q95 rule** (discovery default, `--dsr-rule q95`): pass when z >= the run's Monte-Carlo spread `quantile(M, 0.95) - mean(M)` (`notes.dsr_trials.neff_ev_q95_spread`).
  - **Legacy rule** (`--dsr-rule emax`, discovery runs without a recorded q95 spread, and screening-sweep candidates): pass when p = 1 - Φ(z) < 0.05, i.e. z > 1.6449 (the overfitting pipeline tests 1 - p >= `confidence_threshold` 0.95, which differs only at p exactly 0.05).

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
- Walk-forward efficiency (length-normalised, Pardo): `mean(OOS_return / OOS_days) / mean(IS_return / IS_days)`, days counted inclusive from each window's dates; 0.0 when the mean IS return per day is below 0.001 / mean IS days (no in-sample edge); capped at 5.0 -- should be > 0.5
- OOS Sharpe: Sharpe ratio computed only on out-of-sample periods
- Consistency: percentage of OOS windows that are profitable

**Filter threshold:** walk-forward efficiency > 0.5 AND more than 50% of OOS windows profitable (OOS return > 0) AND the sum of OOS window returns > 0

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
  risk_per_trade: 0.02            # 2% of balance per trade
  max_position_pct: 0.5           # cap: 0.5 x balance per symbol (run default)
  max_leverage: 10
```

The run-level sizing implemented today is the min(risk, cap) rule (defaults: 2% risk per trade, 0.5 × balance cap per symbol) — see [§ Position sizing](#position-sizing).

**Available methods:**

| Method | Formula | Use Case |
|--------|---------|----------|
| Fixed Fractional | `position = (balance * risk_pct) / (entry - stop_loss)` | Conservative, consistent risk |
| Kelly Criterion | `f* = (W - (1-W)/R) * kelly_fraction` | Optimal growth, uses half-Kelly (0.5) |
| ATR-Based | `position = (balance * risk_pct) / (ATR * multiplier)` | Volatility-adaptive sizing |

All methods enforce:
- Maximum position size as % of balance
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
4. Set node state to `HALTED`
5. Wait for manual intervention (operator reviews, fixes, resumes)

Transient errors (WebSocket disconnect, API timeout) are handled by NautilusTrader's built-in reconnection with exponential backoff. Strategy-level errors (invalid signal, sizing calculation failure) trigger the halt-and-alert flow.

### Risk Controls

- **Exposure refusal:** paper refuses to start when the effective `max_position_pct` (the validated run's stored value, or paper's sizing override; default 0.5) exceeds paper's effective leverage (the validated run's leverage, or paper's leverage override; default 10). `POST /api/paper/start` returns 422 before any run row is written, and the node refuses on every other start path. The check is per symbol (N symbols can hold N × cap); a start whose maximum possible margin (`n_symbols × cap / leverage`) exceeds the cushion **warns but proceeds**. **Margin limit at order time:** the paper guard enforces the run-level `max_margin_pct` (§ Position sizing) on every entry — margin in use (open positions + pending entries + the new entry) ≤ `max_margin_pct × equity`, at the venue-**confirmed** per-symbol leverage, a refused entry never shrunk and exits never blocked — and alerts once per symbol until an entry on that symbol passes again. The old notional `max_total_exposure_pct` is **superseded and not read** (bead vibe-quant-9r0m5).
- **Halt restore:** after a restart, a `MANUAL`, `ERROR`, `SIGNAL` or `KILL_SWITCH` halt comes back `HALTED` until the operator resumes; on the first halted guard tick the node closes owned positions and stops strategies exactly as the original halt did. Risk halts (drawdown, consecutive losses, daily loss) re-trip from the restored risk numbers. Old checkpoints without a halt reason start `RUNNING`; a failed restart keeps the halt and its message. A halt requested while the node is still starting up (SIGUSR1 during start-up) is recorded and applied through the trading guard as soon as it is built, before any strategy runs: the session starts HALTED/SIGNAL, flattens and alerts once (the alert arrives when the guard applies it, at most about 25 s plus database waits after the signal).
- **Kill switch fails closed:** an unreadable system kill switch reads as **engaged**. A missing `system_state` row is re-created **ENGAGED** with reason `kill-switch record was missing; re-created engaged` (`vibe_quant/db/schema.py`); a brand-new database still starts unlocked, and the first unlock after a re-create is refused (409) so the operator unlocks again deliberately. Reads go through a background reader (one read every second) so the trading loop never waits on the database; the node halts `ERROR` when the last clean read is **8 s** old — a wall-clock bound ≤ 15 s for **database** and local event-file stalls: event-log (JSONL) appends run on their own worker thread (bead vibe-quant-ogd37), a full queue (10 000 events) drops new events and counts them, and an unwritable log refuses to start. A critical alert that must know its line landed (an exchange liquidation) can wait up to 5 s for the disk before it is sent; every other alert is sent without waiting. **Resume** needs a clean, not-engaged read at most **2 s** old. `GET /api/system/status` is a **503** when the row cannot be read (never a default 200), and the header shows "Kill switch state unknown" in red while **keeping the Kill button**; a `POST /kill` always cascades to the running sessions and reports a 503 if its write did not land. Command handling and checkpoint writes run on worker threads, and shutdown always reaches `stop_async`.
- **Risk carry-over:** a new paper session inherits the last session's risk bookkeeping (high water mark, UTC-day baseline, losing streak) from the newest checkpoint of another trader on the same account, so a fresh trader id cannot reset the drawdown/daily-loss/streak limits; drawdown and streak halts therefore re-trip until "start fresh" or a raised limit. Account = venue + testnet/live + first 16 hex of `sha256(api_key)` (the key is never stored; a LIVE start with no saved risk state to carry is refused unless `confirm_fresh_live` is given; a key rotation by the same trader id carries its own risk bookkeeping and halt and needs no consent (see One session per account); testnet and live never mix). Of another session's halts only ERROR is inherited (an unrecognised reason counts as ERROR); the trader's own halt always wins, while a newer matching account checkpoint of another trader supplies the risk numbers; the trader's own checkpoint counts only if its account matches (or it predates the account key, or it is the same venue + env under another key: a key rotation carries the own risk bookkeeping and halt, never the liquidation watch — bead vibe-quant-qa42q), so the same trader id on testnet then live carries nothing (the other account's row is logged and skipped) and going back to testnet restores the newest testnet row, halt included, with or without `start_fresh`; the trader's NEWEST row in the same venue + env is the source, so rotating back A->B->A restores B's newer row (bead vibe-quant-ae6ah). Inheriting from another trader logs a WARNING and a drawdown/streak halt on inherited risk names the source. `start_fresh` (config / `--config` JSON key, default false) skips the cross-session risk/halt inheritance and never discards the trader's own checkpoint; the liquidation watch is still inherited from the newer account row. An unreadable chosen checkpoint halts the session ERROR (fail closed); unrelated malformed rows are skipped (bead vibe-quant-3fqw9). The node also checkpoints immediately after every guarded position close, so the just-booked losing streak survives a SIGKILL (bead vibe-quant-8vc7h). The liquidation watch (forceOrders cursors and dedupe keys) comes from the same source checkpoint merged with the trader's own row — the newer cursor wins for a symbol both name, a symbol only the own row names keeps its own cursor, and the gap boundary is the older of the two timestamps — so no venue liquidation during the gap between two sessions on one account is skipped; `start_fresh` skips only the risk/halt side of that lookup — the watch is still inherited and merged with the own row (beads vibe-quant-1zh0m, vibe-quant-1ghod). Symbols with a restored cursor that the session does not trade are polled too, after the traded ones and at the same cadence, with the same per-symbol backoff and degraded/recovered alerts, so a liquidation of a leftover position on such a symbol is reported during the session and its cursor advances in every checkpoint (beads vibe-quant-0m3tf, vibe-quant-1cvw6); every due symbol is fetched concurrently each cycle (bead vibe-quant-qe6ye); each watched symbol costs forceOrders weight 20 per 60 s cycle and the node logs a WARNING when the watch exceeds 240/min (10% of Binance's 2400/min IP cap, which every session and NT share per IP), so before live keep watched symbols × sessions per IP ≤ 12 (bead vibe-quant-altqb); the dedupe keys stay bounded by the 7-day lookback; a LIVE start with no saved risk state to carry (no checkpoint on the account and no earlier live session of that trader id on another key in the same env: a first session, or a new trader id on a new key) is refused unless `confirm_fresh_live` / `--confirm-fresh-live` is given; a refused CLI start with `--run-id` (e.g. `/restore` of a trader with no risk state to carry) records the refusal text as the run's and job's terminal failure, and the API launch cannot overwrite it: `start_job` reopens the run and job (a fresh attempt) before the spawn, and every later launch write is conditional, so a terminal status never goes back to `running` and its error is never cleared (bead vibe-quant-6ub67).
- **One session per account:** paper holds an OS advisory `flock` on one file per account id (in `<OS user's home from the passwd database, not $HOME>/.vibe-quant/paper_locks`, per machine: a main checkout, a worktree and a ui-check stack contend on the same file) for the whole process life, so a second session on the same account is refused instead of running with its own separate risk guards: `POST /api/paper/start` and `/restore` return 409 (`Paper session already holds account <id> ...`) before any run row is written, and `python -m vibe_quant.paper.cli start` exits 1 with `AccountBusyError`. The account identity is the risk-carry-over one (venue + testnet/live + key hash), so testnet and live never block each other. The kernel drops the lock when the holding process dies (SIGKILL included), so there is no stale lock to clear; an unreadable holder body (I/O error) still refuses, naming the holder `unknown`; the API probe is advisory and the CLI lock is the gate (bead vibe-quant-rw1g8). A second API key on the same Binance account is a different account id (no lock contention, separate risk state), so the operator rule is one key per live account (bead vibe-quant-6ub67): (1) check, while the backend is up (a `-readonly` open of a closed WAL DB fails with error 14) — `sqlite3 -readonly data/state/vibe_quant.db "SELECT DISTINCT json_extract(node_status,'$.account') FROM paper_trading_checkpoints WHERE json_valid(node_status) AND json_extract(node_status,'$.account') LIKE 'binance:live:%'"` lists the live key ids still in the checkpoint history (pruning keeps the newest rows per trader id); more than one id for one Binance account means two keys traded it (`printf %s "$KEY" | shasum -a 256 | cut -c1-16` maps a key to its id); (2) rotate — keep the same `trader_id`: the rotation carries that trader's own risk bookkeeping and halt to the new key (a halted trader comes back HALTED and is resumed deliberately; a new `trader_id` on the new key has nothing to carry and needs `confirm_fresh_live=true`), so flatten first with `POST /api/paper/halt` unless you accept positions unguarded until the new start; then stop every LIVE session, revoke the old key in Binance API Management, put the new key in the backend env, restart the backend, and `POST /api/paper/start` with `testnet=false`, `confirm_live=true`, the same `trader_id` and the strategy/validation run (no `confirm_fresh_live`: the same `trader_id` carries its risk and halt, so no consent is asked); never run two keys of one account at once; (3) run every session as the same OS user (sudo or a LaunchDaemon resolves another lock dir). The paper child records its own pid and process start time in its job row before trading and refuses to trade when it cannot, so /stop, kill and the kill-switch cascade always reach a trading session (bead vibe-quant-qgtdz); one session per trader id as well: a second process for the same trader id (another key, or testnet beside live) exits 1 with `TraderBusyError` (bead vibe-quant-2q7cv).
- **Telegram start gate:** paper and live refuse to start when Telegram alerts are **configured but not working** — verified with one silent `sendChatAction` probe (checks the token, the chat id and the permission to post) before the trading node is built, so 0 orders are structurally possible and an expired token or a wrong chat id can never silently drop a CRITICAL alert (halt, kill switch, liquidation). Exactly one of `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` set is refused too; with neither set (or both empty) the session starts as before with alerts disabled. `POST /api/paper/start` and `/restore` return 503 before any run row is written. A restore after a crash is refused too; positions stay at the venue until Telegram works or both vars are unset and the backend restarted. The way-out is a backend env change (restart the backend). Refusal messages never contain the bot token (bead vibe-quant-u2r9r).

---

## 12. Observability & Alerts

### Structured Event Logging

Every significant event is logged as a JSON line. A validation run writes `logs/events/{run_id}.jsonl`; a paper session writes `logs/paper/{trader_id}.jsonl` (its `logs_path`, stored in the run parameters; reconciliation reads that file). `VIBE_QUANT_LOG_DIR` moves both:

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

**Goal:** Run full-fidelity NautilusTrader backtests with custom fills, own-1m-detail-bar order release (latency presets apply to tick data only), position sizing, and risk management.

### Deliverables

1. **Validation venue configuration**
   - Custom `FillModel`: volume-based slippage with square-root market impact
   - Fee model: Binance VIP0 maker 0.02% / taker 0.05% (`vibe_quant/fees.py`)
   - `LatencyModelConfig` with presets, tick data only (co-located, domestic, international, retail, custom)
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
   - Configures BacktestEngine with venue (including fill model; the latency preset applies to tick data only), data, strategy
   - Runs backtest, extracts results
   - Stores results in SQLite (backtest_results + trades tables)
   - Writes structured event log

5. **Screening-to-validation consistency check**
   - Compare screening results vs validation results for same parameters
   - Report: which candidates improved/degraded when moving to full-fidelity execution
   - Expected: validation results generally worse due to realistic fills, order-release timing, and slippage (latency preset applies to tick data only)
   - Flag candidates that degrade > 50% as "execution-sensitive"

### Acceptance Criteria

- DSL compiles to valid NautilusTrader Strategy that runs without errors
- Validation backtest produces realistic P&L including fees, slippage, funding, and order-release timing effects
- Position sizing correctly limits risk per trade and total exposure
- Risk circuit breakers trigger at configured thresholds
- Latency preset visibly affects tick-data results (bar validation releases orders on the instrument's own next 1m detail bar)
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
   - Compute WF efficiency: `mean(OOS_return / OOS_days) / mean(IS_return / IS_days)`
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
   - Node state machine (`NodeState`, `paper/node.py`): INITIALIZING -> RUNNING; RUNNING -> PAUSED (reduce-only, `POST /api/paper/halt` with `mode=pause`) or HALTED (flatten, stop strategies; a daily-loss halt keeps strategies running and lifts at the next UTC midnight) -> manual RESUME; STOPPED and ERROR end a session

5. **Paper trading dashboard tab**
   - Live P&L display
   - Open positions table
   - Recent trades list
   - Node status (RUNNING/PAUSED/HALTED; also INITIALIZING/STOPPED/ERROR)
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
   - **Selection**: deterministic crowding by default (each offspring replaces its most similar parent when at least as fit; 1 elite kept); `--no-crowding` (API `crowding_enabled=false`) = classic tournament (`tournament_size`, default 3). Both rank by the scalar `adjusted_score`; there is no Pareto ranking in discovery (Pareto fronts exist only in screening sweeps)
   - **Elitism**: tournament mode keeps the top `elite_count` (default 2) unchanged; crowding keeps 1

3. **Fitness function**
   - NautilusTrader screening-mode backtest of each candidate (parallel via multiprocessing)
   - Scalar fitness (`discovery/fitness.py`): `raw = 0.35·Sharpe_n + 0.25·(1 - MaxDD) + 0.20·PF_n + 0.20·Return_n` (Sharpe clamped to [-1, 4], PF to [0, 5], return to [-100%, +200%], each scaled to [0, 1]); `adjusted_score = max(0, raw - complexity - overtrade - SL/TP-imbalance penalties)`, and 0 below `min_trades` (50) or at return <= 0. With `eval_windows` > 1 the metrics are worst-of-N windows
   - Penalties: complexity 0.02 per scalar gene above 2 (cap 0.1); overtrade 0.05 per 100 trades above the timeframe threshold (cap 0.3); SL/TP imbalance 0.02 per unit of SL/TP ratio above 5 (cap 0.15)
   - Gates are not part of the score: they run once on the final top-K (guard rails below)

4. **Discovery pipeline**
   - Initialize population (N random valid strategies from grammar rules)
   - For each generation:
     - Evaluate fitness via NautilusTrader screening mode (parallel; `--evaluator fast` scores with the fast tier and re-runs the top-K and every cached genome with an MTM series on NT before any gate (each NT result must equal the fast one in every field, else the run fails))
     - Breed offspring (crossover + mutation) and select (crowding or tournament, above)
     - Inject random immigrants when population entropy falls below `entropy_threshold` (0.4)
   - Terminate after M generations, or when the best score has not improved over the last `convergence_generations` (capped at M // 2; checked from generation 2N on)
   - Output: top K strategies as DSL YAML for validation

5. **Dashboard integration**
   - Discovery tab in Streamlit
   - Configure: population size, generations, indicator pool, symbols, timeframe
   - Live generation-by-generation progress visualization
   - Fitness evolution chart (best/mean/worst per generation)
   - Final results: top discovered strategies with full metrics

6. **Guard rails**
   - Minimum 50 trades per candidate (prevents overfitting to few trades)
   - Complexity penalty prevents bloated strategies, plus a hard cap of <= 10 scalar genes (entry + exit, long + short; MA genes excluded) enforced inside the GA (init, mutation, crossover, seeding trim the longest list's tail without RNG); the final guardrail is a fail-closed backstop
   - DSR applied to the whole discovery run on its **effective** trial count: `notes.dsr_trials.run_n_eff` (this run's EV-matched count; raw when fewer than 2 retained series) plus the summed `run_n_eff` of earlier same-timeframe overlapping-train runs recorded under `n_rule` `cumulative_effective_same_tf_overlapping_train`; earlier runs are tallied (`prior_legacy_raw`) but not charged (see `docs/claude/discovery.md`, comparability break 2026-10-18)
   - Rolling-window stability check (opt-in: `--wfa-oos-step-days N`, API `wfa_oos_step_days`; default 0 = off): tiles the TRAIN range (never the holdout) with back-to-back N-day windows, re-runs each champion unchanged on every window (no re-optimisation, no efficiency ratio) and passes when at least `wfa_min_consistency` (default 0.75) of the windows have return > 0; a range too short for one window fails closed. The windows are in-sample for the GA, so this measures stability, not out-of-sample skill (the holdout is that gate). §8's walk-forward analysis runs only in the overfitting CLI (`python -m vibe_quant overfitting`); promotion does not require it
   - Multi-symbol scoring (`--symbol-agg`, API `symbol_agg`): `portfolio` (default) runs ONE shared-account backtest over all symbols, so one symbol can carry the rest; `worst` (opt-in) backtests each symbol alone (all eval windows per symbol) and scores the genome by its worst symbol: min Sharpe / return / PF, max drawdown, MIN trades across symbols (so `min_trades` applies per symbol), bootstrap trade returns concatenated. The GA RANK score in worst mode is soft (a hard worst-of score is 0 whenever any symbol loses, so the GA had no gradient): `adjusted_score = 0.5*min + 0.5*median` of the PER-SYMBOL adjusted fitness (each symbol scored exactly like a standalone run on its own window-aggregated metrics, incl. its hard trade/return gate and penalties; recorded in notes as `worst_mode_score` and per champion as `symbol_scores`); the reported metrics stay worst-of, and there is no across-symbol early exit (per-window early exit inside a symbol still applies). Champions keep a HARD worst-symbol train gate before the other gates: every symbol's train worst-of-windows return must be > 0 and its trades >= `min_trades`, else the candidate is rejected (`guardrail_rejections` stage `worst_symbol_train`, e.g. "worst-symbol train return <= 0 (SYM)"); holdout/bootstrap/DSR/cross-window/WFA gates are unchanged. Gate-passing candidates are ranked ahead of gate-failing ones when picking top-K. The same mode is used by every backtest fn of the run (GA, holdout, full-range headline, cross-window, WFA) and is recorded in the run notes as `symbol_agg`; a replay of such a run is a portfolio backtest and carries a `metrics_note`. Single-symbol runs are identical in both modes. In worst mode the GA soft score applies each symbol's own penalties (overtrade on that symbol's trades); only the reported worst-of fields and the hard worst-symbol champion gate use per-symbol minimums. Worst-mode `full_range`/`holdout` entries also persist `*trades_sum` and per-symbol `symbol_metrics`; validation consistency and replay-drift compare trades against `trades_sum` (skipped, with a note, for legacy notes without it) because validation/replay are portfolio runs, and read the reference Sharpe as the worst symbol.

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
| 1 | Core engine | NautilusTrader (single engine) | Execution realism critical for leveraged perps (run leverage 1-125x, default 10x); Rust performance; backtest-live parity; models funding always and liquidation as an opt-in overlay, even during screening |
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
