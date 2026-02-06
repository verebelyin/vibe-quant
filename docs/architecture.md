# Architecture

> For the full implementation spec, see [`SPEC.md`](../SPEC.md).

## Overview

vibe-quant is a single-engine trading system built on NautilusTrader (Rust core). The engine operates in two modes -- screening (fast, parallel) and validation (full-fidelity) -- sharing the same strategy code throughout.

```
Strategy DSL (YAML)
    |
    v
DSL Compiler --> NautilusTrader Strategy subclass
    |
    +---> Screening Pipeline (simplified fills, multiprocessing)
    |         |
    |         v
    |     Overfitting Filters (DSR, WFA, Purged K-Fold)
    |         |
    |         v
    +---> Validation Pipeline (custom fills, latency, full cost)
              |
              v
          Paper Trading (Binance testnet) --> Live Execution
```

## Core Design Principles

1. **Single engine, two fidelities** -- NautilusTrader for both screening and validation. Leverage, funding rates, and liquidation modeled even during screening.
2. **Single source of truth** -- strategy logic defined once in YAML DSL, auto-compiled to NautilusTrader Strategy subclasses.
3. **Separation of concerns** -- strategy signals decoupled from position sizing, risk management, and execution. Each is a pluggable module.
4. **Defense in depth** -- multi-layer overfitting prevention with toggleable filters. Dual-level risk circuit breakers (strategy + portfolio).
5. **Data reproducibility** -- raw downloaded data archived in SQLite. Catalog rebuildable from archive at any time.

---

## System Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT DASHBOARD                          │
│  Launch backtests | Configure params | View results | Monitor paper │
└────────┬─────────────────────────┬──────────────────────┬───────────┘
         │                         │                      │
         v                         v                      v
┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  STRATEGY DSL   │  │  SCREENING PIPELINE  │  │  RESULTS & ANALYTICS │
│  (YAML)         │──>  (NT Screening Mode) │──>  Multi-metric ranking │
│  Indicators +   │  │  Parallel param sweep│  │  Pareto front        │
│  Conditions +   │  │  Simplified fills    │  │  SQLite storage      │
│  Actions        │  │  Full margin/funding │  │                      │
│  Multi-TF +     │  └──────────┬───────────┘  └──────────────────────┘
│  Time filters   │             │
└────────┬────────┘             v
         │            ┌──────────────────────┐
         │            │  OVERFITTING FILTERS │
         │            │  Deflated Sharpe     │
         │            │  Walk-Forward        │
         │            │  Purged K-Fold CV    │
         │            │  (all toggleable)    │
         │            └──────────┬───────────┘
         │                       │
         v                       v
┌─────────────────────────────────────────────────────────────────────┐
│                NAUTILUSTRADER ENGINE (Validation Mode)               │
│  Event-driven backtesting | Realistic execution simulation          │
│  Custom FillModel | LatencyModel | Fees + Funding | Liquidation     │
├─────────────────────────────────────────────────────────────────────┤
│  Position Sizing     │  Risk Management      │  Event Logger        │
│  (Kelly/FF/ATR)      │  (Strategy+Portfolio)  │  (Structured JSON)   │
└────────┬────────────────────────┬────────────────────────────────────┘
         │                        │
         v                        v
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│  BINANCE ADAPTER │  │  DATA LAYER                                  │
├──────────────────┤  │  ParquetDataCatalog (multi-TF bars)          │
│  ETHEREAL ADAPTER│  │  Raw Data Archive (SQLite, immutable)        │
└──────────────────┘  │  DuckDB query layer (research)               │
                      └──────────────────────────────────────────────┘
```

---

## Module Responsibilities

### `data/` -- Data Ingestion & Archive

Downloads OHLCV klines and funding rates from Binance (and Ethereal). Raw data stored in immutable SQLite archive before processing into NautilusTrader's ParquetDataCatalog.

- **downloader.py**: Binance API client (bulk CSV from data.binance.vision + REST API for gaps)
- **archive.py**: SQLite raw data archive (`raw_klines`, `raw_funding_rates` tables)
- **catalog.py**: Builds ParquetDataCatalog from archived data (1m, 5m, 15m, 1h, 4h bars)
- **verify.py**: Gap detection, OHLC consistency checks, data quality validation
- **ingest.py**: CLI orchestration (download -> archive -> catalog -> verify)

### `dsl/` -- Strategy DSL

Declarative YAML strategies compiled to NautilusTrader Strategy subclasses.

- **schema.py**: Pydantic models for strategy definition (indicators, conditions, time filters, sweep ranges)
- **parser.py**: YAML parsing and validation with error messages
- **indicators.py**: Indicator registry mapping DSL types to NautilusTrader classes
- **conditions.py**: Condition parser (`crosses_above`, `<`, `>`, `between`, etc.)
- **compiler.py**: Generates NautilusTrader Strategy subclass with `on_start()`, `on_bar()`, multi-TF subscriptions

### `screening/` -- Parameter Sweep

Fast parameter sweeps using NautilusTrader in screening mode (simplified fills, no latency) with multiprocessing parallelism.

- **pipeline.py**: Builds parameter grid, distributes across CPU cores, collects results
- **consistency.py**: Pareto front ranking (Sharpe, 1-MaxDD, ProfitFactor), hard filters (min trades, max DD)

Screening still models leverage, margin, funding rates, and liquidation -- unlike vectorized alternatives.

### `validation/` -- Full-Fidelity Backtesting

Top screening candidates validated with realistic execution simulation.

- **runner.py**: Configures and runs NautilusTrader BacktestEngine with full cost modeling
- **fill_model.py**: Custom FillModel with volume-based slippage (`spread/2 + 0.1 * volatility * sqrt(order_size / avg_volume)`)
- **latency.py**: LatencyModel presets (co-located 1ms, domestic 20ms, international 100ms, retail 200ms)
- **venue.py**: Binance futures venue configuration (margin, leverage, fees)

### `overfitting/` -- Statistical Validation

Three independently toggleable filters between screening and validation.

- **dsr.py**: Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). Corrects for multiple-testing bias. Threshold: DSR > 0.95.
- **wfa.py**: Walk-Forward Analysis. Sliding 9m train / 3m test / 1m step windows (~13 windows over 2 years). Threshold: WF efficiency > 0.5, >50% windows profitable.
- **purged_kfold.py**: Purged K-Fold CV (K=5). Embargo periods prevent information leakage. Threshold: mean OOS Sharpe > 0.5, std < 1.0.
- **pipeline.py**: Orchestrates filter chain, tags results pass/fail per filter.

### `risk/` -- Position Sizing & Risk Management

Pluggable modules decoupled from strategy logic.

- **sizing.py**: Three position sizers:
  - Fixed Fractional: `position = (equity * risk_pct) / (entry - stop_loss)`
  - Kelly Criterion: `f* = (W - (1-W)/R) * kelly_fraction` (half-Kelly default)
  - ATR-Based: `position = (equity * risk_pct) / (ATR * multiplier)`
- **actors.py**: RiskActor for strategy-level checks (drawdown halt, daily loss limit, consecutive losses) and portfolio-level checks (total exposure, correlated positions)
- **config.py**: Risk configuration schemas

### `discovery/` -- Genetic Strategy Optimization

Automated strategy discovery using evolutionary algorithms.

- **genome.py**: Strategy encoded as chromosome of (indicator, parameter, condition) genes
- **operators.py**: Tournament selection, crossover, mutation, elitism
- **fitness.py**: Multi-objective scoring (Sharpe, MaxDD, ProfitFactor) with complexity penalty
- **guardrails.py**: Min 50 trades, DSR correction for total candidates tested
- **pipeline.py**: Population initialization, generation loop, convergence detection

### `paper/` -- Paper Trading

NautilusTrader TradingNode connected to Binance testnet for live-data paper trading.

- **node.py**: TradingNode setup with WebSocket data + simulated execution
- **persistence.py**: State save/restore to SQLite for crash recovery
- **config.py**: Paper trading configuration (testnet credentials, strategy selection)
- **cli.py**: CLI entry point for spawning paper trading subprocess

### `ethereal/` -- Ethereal DEX Integration

Custom NautilusTrader adapter for Ethereal decentralized exchange.

- **data_client.py**: HTTP/WebSocket client for market data
- **execution_client.py**: Order submission with EIP-712 signed authentication
- **instruments.py**: Token/pair metadata (BTC/ETH/SOL, 0%/0.03% maker/taker, 1h funding)
- **venue.py**: Venue configuration and latency presets
- **ingestion.py**: Historical data download from Ethereal archive

### `dashboard/` -- Streamlit UI

Full lifecycle management across 7 pages.

| Page | Purpose |
|------|---------|
| Strategy Management | Create/edit YAML strategies, version history |
| Backtest Launch | Configure and run screening/validation sweeps |
| Results Analysis | Equity curves, Pareto fronts, trade logs, metrics |
| Discovery | Genetic optimizer configuration and progress |
| Paper Trading | Live P&L, open positions, strategy controls |
| Data Management | Catalog status, data updates, quality checks |
| Settings | Sizing/risk presets, Telegram config, system info |

### `jobs/` -- Background Job Management

- **manager.py**: Subprocess spawning with PID tracking, heartbeat monitoring (30s interval), stale job cleanup (120s timeout)

### `logging/` -- Event Logging

- **events.py**: Typed event dataclasses (TradeEvent, OrderEvent, RiskEvent)
- **writer.py**: SQLite trade log writer
- **query.py**: Query and compute metrics from trade logs

### `alerts/` -- Telegram Notifications

- **telegram.py**: Alert service for errors, circuit breakers, daily P&L summaries. Rate-limited (1 per type per minute).

---

## Data Architecture

```
data/
├── catalog/                    # NautilusTrader ParquetDataCatalog
│   └── data/
│       ├── bar/                # OHLCV bars by symbol and timeframe
│       │   ├── btcusdt-perp.binance/
│       │   │   ├── 1-MINUTE-LAST/
│       │   │   ├── 5-MINUTE-LAST/
│       │   │   ├── 15-MINUTE-LAST/
│       │   │   ├── 1-HOUR-LAST/
│       │   │   └── 4-HOUR-LAST/
│       │   ├── ethusdt-perp.binance/
│       │   └── solusdt-perp.binance/
│       ├── funding_rate_update/
│       └── instrument/
├── archive/
│   └── raw_data.db            # SQLite: immutable raw CSV/API archive
├── state/
│   └── vibe_quant.db          # SQLite (WAL): strategies, results, configs
└── logs/
    └── events/                # Structured JSON event logs
```

**Data flow**: Binance API -> raw SQLite archive -> ParquetDataCatalog -> backtesting engine

**Symbols**: BTCUSDT, ETHUSDT, SOLUSDT (USDT-M perpetuals)
**History**: 2 years, 5 timeframes (1m, 5m, 15m, 1h, 4h)
**Volume**: ~3.85M bars (~40-80 MB compressed Parquet)

---

## State Management

All state in SQLite (WAL mode, `busy_timeout=5000`). Key tables:

| Table | Purpose |
|-------|---------|
| `strategies` | DSL configs (JSON), versioned |
| `sizing_configs` | Position sizing presets |
| `risk_configs` | Risk management presets |
| `backtest_runs` | Run metadata (strategy, params, status, PID) |
| `backtest_results` | Performance metrics per run |
| `trades` | Individual trade records (entry, exit, fees, funding) |
| `sweep_results` | Bulk screening results with Pareto flags |
| `background_jobs` | Subprocess tracking (PID, heartbeat, status) |

---

## Screening vs Validation

| Aspect | Screening | Validation |
|--------|-----------|------------|
| Purpose | Fast parameter exploration | Realistic performance estimate |
| Fill model | Simple probabilistic slippage | Volume-based (square-root impact) |
| Latency | None | Configurable (1ms - 200ms) |
| Fee model | Standard | Binance maker/taker tiers |
| Parallelism | Multiprocessing across cores | Single run |
| Leverage/funding/liquidation | Yes | Yes |
| Output | Pareto-ranked candidates | Detailed results + trade log |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single engine | NautilusTrader | Execution realism critical at 20x leverage; Rust perf; backtest-live parity |
| Screening approach | NT screening mode + multiprocessing | Correctly models margin/funding/liquidation unlike vectorized alternatives |
| Strategy format | Declarative YAML DSL | Single source compiles to NT Strategy; enables UI form generation |
| State storage | SQLite (WAL) | Zero infrastructure; concurrent read/write from dashboard + backtests |
| Data storage | ParquetDataCatalog + raw archive | Native NT integration; archive enables reproducibility |
| Overfitting | Full pipeline, toggleable | Walk-Forward + DSR + Purged K-Fold; each independently controllable |
| Risk management | Strategy + portfolio level | Defense in depth; dual circuit breakers |
| Background jobs | Subprocess + PID + heartbeat | No extra infra (no Redis/Celery) |
