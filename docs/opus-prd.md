# Algorithmic Trading Backtesting Framework
## Product Requirements Document & Technical Specification

The optimal path forward combines **FreqTrade** as the core backtesting engine with a custom orchestration layer for managing 100-200 strategies across 200-500 symbols. This hybrid approach delivers **80% of functionality out-of-the-box** while requiring only 8-12 weeks of custom development for the remaining requirements—versus 16-24+ weeks for a fully custom build.

---

# Part 1: Product Requirements Document

## Executive summary

This PRD defines requirements for an automated algorithmic trading backtesting framework supporting Binance and Ethereal exchanges. The system must handle **100-200 strategies**, **200-500 symbols**, and **5 years of 1-minute historical data** with realistic simulation of leverage trading up to 20x, including fees, slippage, funding rates, and network latency. Key capabilities include paper trading with real-time data, comprehensive statistics, and bankroll management strategies.

**Target users:** Quantitative traders and algorithmic trading teams requiring institutional-grade backtesting with crypto-specific features (perpetual futures, funding rates, high leverage).

## Core user stories

**Strategy Development:**
- As a quant, I want to write trading strategies in Python/TypeScript and backtest them against historical data so I can evaluate performance before risking capital
- As a strategy developer, I want to test multiple parameter combinations in parallel so I can optimize strategy performance efficiently
- As a researcher, I want to combine multiple strategies into portfolios so I can evaluate diversification benefits

**Realistic Simulation:**
- As a trader, I want backtests to include trading fees, slippage, and funding rates so results reflect real-world conditions
- As a leverage trader, I want accurate margin calculations and liquidation simulation so I understand my true risk
- As a risk manager, I want network latency simulation so I can evaluate execution quality degradation

**Operations:**
- As a portfolio manager, I want to manage 100-200 strategies in a database so I can select and combine strategies efficiently
- As an operator, I want paper trading with live data so I can validate strategies before live deployment
- As an analyst, I want comprehensive performance reports with Sharpe ratio, drawdown analysis, and trade statistics

## Functional requirements

### Data Management (P0 - Critical)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| DM-1 | Support Binance spot and futures data | Fetch 5 years of 1-minute OHLCV data for 500 symbols |
| DM-2 | Support Ethereal exchange data | Integrate via REST/WebSocket APIs with EIP-712 authentication |
| DM-3 | Local data storage | Store data in TimescaleDB/Parquet with <3 GB compressed footprint |
| DM-4 | Data normalization | Unified schema across exchanges with timestamp, OHLCV, volume |

### Backtesting Engine (P0 - Critical)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| BE-1 | Event-driven architecture | Same code works for backtest and paper trading |
| BE-2 | Support 1-minute+ candles | Configurable timeframes from 1m to 1d |
| BE-3 | Parallel symbol processing | Run 200-500 symbols concurrently |
| BE-4 | 5-year backtest execution | Complete single-strategy backtest in <5 minutes |

### Trading Simulation (P0 - Critical)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| TS-1 | Fee simulation | Configurable maker/taker fees (0.02%-0.10%), BNB discounts |
| TS-2 | Slippage modeling | Volume-based slippage calculation using order book depth estimation |
| TS-3 | Stop loss/Take profit | Fixed, trailing, and ATR-based stop types |
| TS-4 | Leverage trading (1-20x) | Isolated/cross margin, liquidation price calculation |
| TS-5 | Funding rate simulation | 8-hour funding payments for perpetual futures |
| TS-6 | Interest calculations | Track borrowing costs for margin positions |

### Strategy Management (P1 - High)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| SM-1 | Strategy database | Store 100-200 strategies with parameters, versions, metadata |
| SM-2 | Strategy selection/combination | Select strategies by type, performance metrics, parameters |
| SM-3 | Parameter versioning | Track parameter changes with backtest result linkage |

### Risk & Bankroll (P1 - High)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| RB-1 | Position sizing methods | Kelly Criterion, fixed fractional, volatility-based |
| RB-2 | Risk limits | Max position size, daily loss limits, drawdown-based pausing |
| RB-3 | Portfolio risk management | Correlation-adjusted position limits, portfolio heat tracking |

### Paper Trading (P1 - High)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| PT-1 | Real-time data ingestion | WebSocket feeds from Binance/Ethereal |
| PT-2 | Simulated order matching | Price-time priority with realistic fills |
| PT-3 | State persistence | Redis-backed position and order state |

### Reporting & Analytics (P1 - High)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| RA-1 | Performance metrics | Sharpe, Sortino, Calmar, max drawdown, profit factor |
| RA-2 | Trade statistics | Win rate, avg win/loss, trade duration, consecutive wins/losses |
| RA-3 | Visualizations | Equity curves, drawdown charts, monthly heatmaps |
| RA-4 | Benchmark comparison | Compare against buy-and-hold and custom benchmarks |

### Network Simulation (P2 - Medium)
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| NS-1 | Latency simulation | Configurable mean latency (1-50ms) with variance |
| NS-2 | Network spikes | Random latency spikes (1% probability, 10x multiplier) |

## Non-functional requirements

- **Performance:** Single strategy backtest (5 years, 500 symbols) completes in <5 minutes
- **Scalability:** Support 200 concurrent strategy backtests with 32-core CPU
- **Storage:** 5 years of 1-minute data for 500 symbols stored in <5 GB (compressed)
- **Reliability:** Paper trading maintains state through restarts via persistence layer
- **Extensibility:** Plugin architecture for new exchanges, indicators, and position sizing methods

---

# Part 2: Technical Specification

## Build vs. buy analysis

### Evaluation of existing solutions

| Solution | Binance Support | Leverage Trading | Paper Trading | 500 Symbol Scale | Strategy DB | Verdict |
|----------|-----------------|------------------|---------------|------------------|-------------|---------|
| **FreqTrade** | ✅ Excellent | ✅ Full (20x) | ✅ Dry-run mode | ⚠️ Multi-instance | 🔧 Custom needed | **Best fit** |
| **Nautilus Trader** | ✅ Stable | ✅ Full | ✅ Sandbox mode | ✅ Native | 🔧 Custom needed | High-performance option |
| **VectorBT** | ✅ Data only | ⚠️ Pro only | ❌ No | ✅ Vectorized | 🔧 Custom needed | Research only |
| **QuantConnect** | ✅ Full | ✅ Full | ✅ Full | ✅ Enterprise | ⚠️ Limited local | Enterprise option |
| **Backtrader** | ⚠️ Via fork | ❌ No | ❌ No | ⚠️ Poor | 🔧 Custom needed | Legacy, avoid |

### Recommended approach: FreqTrade + custom orchestration

**What FreqTrade provides (80% of requirements):**
- Binance futures integration with leverage up to 20x
- Dry-run (paper trading) mode with real-time data
- Fee and slippage simulation (configurable up to 5%)
- Stop loss and take profit (including trailing)
- Comprehensive performance statistics
- Hyperopt parameter optimization
- SQLite trade persistence

**Custom development required (20% of requirements):**
- Strategy orchestration layer for 100-200 strategies
- Multi-symbol parallel processing manager
- Ethereal exchange integration adapter
- Enhanced bankroll management (Kelly Criterion, correlation-based limits)
- Interest rate tracking for leveraged positions
- Network latency simulation layer
- Unified reporting dashboard

**Development effort:**
| Component | Effort | Description |
|-----------|--------|-------------|
| FreqTrade learning/setup | 2 weeks | Configuration, custom strategy development patterns |
| Strategy orchestration layer | 4 weeks | Database, scheduler, parallel execution manager |
| Ethereal adapter | 2 weeks | API integration using ethereal-sdk |
| Enhanced bankroll management | 1 week | Kelly Criterion, correlation limits, drawdown controls |
| Latency simulation | 1 week | Configurable latency injection layer |
| Reporting dashboard | 2 weeks | Streamlit/Dash-based unified reporting |
| **Total MVP** | **12 weeks** | Full requirements coverage |

## System architecture

### High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │ Strategy Manager │  │  Job Scheduler   │  │ Results Aggregator│         │
│  │   (PostgreSQL)   │  │ (Celery/Redis)   │  │    (Dashboard)    │         │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘         │
└───────────┼──────────────────────┼──────────────────────┼───────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKTESTING CORE (FreqTrade)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  Backtest Engine │  │   Paper Trading  │  │  Strategy Base   │         │
│  │  (Event-Driven)  │  │   (Dry-Run Mode) │  │   (IStrategy)    │         │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘         │
└───────────┼──────────────────────┼──────────────────────┼───────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  TimescaleDB     │  │  Parquet Files   │  │  Redis Cache     │         │
│  │  (OHLCV + Meta)  │  │ (Historical Data)│  │  (State/Pub-Sub) │         │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘         │
└───────────┼──────────────────────┼──────────────────────┼───────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXCHANGE ADAPTERS                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │ Binance Adapter  │  │ Ethereal Adapter │  │ Latency Simulator│         │
│  │ (CCXT/Native)    │  │ (ethereal-sdk)   │  │                  │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component specifications

**Strategy Manager (Custom):**
- PostgreSQL database storing 100-200 strategies with JSON parameters
- Strategy versioning with parent-child relationships
- Query interface for filtering by type, performance, parameters
- REST API for strategy CRUD operations

**Job Scheduler (Celery + Redis):**
- Distributes backtest jobs across worker pool
- Priority queue for urgent backtests
- Resource management (memory limits, timeouts)
- Progress tracking and cancellation support

**Backtest Engine (FreqTrade Core):**
- Event-driven backtesting with bar-by-bar processing
- Leverage callback for dynamic leverage per trade
- Custom stop loss callback for trailing/ATR-based stops
- Hyperopt integration for parameter optimization

**Exchange Adapters:**
- Binance: Native FreqTrade integration (spot + futures)
- Ethereal: Custom adapter using ethereal-sdk with EIP-712 signing
- Common interface for data fetching and order simulation

### Data pipeline architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ data.binance.   │     │ Binance REST    │     │ Ethereal Archive│
│ vision (Bulk)   │     │ API (Recent)    │     │ API             │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION SERVICE                       │
│  • Parallel downloads (10 concurrent)                          │
│  • Rate limit management (5000 weight/min for Binance)         │
│  • Data validation and gap detection                           │
│  • Normalization to unified schema                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │ Parquet Files   │  │ TimescaleDB     │  │ Redis           ││
│  │ (Cold Storage)  │  │ (Hot Data)      │  │ (Real-time)     ││
│  │ 5yr history     │  │ 30-day window   │  │ Live streams    ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Data volume estimates:**
- 500 symbols × 5 years × 252 days × 390 minutes = **245 million candles**
- Raw storage: ~12.8 GB uncompressed
- With TimescaleDB compression: **~1.5-3 GB**

## Database schema design

### Strategy management tables

```sql
-- Core strategy definition
CREATE TABLE strategies (
    strategy_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    strategy_type VARCHAR(50),              -- 'momentum', 'mean_reversion', 'arbitrage'
    asset_classes TEXT[],                   -- ARRAY['crypto_spot', 'crypto_futures']
    timeframe VARCHAR(20) NOT NULL,         -- '1m', '5m', '1h'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1,
    code_path VARCHAR(255)                  -- Path to strategy Python file
);

-- Strategy parameters with versioning
CREATE TABLE strategy_parameters (
    param_id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(strategy_id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    parameters JSONB NOT NULL,              -- {'rsi_period': 14, 'rsi_overbought': 70}
    bankroll_config JSONB,                  -- Position sizing configuration
    risk_config JSONB,                      -- Risk management parameters
    created_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT,
    UNIQUE(strategy_id, version)
);

-- Backtest results with comprehensive metrics
CREATE TABLE backtest_results (
    backtest_id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(strategy_id),
    param_version INTEGER,
    exchange VARCHAR(20) NOT NULL,          -- 'binance', 'ethereal'
    symbols TEXT[] NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2),
    final_capital DECIMAL(15,2),
    
    -- Performance metrics
    total_return DECIMAL(10,4),
    cagr DECIMAL(10,4),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(10,4),
    max_drawdown_duration_days INTEGER,
    volatility_annual DECIMAL(10,4),
    
    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(6,4),
    profit_factor DECIMAL(8,4),
    avg_win DECIMAL(15,2),
    avg_loss DECIMAL(15,2),
    largest_win DECIMAL(15,2),
    largest_loss DECIMAL(15,2),
    avg_trade_duration_hours DECIMAL(10,2),
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    
    -- Costs breakdown
    total_fees_paid DECIMAL(15,2),
    total_funding_paid DECIMAL(15,2),
    total_slippage_cost DECIMAL(15,2),
    
    -- Execution metadata
    execution_time_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    FOREIGN KEY (strategy_id, param_version) 
        REFERENCES strategy_parameters(strategy_id, version)
);

-- Individual trade records
CREATE TABLE trades (
    trade_id SERIAL PRIMARY KEY,
    backtest_id INTEGER REFERENCES backtest_results(backtest_id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,         -- 'LONG', 'SHORT'
    leverage INTEGER DEFAULT 1,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    entry_price DECIMAL(15,8) NOT NULL,
    exit_price DECIMAL(15,8),
    quantity DECIMAL(15,8) NOT NULL,
    initial_margin DECIMAL(15,2),
    
    -- Costs
    entry_fee DECIMAL(10,4),
    exit_fee DECIMAL(10,4),
    funding_fees DECIMAL(10,4),
    slippage_cost DECIMAL(10,4),
    
    -- Results
    gross_pnl DECIMAL(15,2),
    net_pnl DECIMAL(15,2),
    roi_percent DECIMAL(10,4),
    exit_reason VARCHAR(50),                -- 'signal', 'stop_loss', 'take_profit', 'liquidation'
    
    INDEX idx_trades_backtest (backtest_id),
    INDEX idx_trades_symbol (symbol),
    INDEX idx_trades_entry (entry_time)
);
```

### OHLCV data tables (TimescaleDB)

```sql
-- Market data hypertable
CREATE TABLE ohlcv_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    open DECIMAL(15,8) NOT NULL,
    high DECIMAL(15,8) NOT NULL,
    low DECIMAL(15,8) NOT NULL,
    close DECIMAL(15,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    quote_volume DECIMAL(20,8),
    num_trades INTEGER
);

-- Convert to TimescaleDB hypertable with weekly chunks
SELECT create_hypertable('ohlcv_data', 'time', 
    chunk_time_interval => INTERVAL '1 week');

-- Enable compression (90%+ space savings)
ALTER TABLE ohlcv_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,exchange'
);
SELECT add_compression_policy('ohlcv_data', INTERVAL '7 days');

-- Continuous aggregate for hourly data (faster queries)
CREATE MATERIALIZED VIEW ohlcv_1h WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    exchange,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume
FROM ohlcv_data
GROUP BY bucket, symbol, exchange;
```

## Trading mechanics implementation

### Leverage and margin calculations

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

@dataclass
class Position:
    symbol: str
    side: str                    # 'long' or 'short'
    entry_price: Decimal
    quantity: Decimal
    leverage: int
    margin_type: str             # 'isolated' or 'cross'
    initial_margin: Decimal
    maintenance_margin_rate: Decimal = Decimal('0.005')  # 0.5% default
    
    @property
    def position_value(self) -> Decimal:
        return self.quantity * self.entry_price
    
    @property
    def maintenance_margin(self) -> Decimal:
        return self.position_value * self.maintenance_margin_rate
    
    def liquidation_price(self, extra_margin: Decimal = Decimal('0')) -> Decimal:
        """Calculate liquidation price for isolated margin position."""
        margin_diff = self.initial_margin - self.maintenance_margin + extra_margin
        
        if self.side == 'long':
            return self.entry_price - (margin_diff / self.quantity)
        else:  # short
            return self.entry_price + (margin_diff / self.quantity)
    
    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL at current price."""
        if self.side == 'long':
            return self.quantity * (current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - current_price)
    
    def roi_percent(self, current_price: Decimal) -> Decimal:
        """Calculate ROI as percentage of margin (leverage-amplified)."""
        pnl = self.unrealized_pnl(current_price)
        return (pnl / self.initial_margin) * 100


class FeeCalculator:
    """Calculate trading fees including maker/taker and funding."""
    
    def __init__(self, maker_fee: Decimal = Decimal('0.0002'),
                 taker_fee: Decimal = Decimal('0.0005'),
                 funding_interval_hours: int = 8):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.funding_interval_hours = funding_interval_hours
    
    def trading_fee(self, notional: Decimal, is_maker: bool) -> Decimal:
        rate = self.maker_fee if is_maker else self.taker_fee
        return notional * rate
    
    def funding_fee(self, position_value: Decimal, funding_rate: Decimal,
                    hours_held: int) -> Decimal:
        """Calculate cumulative funding fees for position duration."""
        periods = hours_held / self.funding_interval_hours
        return position_value * funding_rate * Decimal(str(periods))


class SlippageModel:
    """Volume-based slippage estimation."""
    
    def __init__(self, base_slippage_bps: Decimal = Decimal('1'),
                 volume_impact_factor: Decimal = Decimal('0.1')):
        self.base_slippage = base_slippage_bps / 10000
        self.volume_impact_factor = volume_impact_factor
    
    def estimate(self, order_size: Decimal, avg_daily_volume: Decimal,
                 volatility: Optional[Decimal] = None) -> Decimal:
        """Estimate slippage as fraction of price."""
        volume_ratio = order_size / avg_daily_volume
        volume_impact = self.volume_impact_factor * (volume_ratio ** Decimal('0.5'))
        
        volatility_impact = Decimal('0')
        if volatility:
            volatility_impact = volatility * Decimal('0.5')
        
        return self.base_slippage + volume_impact + volatility_impact
```

### Bankroll management implementation

```python
from enum import Enum
from typing import List, Dict
import math

class SizingMethod(Enum):
    FIXED_FRACTIONAL = 'fixed_fractional'
    KELLY = 'kelly'
    VOLATILITY_BASED = 'volatility_based'

@dataclass
class BankrollConfig:
    sizing_method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    risk_per_trade: Decimal = Decimal('0.02')      # 2% risk per trade
    kelly_fraction: Decimal = Decimal('0.5')       # Half-Kelly
    max_position_pct: Decimal = Decimal('0.10')    # 10% max single position
    max_portfolio_risk: Decimal = Decimal('0.06')  # 6% total portfolio at risk
    max_daily_loss: Decimal = Decimal('0.02')      # 2% daily loss limit
    max_drawdown_pause: Decimal = Decimal('0.15')  # Pause at 15% drawdown
    max_leverage: int = 20


class BankrollManager:
    """Position sizing and risk management."""
    
    def __init__(self, config: BankrollConfig, initial_capital: Decimal):
        self.config = config
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trade_history: List[Dict] = []
    
    def fixed_fractional_size(self, entry_price: Decimal, 
                               stop_loss: Decimal) -> Decimal:
        """Calculate position size using fixed fractional method."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return Decimal('0')
        
        dollar_risk = self.equity * self.config.risk_per_trade
        position_size = dollar_risk / risk_per_share
        
        # Apply maximum position limit
        max_position = self.equity * self.config.max_position_pct / entry_price
        return min(position_size, max_position)
    
    def kelly_size(self, win_rate: Decimal, avg_win: Decimal, 
                   avg_loss: Decimal) -> Decimal:
        """Calculate position size using Kelly Criterion."""
        if avg_loss == 0:
            return Decimal('0')
        
        win_loss_ratio = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        kelly_pct = max(Decimal('0'), kelly_pct)  # No negative sizing
        
        # Apply fractional Kelly
        adjusted_kelly = kelly_pct * self.config.kelly_fraction
        
        return self.equity * adjusted_kelly
    
    def volatility_size(self, atr: Decimal, entry_price: Decimal,
                        atr_multiplier: Decimal = Decimal('2')) -> Decimal:
        """Calculate position size based on ATR volatility."""
        dollar_risk = self.equity * self.config.risk_per_trade
        stop_distance = atr * atr_multiplier
        
        position_value = dollar_risk / (stop_distance / entry_price)
        return position_value / entry_price
    
    @property
    def current_drawdown(self) -> Decimal:
        if self.peak_equity == 0:
            return Decimal('0')
        return (self.peak_equity - self.equity) / self.peak_equity
    
    def is_trading_allowed(self) -> bool:
        """Check if trading should be paused due to drawdown."""
        return self.current_drawdown < self.config.max_drawdown_pause
    
    def drawdown_scaling_factor(self) -> Decimal:
        """Scale position sizes based on current drawdown."""
        dd = self.current_drawdown
        if dd >= self.config.max_drawdown_pause:
            return Decimal('0')
        elif dd >= Decimal('0.10'):  # 10% drawdown = half size
            return Decimal('0.5')
        return Decimal('1.0')
```

## Exchange integration specifications

### Binance integration (via FreqTrade)

FreqTrade provides native Binance integration. Key configuration:

```json
{
    "exchange": {
        "name": "binance",
        "key": "${BINANCE_API_KEY}",
        "secret": "${BINANCE_API_SECRET}",
        "ccxt_config": {
            "enableRateLimit": true,
            "options": {
                "defaultType": "future"
            }
        }
    },
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "stake_currency": "USDT",
    "dry_run": true
}
```

**Historical data fetching strategy:**
1. Use `data.binance.vision` for bulk historical data (no rate limits)
2. Download monthly archives in parallel (10 concurrent connections)
3. Use REST API for recent data (<30 days)
4. Store in Parquet files for efficient access

### Ethereal integration (custom adapter)

```python
from ethereal_sdk import EtherealClient
from typing import List, Dict
import asyncio

class EtherealAdapter:
    """Adapter for Ethereal exchange integration."""
    
    def __init__(self, private_key: str, testnet: bool = True):
        base_url = "https://api.etherealtest.net" if testnet else "https://api.ethereal.trade"
        self.client = EtherealClient(
            private_key=private_key,
            base_url=base_url
        )
        self.product_cache: Dict[str, str] = {}  # symbol -> product_id
    
    async def initialize(self):
        """Cache product IDs for symbol lookup."""
        products = await self.client.get_products()
        for product in products:
            self.product_cache[product['symbol']] = product['id']
    
    async def fetch_historical_ohlcv(self, symbol: str, 
                                      start_time: int, end_time: int,
                                      interval: str = '1m') -> List[Dict]:
        """Fetch historical OHLCV data from Archive API."""
        # Ethereal uses Archive API for historical data
        # May need to aggregate trades into candles
        product_id = self.product_cache.get(symbol)
        if not product_id:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        # Implementation depends on Archive API structure
        # May need to fetch trades and aggregate to OHLCV
        pass
    
    async def get_market_price(self, symbol: str) -> Dict:
        """Get current market price."""
        product_id = self.product_cache[symbol]
        return await self.client.get_market_price(product_id=product_id)
    
    async def place_order(self, symbol: str, side: str, 
                          quantity: float, price: float = None,
                          order_type: str = 'limit') -> Dict:
        """Place order on Ethereal (uses EIP-712 signing)."""
        product_id = self.product_cache[symbol]
        
        if order_type == 'market':
            return await self.client.place_market_order(
                product_id=product_id,
                side=side,
                quantity=str(quantity)
            )
        else:
            return await self.client.place_limit_order(
                product_id=product_id,
                side=side,
                quantity=str(quantity),
                price=str(price)
            )
```

## Performance optimization strategies

### Parallel processing architecture

```python
from concurrent.futures import ProcessPoolExecutor
from celery import Celery
import redis

# Celery configuration for distributed backtesting
celery_app = Celery('backtester',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task(bind=True)
def run_backtest(self, strategy_id: int, symbols: List[str],
                 start_date: str, end_date: str, params: Dict):
    """Run backtest as Celery task."""
    from freqtrade.optimize.backtesting import Backtesting
    
    config = build_config(strategy_id, symbols, start_date, end_date, params)
    backtesting = Backtesting(config)
    results = backtesting.start()
    
    # Store results in database
    save_backtest_results(strategy_id, results)
    return results['summary']


class BacktestOrchestrator:
    """Orchestrate parallel backtest execution."""
    
    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers
        self.redis = redis.Redis()
    
    async def run_parameter_sweep(self, strategy_id: int, 
                                   param_grid: List[Dict]) -> List[Dict]:
        """Run backtests for all parameter combinations in parallel."""
        tasks = []
        for params in param_grid:
            task = run_backtest.delay(strategy_id, params)
            tasks.append(task)
        
        # Collect results
        results = []
        for task in tasks:
            result = task.get(timeout=3600)  # 1 hour timeout
            results.append(result)
        
        return results
    
    async def run_multi_strategy(self, strategy_ids: List[int],
                                  symbols: List[str],
                                  date_range: tuple) -> Dict:
        """Run multiple strategies in parallel."""
        tasks = [
            run_backtest.delay(sid, symbols, *date_range, {})
            for sid in strategy_ids
        ]
        
        return [task.get(timeout=3600) for task in tasks]
```

### Memory optimization

```python
import pandas as pd
import pyarrow.parquet as pq

class DataLoader:
    """Memory-efficient data loading."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_symbol_data(self, symbol: str, start: str, end: str,
                         columns: List[str] = None) -> pd.DataFrame:
        """Load data with column pruning and date filtering."""
        if columns is None:
            columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        # Use predicate pushdown for efficient filtering
        filters = [
            ('symbol', '=', symbol),
            ('time', '>=', pd.Timestamp(start)),
            ('time', '<=', pd.Timestamp(end))
        ]
        
        df = pq.read_table(
            self.data_path,
            columns=columns,
            filters=filters
        ).to_pandas()
        
        # Use memory-efficient dtypes
        df = df.astype({
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'volume': 'int32'
        })
        
        return df
```

## Network latency simulation

```python
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class LatencyConfig:
    mean_latency_ms: float = 5.0
    latency_std_ms: float = 2.0
    spike_probability: float = 0.01
    spike_multiplier: float = 10.0
    min_latency_ms: float = 0.5

class LatencySimulator:
    """Simulate realistic network latency for order execution."""
    
    def __init__(self, config: LatencyConfig):
        self.config = config
    
    def get_latency_ms(self) -> float:
        """Generate random latency sample."""
        base = np.random.normal(
            self.config.mean_latency_ms,
            self.config.latency_std_ms
        )
        
        # Random spike simulation
        if np.random.random() < self.config.spike_probability:
            base *= self.config.spike_multiplier
        
        return max(self.config.min_latency_ms, base)
    
    def simulate_order_execution(self, order_time: float, 
                                  price_at_order: float,
                                  price_series: pd.Series) -> tuple:
        """Simulate order execution with latency-adjusted fill price."""
        latency_ms = self.get_latency_ms()
        execution_time = order_time + (latency_ms / 1000)
        
        # Find price at execution time
        execution_idx = price_series.index.searchsorted(execution_time)
        if execution_idx >= len(price_series):
            execution_price = price_series.iloc[-1]
        else:
            execution_price = price_series.iloc[execution_idx]
        
        return execution_price, latency_ms
```

## Technology stack summary

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Language** | Python 3.11+ | FreqTrade compatibility, ecosystem |
| **Backtesting Core** | FreqTrade 2025.x | Best crypto leverage support, active development |
| **Time Series DB** | TimescaleDB | 90%+ compression, continuous aggregates |
| **Strategy DB** | PostgreSQL | JSONB for parameters, strong indexing |
| **Cache/Queue** | Redis | State persistence, Celery broker |
| **Task Queue** | Celery | Distributed backtest execution |
| **Data Format** | Parquet | 10x compression, fast reads |
| **Dashboard** | Streamlit/Plotly | Rapid development, interactive |
| **Containerization** | Docker Compose | Local deployment simplicity |

## Implementation timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Foundation** | Weeks 1-3 | FreqTrade setup, Binance data pipeline, basic strategy execution |
| **Phase 2: Core Features** | Weeks 4-7 | Strategy database, orchestration layer, parallel execution |
| **Phase 3: Exchange Integration** | Weeks 8-9 | Ethereal adapter, unified data normalization |
| **Phase 4: Advanced Features** | Weeks 10-11 | Bankroll management, latency simulation, enhanced reporting |
| **Phase 5: Polish & Testing** | Week 12 | Dashboard, documentation, integration testing |

**Resource requirements:**
- **CPU:** 32+ cores recommended (AMD EPYC/Intel Xeon)
- **RAM:** 64-128 GB for parallel backtesting
- **Storage:** 500 GB NVMe SSD
- **Team:** 1-2 senior Python developers with trading systems experience

## Risk assessment and mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FreqTrade API changes | High | Medium | Pin versions, maintain fork if needed |
| Ethereal API limitations | Medium | Medium | Fallback to Binance-only, monitor API evolution |
| Performance bottlenecks | Medium | Low | Profiling, Rust extensions for hot paths |
| Data quality issues | High | Medium | Validation pipeline, gap detection, alerting |

---

## Appendix A: Key API endpoints reference

### Binance API

| Purpose | Endpoint | Rate Limit |
|---------|----------|------------|
| Historical Klines | `GET /api/v3/klines` | 2 weight |
| Futures Klines | `GET /fapi/v1/klines` | 1-10 weight |
| Funding Rate | `GET /fapi/v1/fundingRate` | 1 weight |
| Mark Price | `GET /fapi/v1/premiumIndex` | 1 weight |
| WebSocket Klines | `wss://stream.binance.com/ws/<symbol>@kline_<interval>` | 1024 streams/connection |
| Bulk Data | `https://data.binance.vision/data/spot/monthly/klines/` | No limit |

### Ethereal API

| Purpose | Endpoint | Notes |
|---------|----------|-------|
| Products | `GET /v1/product` | List all tradeable products |
| Market Price | `GET /v1/product/market-price` | Current bid/ask/oracle |
| Place Order | `POST /v1/order` | Requires EIP-712 signature |
| Order Fills | `GET /v1/order/fill` | Historical fills |
| Archive Data | `https://archive.ethereal.trade` | Historical data access |
| WebSocket | `wss://ws.ethereal.trade/v1/stream` | Socket.IO protocol |

## Appendix B: Performance metrics formulas

```
Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Std Dev
Sortino Ratio = (Portfolio Return - Risk Free Rate) / Downside Deviation  
Calmar Ratio = CAGR / |Maximum Drawdown|
Max Drawdown = (Peak - Trough) / Peak
Profit Factor = Gross Profits / Gross Losses
Win Rate = Winning Trades / Total Trades
Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
Kelly % = Win Rate - [(1 - Win Rate) / Win-Loss Ratio]
```

## Appendix C: Default configuration values

```yaml
# Backtesting defaults
backtesting:
  timeframe: "1m"
  stake_amount: "unlimited"
  max_open_trades: 10
  
# Fee simulation
fees:
  maker: 0.0002      # 0.02%
  taker: 0.0005      # 0.05%
  
# Slippage simulation  
slippage:
  base_bps: 1.0
  volume_impact: 0.1
  
# Leverage settings
leverage:
  max: 20
  default: 1
  margin_mode: "isolated"
  
# Bankroll management
bankroll:
  method: "fixed_fractional"
  risk_per_trade: 0.02
  max_position_pct: 0.10
  max_daily_loss: 0.02
  max_drawdown_pause: 0.15
  
# Latency simulation
latency:
  mean_ms: 5
  std_ms: 2
  spike_probability: 0.01
  spike_multiplier: 10
```