# Algorithmic Crypto Trading Bot: Complete Specification

**Document Version:** 1.0  
**Date:** February 2, 2026  
**Status:** Draft for Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Requirements Document (PRD)](#2-product-requirements-document-prd)
3. [Software Specification](#3-software-specification)
4. [Architecture Specification](#4-architecture-specification)
5. [Exchange Integration Specifications](#5-exchange-integration-specifications)
6. [Overfitting Prevention Framework](#6-overfitting-prevention-framework)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Risk Assessment & Mitigation](#8-risk-assessment--mitigation)
9. [Technology Recommendations](#9-technology-recommendations)

---

## 1. Executive Summary

### 1.1 Project Vision

Build a production-grade algorithmic trading system for cryptocurrency perpetual futures with a modular, extensible backtesting framework that prioritizes **realistic simulation**, **overfitting prevention**, and **seamless transition from research to live trading**.

### 1.2 Core Objectives

| Objective | Description | Success Criteria |
|-----------|-------------|------------------|
| **Flexible Backtesting** | Support multiple algorithms, timeframes, and data sources | Run 10+ strategies across 5+ timeframes simultaneously |
| **Realistic Simulation** | Model fees, slippage, funding rates, liquidation, latency | <5% deviation between backtest and paper trading results |
| **Overfitting Prevention** | Detect and prevent curve-fitting to historical noise | Pass walk-forward validation with >60% OOS consistency |
| **Production Readiness** | Paper trading and live trading with identical codebase | Zero code changes between backtest and live execution |
| **Performance** | Fast iteration cycles for strategy development | <30 seconds for 1-year backtest on 1-minute data |

### 1.3 Recommended Solution Architecture

Based on comprehensive framework evaluation, the recommended approach is a **hybrid architecture**:

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Rapid Screening** | VectorBT (Python) | 1000x faster parameter optimization via vectorization |
| **Production Engine** | NautilusTrader (Python/Rust) | Identical backtest/live code, institutional-grade execution |
| **Exchange Adapters** | Custom + CCXT | Binance native, Ethereal requires custom EIP-712 adapter |
| **Data Storage** | Parquet + DuckDB | Columnar storage, query without full load |
| **Dashboard** | Streamlit/Plotly Dash | Rapid development, rich visualizations |

### 1.4 Key Constraints

- **Ethereal Integration**: No existing framework supports Ethereal's EIP-712 authentication—requires 2-4 weeks custom development
- **Leverage Risk**: 20x leverage with proper liquidation modeling is critical; miscalculation can cause catastrophic losses
- **Licensing**: Avoid AGPL (Backtesting.py) and GPL (Backtrader) for commercial deployment flexibility

---

## 2. Product Requirements Document (PRD)

### 2.1 Product Overview

#### 2.1.1 Problem Statement

Manual cryptocurrency trading is inefficient, emotionally driven, and cannot capture opportunities across multiple markets simultaneously. Existing backtesting tools either lack realistic simulation (leading to overfit strategies that fail in production) or require extensive custom development for crypto-specific features like funding rates and leverage.

#### 2.1.2 Solution

An integrated algorithmic trading platform that:
1. Enables rapid strategy prototyping with vectorized backtesting
2. Validates strategies with event-driven simulation including realistic market mechanics
3. Deploys identical strategy code to paper and live trading
4. Provides comprehensive analytics to detect overfitting before capital deployment

### 2.2 Functional Requirements

#### 2.2.1 Backtesting Engine (FR-100 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-101 | Multi-algorithm support | P0 | Run 5+ strategies in single backtest session |
| FR-102 | Multi-timeframe analysis | P0 | Combine 1m, 5m, 15m, 1h, 4h, 1d signals in one strategy |
| FR-103 | Multi-asset portfolios | P1 | Backtest portfolio of 10+ trading pairs simultaneously |
| FR-104 | Historical data replay | P0 | Replay any time range with configurable speed |
| FR-105 | Vectorized screening mode | P0 | Test 10,000+ parameter combinations in <10 minutes |
| FR-106 | Event-driven validation mode | P0 | Bar-by-bar simulation with realistic execution |

#### 2.2.2 Trade Simulation (FR-200 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-201 | Fee modeling | P0 | Maker/taker rates configurable per exchange |
| FR-202 | Slippage modeling | P0 | Volume-based market impact + configurable spread |
| FR-203 | Funding rate simulation | P0 | 8-hour (Binance) and 1-hour (Ethereal) intervals |
| FR-204 | Leverage up to 20x | P0 | Accurate margin and liquidation calculations |
| FR-205 | Liquidation simulation | P0 | Position liquidated when margin ratio ≥100% |
| FR-206 | Network latency simulation | P1 | Configurable delay (50-500ms) with jitter |
| FR-207 | Partial fill modeling | P2 | Order book walk-through for large orders |

#### 2.2.3 Order Management (FR-300 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-301 | Market orders | P0 | Execute at next available price + slippage |
| FR-302 | Limit orders | P0 | Queue position simulation for fills |
| FR-303 | Stop-loss orders | P0 | Trigger at mark price, execute at market |
| FR-304 | Take-profit orders | P0 | Trigger at mark price, execute at limit |
| FR-305 | Trailing stop orders | P1 | Dynamic stop adjustment based on price movement |
| FR-306 | OCO (One-Cancels-Other) | P1 | SL+TP bracket orders with automatic cancellation |
| FR-307 | Time-in-force options | P2 | GTC, IOC, FOK, GTX support |

#### 2.2.4 Risk & Bankroll Management (FR-400 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-401 | Fixed fractional sizing | P0 | Risk X% of equity per trade |
| FR-402 | Kelly criterion sizing | P1 | Optimal f calculation with fractional Kelly option |
| FR-403 | ATR-based sizing | P1 | Position size inversely proportional to volatility |
| FR-404 | Maximum position limits | P0 | Configurable per-asset and portfolio-wide limits |
| FR-405 | Daily loss limits | P0 | Auto-halt trading at X% daily drawdown |
| FR-406 | Maximum drawdown controls | P0 | Reduce position size as drawdown increases |
| FR-407 | Correlation-aware exposure | P2 | Adjust sizing for correlated positions |

#### 2.2.5 Analytics & Reporting (FR-500 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-501 | Core metrics | P0 | Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor |
| FR-502 | Equity curve visualization | P0 | Interactive chart with trade markers |
| FR-503 | Drawdown analysis | P0 | Drawdown chart with duration and recovery time |
| FR-504 | Trade distribution | P0 | Histogram of returns, holding periods |
| FR-505 | Monthly returns heatmap | P1 | Seasonal pattern visualization |
| FR-506 | Rolling metrics | P1 | 30/60/90-day rolling Sharpe, win rate |
| FR-507 | Benchmark comparison | P2 | Strategy vs buy-and-hold, vs other strategies |
| FR-508 | Trade-by-trade log | P0 | Exportable CSV with all trade details |

#### 2.2.6 Overfitting Detection (FR-600 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-601 | Walk-forward analysis | P0 | Configurable train/test windows with anchored/rolling |
| FR-602 | Out-of-sample testing | P0 | Automatic holdout set with no data leakage |
| FR-603 | Deflated Sharpe Ratio | P1 | Account for multiple testing bias |
| FR-604 | Parameter sensitivity | P1 | Heatmap of performance across parameter ranges |
| FR-605 | Monte Carlo simulation | P2 | Randomized trade order stress testing |
| FR-606 | Overfitting probability score | P1 | Aggregate metric combining multiple indicators |

#### 2.2.7 Paper Trading (FR-700 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-701 | Live data streaming | P0 | Real-time WebSocket connection to exchanges |
| FR-702 | Simulated execution | P0 | Virtual fills with realistic latency/slippage |
| FR-703 | State persistence | P0 | Resume after restart without position loss |
| FR-704 | Real vs simulated comparison | P1 | Track what would have happened with real orders |
| FR-705 | Multi-week continuous run | P0 | Stable operation for 30+ days |

#### 2.2.8 Exchange Integration (FR-800 Series)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-801 | Binance Futures | P0 | Full USDⓈ-M perpetual support |
| FR-802 | Ethereal DEX | P0 | EIP-712 signed authentication |
| FR-803 | Historical data download | P0 | Automated backfill for any date range |
| FR-804 | Real-time data streaming | P0 | <100ms latency for price updates |
| FR-805 | Order submission | P0 | Authenticated order placement |
| FR-806 | Position/balance queries | P0 | Real-time account state |

### 2.3 Non-Functional Requirements

#### 2.3.1 Performance (NFR-100 Series)

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-101 | Vectorized backtest speed | 4M candles in <1 second |
| NFR-102 | Event-driven backtest speed | 1 year × 1-minute × 1 asset in <30 seconds |
| NFR-103 | Parameter sweep | 10,000 combinations in <10 minutes |
| NFR-104 | Live data latency | <100ms from exchange to strategy |
| NFR-105 | Order submission latency | <200ms from signal to exchange |
| NFR-106 | Memory efficiency | <4GB for typical multi-asset backtest |

#### 2.3.2 Reliability (NFR-200 Series)

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-201 | Uptime (paper/live trading) | 99.9% (excluding exchange downtime) |
| NFR-202 | Data integrity | Zero data corruption or loss |
| NFR-203 | Crash recovery | Resume within 60 seconds with state intact |
| NFR-204 | WebSocket reconnection | Automatic with exponential backoff |

#### 2.3.3 Scalability (NFR-300 Series)

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-301 | Concurrent strategies | 10+ strategies running simultaneously |
| NFR-302 | Historical data storage | 5+ years of 1-minute data for 50+ pairs |
| NFR-303 | Parallel backtesting | Distribute across 8+ CPU cores |

#### 2.3.4 Usability (NFR-400 Series)

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-401 | Strategy development time | New strategy implementable in <2 hours |
| NFR-402 | Dashboard responsiveness | All charts load in <3 seconds |
| NFR-403 | Configuration | YAML/JSON config files, no code changes for parameters |

### 2.4 User Stories

#### 2.4.1 Strategy Developer Persona

**US-001**: As a strategy developer, I want to quickly test a new trading idea across multiple parameter combinations so that I can identify promising configurations without waiting hours.

**Acceptance Criteria:**
- Define strategy in <50 lines of code
- Run 1,000 parameter combinations in <5 minutes
- View performance heatmap of results

**US-002**: As a strategy developer, I want to validate my strategy against realistic market conditions so that I can trust the backtest results before risking capital.

**Acceptance Criteria:**
- Backtest includes fees, slippage, funding rates
- Liquidation triggers match exchange behavior
- Results within 5% of paper trading performance

**US-003**: As a strategy developer, I want to detect if my strategy is overfit so that I avoid deploying strategies that will fail in live trading.

**Acceptance Criteria:**
- Walk-forward analysis with 5 rolling windows
- Out-of-sample Sharpe ratio visible
- Warning when IS/OOS performance diverges >50%

#### 2.4.2 Trader Persona

**US-010**: As a trader, I want to run my strategy in paper trading mode so that I can verify performance with real market data before using real money.

**Acceptance Criteria:**
- Paper trading runs continuously for weeks
- All trades logged with entry/exit timestamps
- Performance dashboard updates in real-time

**US-011**: As a trader, I want to set risk limits so that I never lose more than I can afford.

**Acceptance Criteria:**
- Daily loss limit halts trading automatically
- Position size respects maximum risk per trade
- Portfolio-wide exposure limits enforced

---

## 3. Software Specification

### 3.1 System Components

#### 3.1.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Data      │    │  Strategy   │    │  Execution  │                 │
│  │   Layer     │───▶│   Layer     │───▶│   Layer     │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│        │                  │                  │                          │
│        ▼                  ▼                  ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Event Bus / Message Queue                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│        │                  │                  │                          │
│        ▼                  ▼                  ▼                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Risk      │    │  Analytics  │    │  Portfolio  │                 │
│  │   Manager   │    │   Engine    │    │   Manager   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                        EXTERNAL INTERFACES                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │  Binance    │    │  Ethereal   │    │  Dashboard  │                 │
│  │  Adapter    │    │  Adapter    │    │   (Web UI)  │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Layer Specification

#### 3.2.1 Data Models

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from decimal import Decimal

class Exchange(Enum):
    BINANCE = "binance"
    ETHEREAL = "ethereal"

class Side(Enum):
    LONG = "long"
    SHORT = "short"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass(frozen=True)
class Bar:
    """Normalized OHLCV bar data."""
    symbol: str
    exchange: Exchange
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades: int
    timeframe: str  # "1m", "5m", "1h", etc.

@dataclass
class Position:
    """Current position state."""
    symbol: str
    exchange: Exchange
    side: Side
    size: Decimal
    entry_price: Decimal
    leverage: int
    liquidation_price: Decimal
    unrealized_pnl: Decimal
    margin: Decimal
    opened_at: datetime

@dataclass
class Order:
    """Order specification."""
    id: str
    symbol: str
    exchange: Exchange
    side: Side
    order_type: OrderType
    size: Decimal
    price: Optional[Decimal]  # None for market orders
    stop_price: Optional[Decimal]
    take_profit_price: Optional[Decimal]
    leverage: int
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime]
    fill_price: Optional[Decimal]
    fee: Decimal

@dataclass
class Trade:
    """Completed trade record."""
    id: str
    symbol: str
    exchange: Exchange
    side: Side
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    leverage: int
    entry_time: datetime
    exit_time: datetime
    gross_pnl: Decimal
    fees: Decimal
    funding_paid: Decimal
    net_pnl: Decimal
    return_pct: Decimal

@dataclass
class AccountState:
    """Current account snapshot."""
    exchange: Exchange
    balance: Decimal
    equity: Decimal
    margin_used: Decimal
    margin_available: Decimal
    unrealized_pnl: Decimal
    positions: list[Position]
    timestamp: datetime
```

#### 3.2.2 Data Storage Schema

**Parquet File Structure:**
```
data/
├── binance/
│   ├── BTCUSDT/
│   │   ├── 1m/
│   │   │   ├── 2024.parquet
│   │   │   ├── 2025.parquet
│   │   │   └── 2026.parquet
│   │   ├── 5m/
│   │   ├── 1h/
│   │   └── 1d/
│   └── ETHUSDT/
│       └── ...
├── ethereal/
│   ├── BTCUSD/
│   └── ETHUSD/
└── metadata.json
```

**Parquet Schema:**
```python
schema = pa.schema([
    ('timestamp', pa.timestamp('ms')),
    ('open', pa.decimal128(18, 8)),
    ('high', pa.decimal128(18, 8)),
    ('low', pa.decimal128(18, 8)),
    ('close', pa.decimal128(18, 8)),
    ('volume', pa.decimal128(18, 8)),
    ('quote_volume', pa.decimal128(18, 8)),
    ('trades', pa.int32()),
])
```

### 3.3 Strategy Layer Specification

#### 3.3.1 Strategy Interface

```python
from abc import ABC, abstractmethod
from typing import Protocol

class IStrategy(Protocol):
    """Strategy interface for both vectorized and event-driven modes."""
    
    @property
    def name(self) -> str:
        """Unique strategy identifier."""
        ...
    
    @property
    def required_timeframes(self) -> list[str]:
        """List of timeframes needed (e.g., ['1m', '1h'])."""
        ...
    
    @property
    def required_symbols(self) -> list[str]:
        """List of symbols to subscribe to."""
        ...
    
    @abstractmethod
    def initialize(self, context: 'BacktestContext') -> None:
        """Called once at start. Set up indicators, state."""
        ...
    
    @abstractmethod
    def on_bar(self, bars: dict[str, Bar]) -> list['Signal']:
        """Called on each new bar. Return trading signals."""
        ...
    
    @abstractmethod
    def on_fill(self, fill: 'Fill') -> None:
        """Called when order is filled. Update internal state."""
        ...
    
    def on_position_close(self, trade: Trade) -> None:
        """Optional: called when position is closed."""
        pass


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    side: Side
    signal_type: str  # "entry", "exit", "adjust"
    size: Optional[Decimal]  # None = use position sizer
    price: Optional[Decimal]  # None = market order
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    metadata: dict  # Strategy-specific data
```

#### 3.3.2 Strategy Registry

```python
class StrategyRegistry:
    """Central registry for strategy discovery and instantiation."""
    
    _strategies: dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy class."""
        def decorator(strategy_class):
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: dict) -> IStrategy:
        """Instantiate strategy by name with config."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name](**config)
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return all registered strategy names."""
        return list(cls._strategies.keys())


# Example usage
@StrategyRegistry.register("sma_crossover")
class SMACrossoverStrategy:
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
```

### 3.4 Execution Layer Specification

#### 3.4.1 Order Execution Pipeline

```python
class ExecutionEngine:
    """Handles order routing, fill simulation, and state management."""
    
    def __init__(
        self,
        fee_model: 'IFeeModel',
        slippage_model: 'ISlippageModel',
        latency_model: 'ILatencyModel',
        risk_manager: 'RiskManager'
    ):
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        self.latency_model = latency_model
        self.risk_manager = risk_manager
        self.pending_orders: list[Order] = []
    
    def submit_order(self, order: Order, current_bar: Bar) -> Order:
        """Submit order through risk checks and queue for execution."""
        # Risk checks
        risk_result = self.risk_manager.check_order(order)
        if not risk_result.approved:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = risk_result.reason
            return order
        
        # Add latency
        order.execute_after = current_bar.timestamp + self.latency_model.get_delay()
        order.status = OrderStatus.SUBMITTED
        self.pending_orders.append(order)
        return order
    
    def process_bar(self, bar: Bar) -> list[Fill]:
        """Check pending orders against new bar, generate fills."""
        fills = []
        remaining = []
        
        for order in self.pending_orders:
            if bar.timestamp < order.execute_after:
                remaining.append(order)
                continue
            
            fill = self._try_fill(order, bar)
            if fill:
                fills.append(fill)
            elif order.status == OrderStatus.SUBMITTED:
                remaining.append(order)
        
        self.pending_orders = remaining
        return fills
    
    def _try_fill(self, order: Order, bar: Bar) -> Optional[Fill]:
        """Attempt to fill order against bar data."""
        if order.order_type == OrderType.MARKET:
            # Market order fills at open + slippage
            slippage = self.slippage_model.calculate(order, bar)
            fill_price = bar.open * (1 + slippage if order.side == Side.LONG else 1 - slippage)
            fee = self.fee_model.calculate(order.size * fill_price, is_maker=False)
            return Fill(order=order, price=fill_price, size=order.size, fee=fee)
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order fills if price crosses
            if order.side == Side.LONG and bar.low <= order.price:
                fee = self.fee_model.calculate(order.size * order.price, is_maker=True)
                return Fill(order=order, price=order.price, size=order.size, fee=fee)
            elif order.side == Side.SHORT and bar.high >= order.price:
                fee = self.fee_model.calculate(order.size * order.price, is_maker=True)
                return Fill(order=order, price=order.price, size=order.size, fee=fee)
        
        # ... handle other order types
        return None
```

#### 3.4.2 Fee Model Interface

```python
class IFeeModel(Protocol):
    def calculate(self, notional: Decimal, is_maker: bool) -> Decimal:
        """Calculate fee for given notional value."""
        ...

class BinanceFeeModel:
    """Binance perpetual futures fee structure."""
    
    def __init__(self, maker_rate: Decimal = Decimal("0.0002"), 
                 taker_rate: Decimal = Decimal("0.0004")):
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
    
    def calculate(self, notional: Decimal, is_maker: bool) -> Decimal:
        rate = self.maker_rate if is_maker else self.taker_rate
        return notional * rate

class EtherealFeeModel:
    """Ethereal DEX fee structure."""
    
    def __init__(self, maker_rate: Decimal = Decimal("0"), 
                 taker_rate: Decimal = Decimal("0.0003")):
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
    
    def calculate(self, notional: Decimal, is_maker: bool) -> Decimal:
        rate = self.maker_rate if is_maker else self.taker_rate
        return notional * rate
```

#### 3.4.3 Slippage Model Interface

```python
class ISlippageModel(Protocol):
    def calculate(self, order: Order, bar: Bar) -> Decimal:
        """Calculate slippage as fraction of price (e.g., 0.001 = 0.1%)."""
        ...

class VolumeBasedSlippage:
    """Square-root market impact model."""
    
    def __init__(
        self,
        base_spread_bps: Decimal = Decimal("1"),
        impact_coefficient: Decimal = Decimal("0.1"),
        random_noise_bps: Decimal = Decimal("2")
    ):
        self.base_spread = base_spread_bps / Decimal("10000")
        self.impact_coeff = impact_coefficient
        self.noise_bps = random_noise_bps / Decimal("10000")
    
    def calculate(self, order: Order, bar: Bar) -> Decimal:
        # Participation rate: how much of bar volume we're taking
        participation = order.size / bar.volume if bar.volume > 0 else Decimal("0.01")
        
        # Base spread + market impact + noise
        volatility = (bar.high - bar.low) / bar.close  # Intrabar volatility proxy
        market_impact = self.impact_coeff * volatility * participation.sqrt()
        noise = Decimal(str(random.gauss(0, float(self.noise_bps))))
        
        return self.base_spread / 2 + market_impact + abs(noise)
```

#### 3.4.4 Latency Model Interface

```python
class ILatencyModel(Protocol):
    def get_delay(self) -> timedelta:
        """Return simulated network delay."""
        ...

class RealisticLatencyModel:
    """Network latency with jitter and occasional spikes."""
    
    def __init__(
        self,
        base_ms: int = 50,
        jitter_ms: int = 20,
        spike_probability: float = 0.01,
        spike_multiplier: int = 10
    ):
        self.base_ms = base_ms
        self.jitter_ms = jitter_ms
        self.spike_prob = spike_probability
        self.spike_mult = spike_multiplier
    
    def get_delay(self) -> timedelta:
        latency = self.base_ms + random.gauss(0, self.jitter_ms)
        
        if random.random() < self.spike_prob:
            latency *= self.spike_mult
        
        return timedelta(milliseconds=max(1, latency))
```

### 3.5 Risk Management Specification

#### 3.5.1 Position Sizing Models

```python
class IPositionSizer(Protocol):
    def calculate_size(
        self,
        signal: Signal,
        account: AccountState,
        current_price: Decimal
    ) -> Decimal:
        """Calculate position size for given signal."""
        ...

class FixedFractionalSizer:
    """Risk fixed percentage of equity per trade."""
    
    def __init__(self, risk_percent: Decimal = Decimal("0.02")):
        self.risk_percent = risk_percent
    
    def calculate_size(
        self,
        signal: Signal,
        account: AccountState,
        current_price: Decimal
    ) -> Decimal:
        if not signal.stop_loss:
            raise ValueError("Stop loss required for fixed fractional sizing")
        
        risk_amount = account.equity * self.risk_percent
        stop_distance = abs(current_price - signal.stop_loss)
        stop_percent = stop_distance / current_price
        
        position_size = risk_amount / stop_distance
        return position_size

class KellyCriterionSizer:
    """Kelly criterion with fractional Kelly option."""
    
    def __init__(
        self,
        win_rate: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal,
        kelly_fraction: Decimal = Decimal("0.5")  # Half-Kelly recommended
    ):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction
    
    def calculate_kelly(self) -> Decimal:
        """Calculate optimal Kelly fraction."""
        R = self.avg_win / abs(self.avg_loss)
        kelly = self.win_rate - ((1 - self.win_rate) / R)
        return max(Decimal("0"), kelly * self.kelly_fraction)
    
    def calculate_size(
        self,
        signal: Signal,
        account: AccountState,
        current_price: Decimal
    ) -> Decimal:
        kelly = self.calculate_kelly()
        position_value = account.equity * kelly
        return position_value / current_price

class ATRBasedSizer:
    """Volatility-adjusted position sizing using ATR."""
    
    def __init__(
        self,
        risk_percent: Decimal = Decimal("0.02"),
        atr_multiplier: Decimal = Decimal("2")
    ):
        self.risk_percent = risk_percent
        self.atr_mult = atr_multiplier
    
    def calculate_size(
        self,
        signal: Signal,
        account: AccountState,
        current_price: Decimal,
        atr: Decimal
    ) -> Decimal:
        risk_amount = account.equity * self.risk_percent
        stop_distance = atr * self.atr_mult
        return risk_amount / stop_distance
```

#### 3.5.2 Risk Manager

```python
@dataclass
class RiskCheckResult:
    approved: bool
    reason: Optional[str] = None
    adjusted_size: Optional[Decimal] = None

class RiskManager:
    """Pre-trade risk checks and limits enforcement."""
    
    def __init__(self, config: 'RiskConfig'):
        self.config = config
        self.daily_pnl = Decimal("0")
        self.daily_start_equity = None
    
    def check_order(self, order: Order, account: AccountState) -> RiskCheckResult:
        """Run all risk checks on proposed order."""
        
        # Check 1: Daily loss limit
        if self.daily_start_equity:
            daily_drawdown = (self.daily_start_equity - account.equity) / self.daily_start_equity
            if daily_drawdown >= self.config.max_daily_loss:
                return RiskCheckResult(False, "Daily loss limit reached")
        
        # Check 2: Maximum position size
        notional = order.size * order.price * order.leverage
        if notional > self.config.max_position_notional:
            adjusted = self.config.max_position_notional / (order.price * order.leverage)
            return RiskCheckResult(True, "Size reduced to max", adjusted)
        
        # Check 3: Maximum leverage
        if order.leverage > self.config.max_leverage:
            return RiskCheckResult(False, f"Leverage {order.leverage}x exceeds max {self.config.max_leverage}x")
        
        # Check 4: Available margin
        required_margin = notional / order.leverage
        if required_margin > account.margin_available:
            return RiskCheckResult(False, "Insufficient margin")
        
        # Check 5: Maximum drawdown position reduction
        if account.equity < self.daily_start_equity * (1 - self.config.drawdown_reduction_threshold):
            reduction_factor = self._calculate_drawdown_reduction(account)
            adjusted = order.size * reduction_factor
            return RiskCheckResult(True, "Size reduced due to drawdown", adjusted)
        
        return RiskCheckResult(True)
    
    def _calculate_drawdown_reduction(self, account: AccountState) -> Decimal:
        """Reduce position size proportionally as drawdown increases."""
        current_dd = (self.daily_start_equity - account.equity) / self.daily_start_equity
        max_dd = self.config.max_drawdown
        
        if current_dd >= max_dd:
            return Decimal("0")  # Stop trading
        
        return Decimal("1") - (current_dd / max_dd)

@dataclass
class RiskConfig:
    max_leverage: int = 20
    max_position_notional: Decimal = Decimal("100000")
    max_daily_loss: Decimal = Decimal("0.02")  # 2%
    max_drawdown: Decimal = Decimal("0.20")  # 20%
    drawdown_reduction_threshold: Decimal = Decimal("0.10")  # Start reducing at 10%
```

### 3.6 Leverage and Liquidation Specification

#### 3.6.1 Margin Calculations

```python
class MarginCalculator:
    """Calculate margin requirements and liquidation prices."""
    
    # Binance maintenance margin rate tiers
    MAINTENANCE_MARGIN_TIERS = [
        (Decimal("50000"), Decimal("0.004"), Decimal("0")),
        (Decimal("250000"), Decimal("0.005"), Decimal("50")),
        (Decimal("1000000"), Decimal("0.01"), Decimal("1300")),
        (Decimal("5000000"), Decimal("0.025"), Decimal("16300")),
        (Decimal("20000000"), Decimal("0.05"), Decimal("79300")),
    ]
    
    def calculate_initial_margin(
        self,
        notional: Decimal,
        leverage: int
    ) -> Decimal:
        """Initial margin = notional / leverage."""
        return notional / Decimal(leverage)
    
    def calculate_maintenance_margin(
        self,
        notional: Decimal
    ) -> Decimal:
        """Calculate maintenance margin based on position size tiers."""
        for max_notional, mmr, maintenance_amount in self.MAINTENANCE_MARGIN_TIERS:
            if notional <= max_notional:
                return notional * mmr + maintenance_amount
        # Above highest tier
        return notional * Decimal("0.05") + Decimal("79300")
    
    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        leverage: int,
        side: Side,
        mmr: Decimal = Decimal("0.004")
    ) -> Decimal:
        """Simplified liquidation price for isolated margin."""
        if side == Side.LONG:
            return entry_price * (1 - Decimal("1")/Decimal(leverage) + mmr)
        else:  # SHORT
            return entry_price * (1 + Decimal("1")/Decimal(leverage) - mmr)
    
    def check_liquidation(
        self,
        position: Position,
        mark_price: Decimal
    ) -> bool:
        """Check if position should be liquidated."""
        if position.side == Side.LONG:
            return mark_price <= position.liquidation_price
        else:
            return mark_price >= position.liquidation_price

class FundingRateCalculator:
    """Calculate funding payments for perpetual positions."""
    
    def calculate_funding_payment(
        self,
        position: Position,
        funding_rate: Decimal
    ) -> Decimal:
        """
        Funding payment = position notional × funding rate
        Positive rate: longs pay shorts
        Negative rate: shorts pay longs
        """
        notional = position.size * position.entry_price
        payment = notional * funding_rate
        
        if position.side == Side.LONG:
            return -payment if funding_rate > 0 else payment
        else:
            return payment if funding_rate > 0 else -payment
    
    def calculate_accrued_funding(
        self,
        position: Position,
        funding_rates: list[tuple[datetime, Decimal]],
        interval_hours: int = 8  # Binance: 8h, Ethereal: 1h
    ) -> Decimal:
        """Calculate total funding paid/received over position lifetime."""
        total = Decimal("0")
        notional = position.size * position.entry_price
        
        for timestamp, rate in funding_rates:
            if timestamp >= position.opened_at:
                payment = notional * rate
                if position.side == Side.LONG:
                    total -= payment if rate > 0 else -payment
                else:
                    total += payment if rate > 0 else -payment
        
        return total
```

### 3.7 Analytics Engine Specification

#### 3.7.1 Performance Metrics

```python
import numpy as np
import pandas as pd
from scipy import stats

class PerformanceAnalyzer:
    """Calculate comprehensive trading performance metrics."""
    
    def __init__(self, trades: list[Trade], equity_curve: pd.Series):
        self.trades = trades
        self.equity_curve = equity_curve
        self.returns = equity_curve.pct_change().dropna()
    
    def calculate_all_metrics(self) -> dict:
        """Calculate all performance metrics."""
        return {
            # Return metrics
            'total_return': self.total_return(),
            'cagr': self.cagr(),
            'annual_return': self.annual_return(),
            
            # Risk metrics
            'volatility': self.volatility(),
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_duration': self.max_drawdown_duration(),
            'var_95': self.value_at_risk(0.05),
            'cvar_95': self.conditional_var(0.05),
            
            # Risk-adjusted metrics
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'omega_ratio': self.omega_ratio(),
            
            # Trade metrics
            'total_trades': len(self.trades),
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'expectancy': self.expectancy(),
            'avg_holding_period': self.avg_holding_period(),
            
            # Consistency metrics
            'winning_months': self.winning_months(),
            'best_month': self.best_month(),
            'worst_month': self.worst_month(),
        }
    
    def sharpe_ratio(self, risk_free_rate: float = 0.01, periods: int = 252) -> float:
        """Annualized Sharpe ratio."""
        excess_returns = self.returns.mean() * periods - risk_free_rate
        volatility = self.returns.std() * np.sqrt(periods)
        return excess_returns / volatility if volatility > 0 else 0
    
    def sortino_ratio(self, risk_free_rate: float = 0.01, periods: int = 252) -> float:
        """Sortino ratio using downside deviation."""
        excess_returns = self.returns.mean() * periods - risk_free_rate
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() * np.sqrt(periods)
        return excess_returns / downside_std if downside_std > 0 else 0
    
    def calmar_ratio(self) -> float:
        """CAGR / Max Drawdown."""
        mdd = abs(self.max_drawdown())
        return self.cagr() / mdd if mdd > 0 else 0
    
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown."""
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min()
    
    def max_drawdown_duration(self) -> int:
        """Maximum drawdown duration in days."""
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        
        # Find underwater periods
        underwater = drawdown < 0
        groups = (~underwater).cumsum()
        durations = underwater.groupby(groups).sum()
        
        return int(durations.max()) if len(durations) > 0 else 0
    
    def profit_factor(self) -> float:
        """Gross profits / Gross losses."""
        gross_profits = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        return gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if not self.trades:
            return 0
        winners = sum(1 for t in self.trades if t.net_pnl > 0)
        return winners / len(self.trades)
    
    def expectancy(self) -> float:
        """Expected value per trade."""
        if not self.trades:
            return 0
        return sum(float(t.net_pnl) for t in self.trades) / len(self.trades)
    
    def value_at_risk(self, confidence: float = 0.05) -> float:
        """Value at Risk at given confidence level."""
        return np.percentile(self.returns, confidence * 100)
    
    def conditional_var(self, confidence: float = 0.05) -> float:
        """Expected shortfall (CVaR) at given confidence level."""
        var = self.value_at_risk(confidence)
        return self.returns[self.returns <= var].mean()
```

#### 3.7.2 Overfitting Detection Metrics

```python
class OverfittingDetector:
    """Detect and quantify overfitting risk."""
    
    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        num_trials: int,
        variance_sharpe: float,
        skewness: float = 0,
        kurtosis: float = 3
    ) -> float:
        """
        Deflated Sharpe Ratio accounts for multiple testing.
        Returns probability that observed Sharpe is due to luck.
        """
        # Expected maximum Sharpe under null hypothesis
        expected_max = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/num_trials) + \
                       np.euler_gamma * stats.norm.ppf(1 - 1/(num_trials * np.e))
        
        # Adjustment for non-normality
        sr_adj = observed_sharpe * (1 + (skewness/6) * observed_sharpe - \
                 ((kurtosis - 3)/24) * observed_sharpe**2)
        
        # Probability of false discovery
        prob_false = stats.norm.cdf((expected_max - sr_adj) / np.sqrt(variance_sharpe))
        
        return prob_false
    
    def walk_forward_efficiency(
        self,
        in_sample_sharpe: list[float],
        out_sample_sharpe: list[float]
    ) -> dict:
        """
        Walk-forward analysis efficiency metrics.
        High IS/OOS correlation suggests robust strategy.
        """
        correlation = np.corrcoef(in_sample_sharpe, out_sample_sharpe)[0, 1]
        
        avg_is = np.mean(in_sample_sharpe)
        avg_oos = np.mean(out_sample_sharpe)
        degradation = (avg_is - avg_oos) / avg_is if avg_is > 0 else 0
        
        # Count how many OOS periods are profitable
        oos_positive = sum(1 for s in out_sample_sharpe if s > 0)
        consistency = oos_positive / len(out_sample_sharpe)
        
        return {
            'is_oos_correlation': correlation,
            'performance_degradation': degradation,
            'oos_consistency': consistency,
            'avg_in_sample_sharpe': avg_is,
            'avg_out_sample_sharpe': avg_oos,
            'overfitting_probability': max(0, degradation)  # Higher = more overfit
        }
    
    def parameter_sensitivity_score(
        self,
        param_values: np.ndarray,
        performance_values: np.ndarray
    ) -> float:
        """
        Measure how sensitive performance is to parameter changes.
        Low sensitivity (smooth surface) suggests robust strategy.
        """
        # Calculate gradient of performance surface
        gradient = np.gradient(performance_values)
        
        # Coefficient of variation of gradient
        cv = np.std(gradient) / np.abs(np.mean(gradient)) if np.mean(gradient) != 0 else float('inf')
        
        # High CV = performance is sensitive to parameter changes = overfit risk
        return cv
    
    def calculate_overfitting_score(
        self,
        metrics: dict
    ) -> dict:
        """
        Aggregate overfitting indicators into single score.
        """
        # Warning thresholds
        warnings = []
        
        if metrics.get('sharpe_ratio', 0) > 2.0:
            warnings.append("Sharpe ratio >2.0 is suspicious for crypto")
        
        if metrics.get('win_rate', 0) > 0.75:
            warnings.append("Win rate >75% rarely sustainable")
        
        if metrics.get('profit_factor', 0) > 3.0:
            warnings.append("Profit factor >3.0 unlikely to persist")
        
        if metrics.get('performance_degradation', 0) > 0.5:
            warnings.append("50%+ degradation from IS to OOS")
        
        if metrics.get('oos_consistency', 1) < 0.6:
            warnings.append("Less than 60% OOS periods profitable")
        
        # Overall score (0-1, higher = more likely overfit)
        score = len(warnings) / 5
        
        return {
            'overfitting_score': score,
            'risk_level': 'HIGH' if score > 0.6 else 'MEDIUM' if score > 0.3 else 'LOW',
            'warnings': warnings
        }
```

---

## 4. Architecture Specification

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Web UI     │  │    CLI       │  │   Jupyter    │  │   API        │    │
│  │  (Dashboard) │  │  (Backtest)  │  │  (Research)  │  │  (External)  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼──────────────────┼─────────────────┼─────────────────┼────────────┘
          │                  │                 │                 │
          ▼                  ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Backtest Orchestrator                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ Vectorized │  │   Event    │  │   Paper    │  │    Live    │     │  │
│  │  │   Mode     │  │  Driven    │  │  Trading   │  │  Trading   │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
          │                  │                 │                 │
          ▼                  ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CORE DOMAIN                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐   │
│  │   Strategy    │ │   Portfolio   │ │   Execution   │ │     Risk      │   │
│  │   Engine      │ │   Manager     │ │    Engine     │ │   Manager     │   │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘   │
│          │                 │                 │                 │            │
│          └─────────────────┴─────────────────┴─────────────────┘            │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌───────────────────────────────────┐                   │
│                    │          Event Bus                │                   │
│                    │  (MarketData, Signals, Orders,    │                   │
│                    │   Fills, RiskEvents)              │                   │
│                    └───────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
          │                  │                 │                 │
          ▼                  ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFRASTRUCTURE LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐   │
│  │     Data      │ │   Exchange    │ │   Analytics   │ │    State      │   │
│  │   Storage     │ │   Adapters    │ │   Storage     │ │  Persistence  │   │
│  │  (Parquet)    │ │ (Binance/ETH) │ │  (DuckDB)     │ │  (PostgreSQL) │   │
│  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Event-Driven Architecture

#### 4.2.1 Event Types

```python
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

@dataclass
class Event(ABC):
    """Base event class."""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
class MarketEvent(Event):
    """New market data available."""
    symbol: str
    exchange: Exchange
    bar: Bar
    
class SignalEvent(Event):
    """Strategy generated trading signal."""
    strategy_name: str
    signal: Signal
    
class OrderEvent(Event):
    """Order submitted for execution."""
    order: Order
    
class FillEvent(Event):
    """Order was filled."""
    order: Order
    fill_price: Decimal
    fill_size: Decimal
    fee: Decimal
    
class RiskEvent(Event):
    """Risk limit triggered."""
    event_type: str  # "daily_limit", "max_drawdown", "liquidation"
    message: str
    action: str  # "reduce_position", "close_all", "halt_trading"

class FundingEvent(Event):
    """Funding rate payment."""
    symbol: str
    exchange: Exchange
    rate: Decimal
    payment: Decimal
```

#### 4.2.2 Event Bus Implementation

```python
from collections import defaultdict
from typing import Callable, Type
import asyncio

class EventBus:
    """Central event routing and processing."""
    
    def __init__(self):
        self._handlers: dict[Type[Event], list[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        
    def subscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """Register handler for event type."""
        self._handlers[event_type].append(handler)
        
    def publish(self, event: Event) -> None:
        """Add event to processing queue."""
        self._queue.put_nowait(event)
        
    async def process(self) -> None:
        """Process all events in queue."""
        while not self._queue.empty():
            event = await self._queue.get()
            handlers = self._handlers[type(event)]
            
            for handler in handlers:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logging.error(f"Handler error for {type(event)}: {e}")
                    
    async def run_loop(self) -> None:
        """Main event loop for live/paper trading."""
        while True:
            await self.process()
            await asyncio.sleep(0.001)  # 1ms tick
```

### 4.3 Backtest Orchestration

#### 4.3.1 Hybrid Backtesting Flow

```python
class BacktestOrchestrator:
    """Coordinate vectorized screening and event-driven validation."""
    
    def __init__(
        self,
        data_provider: 'DataProvider',
        config: 'BacktestConfig'
    ):
        self.data_provider = data_provider
        self.config = config
        
    async def run_parameter_sweep(
        self,
        strategy_class: type,
        param_grid: dict[str, list],
        symbols: list[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Stage 1: Vectorized parameter sweep for rapid screening.
        Uses VectorBT for speed.
        """
        import vectorbt as vbt
        
        # Load data
        data = {}
        for symbol in symbols:
            data[symbol] = self.data_provider.get_bars(
                symbol, start_date, end_date
            )
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        results = []
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            # Run vectorized backtest
            strategy = strategy_class(**param_dict)
            signals = strategy.generate_signals_vectorized(data)
            
            # VectorBT portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=signals['entries'],
                exits=signals['exits'],
                fees=self.config.fee_rate,
                sl_stop=param_dict.get('stop_loss'),
                tp_stop=param_dict.get('take_profit')
            )
            
            results.append({
                **param_dict,
                'sharpe': portfolio.sharpe_ratio(),
                'max_dd': portfolio.max_drawdown(),
                'total_return': portfolio.total_return(),
                'win_rate': portfolio.win_rate()
            })
        
        return pd.DataFrame(results)
    
    async def validate_strategy(
        self,
        strategy: IStrategy,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        mode: str = 'event_driven'
    ) -> 'BacktestResult':
        """
        Stage 2: Full event-driven validation with realistic execution.
        Uses NautilusTrader-style engine.
        """
        # Initialize components
        event_bus = EventBus()
        portfolio = PortfolioManager(self.config.initial_capital)
        execution = ExecutionEngine(
            fee_model=self.config.fee_model,
            slippage_model=self.config.slippage_model,
            latency_model=self.config.latency_model,
            risk_manager=RiskManager(self.config.risk_config)
        )
        
        # Wire up event handlers
        event_bus.subscribe(MarketEvent, strategy.on_bar)
        event_bus.subscribe(SignalEvent, portfolio.on_signal)
        event_bus.subscribe(OrderEvent, execution.submit_order)
        event_bus.subscribe(FillEvent, portfolio.on_fill)
        event_bus.subscribe(FillEvent, strategy.on_fill)
        
        # Load and replay data
        data_handler = DataHandler(self.data_provider, symbols, start_date, end_date)
        
        strategy.initialize(BacktestContext(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital
        ))
        
        # Main backtest loop
        while data_handler.has_more_data():
            # Emit market events for all symbols at this timestamp
            bars = data_handler.get_next_bars()
            for symbol, bar in bars.items():
                event_bus.publish(MarketEvent(symbol=symbol, bar=bar))
            
            # Process all events generated
            await event_bus.process()
            
            # Check for funding rate events
            self._check_funding_events(event_bus, portfolio, bars)
            
            # Check for liquidations
            self._check_liquidations(event_bus, portfolio, bars)
        
        # Calculate results
        return BacktestResult(
            trades=portfolio.closed_trades,
            equity_curve=portfolio.equity_curve,
            metrics=PerformanceAnalyzer(
                portfolio.closed_trades,
                portfolio.equity_curve
            ).calculate_all_metrics()
        )
    
    def _check_funding_events(
        self,
        event_bus: EventBus,
        portfolio: PortfolioManager,
        bars: dict[str, Bar]
    ) -> None:
        """Check if funding payment is due."""
        for position in portfolio.positions.values():
            if self._is_funding_time(bars[position.symbol].timestamp, position.exchange):
                funding_rate = self.data_provider.get_funding_rate(
                    position.symbol,
                    bars[position.symbol].timestamp
                )
                payment = FundingRateCalculator().calculate_funding_payment(
                    position, funding_rate
                )
                event_bus.publish(FundingEvent(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    rate=funding_rate,
                    payment=payment
                ))
    
    def _check_liquidations(
        self,
        event_bus: EventBus,
        portfolio: PortfolioManager,
        bars: dict[str, Bar]
    ) -> None:
        """Check positions for liquidation."""
        calculator = MarginCalculator()
        for position in portfolio.positions.values():
            mark_price = bars[position.symbol].close
            if calculator.check_liquidation(position, mark_price):
                event_bus.publish(RiskEvent(
                    event_type="liquidation",
                    message=f"Position {position.symbol} liquidated",
                    action="close_position"
                ))
```

### 4.4 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
└─────────────────────────────────────────────────────────────────────┘

[Exchange APIs] ──┬──▶ [Raw Data Queue] ──▶ [Normalizer] ──▶ [Parquet Storage]
                  │                                                │
[Binance REST]────┤                                                │
[Binance WS]──────┤                                                │
[Ethereal REST]───┤                                                ▼
[Ethereal WS]─────┘                                        [DuckDB Analytics]
                                                                   │
                                                                   ▼
                                                           [Backtest Engine]
                                                                   │
                                                                   ▼
                                                           [Strategy Code]
```

#### 4.4.1 Data Provider Interface

```python
class IDataProvider(Protocol):
    """Data access interface for backtesting and live trading."""
    
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m"
    ) -> pd.DataFrame:
        """Get historical OHLCV bars."""
        ...
    
    def get_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get historical funding rates."""
        ...
    
    async def subscribe_bars(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Bar], None]
    ) -> None:
        """Subscribe to live bar updates."""
        ...
    
    def get_available_symbols(self, exchange: Exchange) -> list[str]:
        """List available trading pairs."""
        ...

class ParquetDataProvider(IDataProvider):
    """Data provider using Parquet files with DuckDB queries."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.conn = duckdb.connect()
        
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m"
    ) -> pd.DataFrame:
        # Query Parquet files directly with DuckDB
        path = self.data_dir / f"{symbol}/{timeframe}/*.parquet"
        
        query = f"""
            SELECT * FROM read_parquet('{path}')
            WHERE timestamp >= '{start.isoformat()}'
              AND timestamp <= '{end.isoformat()}'
            ORDER BY timestamp
        """
        
        return self.conn.execute(query).df()
```

---

## 5. Exchange Integration Specifications

### 5.1 Binance Perpetual Futures

#### 5.1.1 API Configuration

| Setting | Production | Testnet |
|---------|------------|---------|
| REST Base URL | `https://fapi.binance.com` | `https://demo-fapi.binance.com` |
| WebSocket URL | `wss://fstream.binance.com` | `wss://fstream.binancefuture.com` |
| Auth Method | HMAC-SHA256 | HMAC-SHA256 |
| Rate Limit | 2,400 weight/min | Same |

#### 5.1.2 Binance Adapter Implementation

```python
import ccxt
import hmac
import hashlib
import aiohttp
import websockets
import json
from urllib.parse import urlencode

class BinanceAdapter:
    """Binance Perpetual Futures adapter."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (
            "https://demo-fapi.binance.com" if testnet 
            else "https://fapi.binance.com"
        )
        self.ws_url = (
            "wss://fstream.binancefuture.com" if testnet
            else "wss://fstream.binance.com"
        )
        
        # Also initialize CCXT for convenience methods
        self.ccxt = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
            'sandbox': testnet
        })
    
    def _sign_request(self, params: dict) -> dict:
        """Add signature to request parameters."""
        params['timestamp'] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1500
    ) -> list[Bar]:
        """Fetch historical klines."""
        all_bars = []
        current_start = start_time
        
        async with aiohttp.ClientSession() as session:
            while current_start < end_time:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_time,
                    'limit': limit
                }
                
                async with session.get(
                    f"{self.base_url}/fapi/v1/klines",
                    params=params
                ) as response:
                    data = await response.json()
                    
                    if not data:
                        break
                    
                    for k in data:
                        all_bars.append(Bar(
                            symbol=symbol,
                            exchange=Exchange.BINANCE,
                            timestamp=datetime.fromtimestamp(k[0] / 1000),
                            open=Decimal(k[1]),
                            high=Decimal(k[2]),
                            low=Decimal(k[3]),
                            close=Decimal(k[4]),
                            volume=Decimal(k[5]),
                            quote_volume=Decimal(k[7]),
                            trades=int(k[8]),
                            timeframe=interval
                        ))
                    
                    # Move to next batch
                    current_start = data[-1][6] + 1  # close_time + 1
                    
                    # Rate limit
                    await asyncio.sleep(0.1)
        
        return all_bars
    
    async def subscribe_klines(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Bar], None]
    ) -> None:
        """Subscribe to live kline updates via WebSocket."""
        stream = f"{symbol.lower()}@kline_{interval}"
        uri = f"{self.ws_url}/ws/{stream}"
        
        async with websockets.connect(uri) as ws:
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                if data['e'] == 'kline':
                    k = data['k']
                    if k['x']:  # Kline closed
                        bar = Bar(
                            symbol=symbol,
                            exchange=Exchange.BINANCE,
                            timestamp=datetime.fromtimestamp(k['t'] / 1000),
                            open=Decimal(k['o']),
                            high=Decimal(k['h']),
                            low=Decimal(k['l']),
                            close=Decimal(k['c']),
                            volume=Decimal(k['v']),
                            quote_volume=Decimal(k['q']),
                            trades=int(k['n']),
                            timeframe=interval
                        )
                        callback(bar)
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        leverage: int = 1
    ) -> dict:
        """Place order on Binance Futures."""
        # Set leverage first
        await self._set_leverage(symbol, leverage)
        
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }
        
        if price:
            params['price'] = str(price)
        if stop_price:
            params['stopPrice'] = str(stop_price)
        
        params = self._sign_request(params)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/fapi/v1/order",
                params=params,
                headers={'X-MBX-APIKEY': self.api_key}
            ) as response:
                return await response.json()
```

### 5.2 Ethereal DEX Integration

#### 5.2.1 API Configuration

| Setting | Mainnet | Testnet |
|---------|---------|---------|
| REST Base URL | `https://api.ethereal.trade` | `https://api.etherealtest.net` |
| WebSocket URL | `wss://ws.ethereal.trade` | `wss://ws.etherealtest.net` |
| Auth Method | EIP-712 Signatures | EIP-712 Signatures |
| Chain ID | 5064014 | Testnet Chain ID |
| Funding Interval | 1 hour | 1 hour |

#### 5.2.2 EIP-712 Signature Implementation

```python
from eth_account import Account
from eth_account.messages import encode_typed_data
import time

class EtherealSigner:
    """EIP-712 message signing for Ethereal authentication."""
    
    DOMAIN = {
        "name": "Ethereal",
        "version": "1",
        "chainId": 5064014,
        "verifyingContract": "0xB3cDC82035C495c484C9fF11eD5f3Ff6d342e3cc"
    }
    
    def __init__(self, private_key: str):
        self.account = Account.from_key(private_key)
        self.address = self.account.address
    
    def sign_order(
        self,
        symbol: str,
        side: str,
        size: Decimal,
        price: Decimal,
        order_type: str
    ) -> dict:
        """Sign order with EIP-712."""
        # Nonce is nanosecond timestamp
        nonce = int(time.time() * 1e9)
        
        message = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"}
                ],
                "Order": [
                    {"name": "symbol", "type": "string"},
                    {"name": "side", "type": "string"},
                    {"name": "size", "type": "uint256"},
                    {"name": "price", "type": "uint256"},
                    {"name": "orderType", "type": "string"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "account", "type": "address"}
                ]
            },
            "primaryType": "Order",
            "domain": self.DOMAIN,
            "message": {
                "symbol": symbol,
                "side": side,
                "size": int(size * 10**18),
                "price": int(price * 10**18),
                "orderType": order_type,
                "nonce": nonce,
                "account": self.address
            }
        }
        
        encoded = encode_typed_data(full_message=message)
        signed = self.account.sign_message(encoded)
        
        return {
            "order": message["message"],
            "signature": signed.signature.hex()
        }

class EtherealAdapter:
    """Ethereal DEX adapter with EIP-712 authentication."""
    
    def __init__(
        self,
        private_key: str,
        testnet: bool = False
    ):
        self.signer = EtherealSigner(private_key)
        self.base_url = (
            "https://api.etherealtest.net" if testnet
            else "https://api.ethereal.trade"
        )
        self.ws_url = (
            "wss://ws.etherealtest.net" if testnet
            else "wss://ws.ethereal.trade"
        )
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int
    ) -> list[Bar]:
        """Fetch historical klines from Ethereal."""
        async with aiohttp.ClientSession() as session:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time
            }
            
            async with session.get(
                f"{self.base_url}/v1/klines",
                params=params
            ) as response:
                data = await response.json()
                
                return [
                    Bar(
                        symbol=symbol,
                        exchange=Exchange.ETHEREAL,
                        timestamp=datetime.fromtimestamp(k['timestamp'] / 1000),
                        open=Decimal(str(k['open'])),
                        high=Decimal(str(k['high'])),
                        low=Decimal(str(k['low'])),
                        close=Decimal(str(k['close'])),
                        volume=Decimal(str(k['volume'])),
                        quote_volume=Decimal(str(k.get('quoteVolume', 0))),
                        trades=k.get('trades', 0),
                        timeframe=interval
                    )
                    for k in data
                ]
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: Decimal,
        price: Optional[Decimal] = None,
        leverage: int = 1
    ) -> dict:
        """Place signed order on Ethereal."""
        # Sign the order
        signed = self.signer.sign_order(
            symbol=symbol,
            side=side,
            size=size,
            price=price or Decimal("0"),
            order_type=order_type
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/order",
                json=signed,
                headers={'Content-Type': 'application/json'}
            ) as response:
                return await response.json()
    
    async def subscribe_book(
        self,
        symbol: str,
        callback: Callable[[dict], None]
    ) -> None:
        """Subscribe to order book updates."""
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps({
                "method": "subscribe",
                "params": {"channel": "book", "symbol": symbol}
            }))
            
            while True:
                message = await ws.recv()
                data = json.loads(message)
                callback(data)
```

---

## 6. Overfitting Prevention Framework

### 6.1 Walk-Forward Analysis

```python
class WalkForwardAnalyzer:
    """Rolling window walk-forward optimization and validation."""
    
    def __init__(
        self,
        train_window: int,  # Days
        test_window: int,   # Days
        step_size: int,     # Days between windows
        anchored: bool = False  # Expanding or rolling window
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.anchored = anchored
    
    async def run(
        self,
        strategy_class: type,
        param_grid: dict,
        data: pd.DataFrame,
        orchestrator: BacktestOrchestrator
    ) -> 'WalkForwardResult':
        """Run walk-forward analysis."""
        results = []
        
        # Generate windows
        windows = self._generate_windows(data)
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # In-sample optimization
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            
            # Find best parameters on training set
            sweep_results = await orchestrator.run_parameter_sweep(
                strategy_class=strategy_class,
                param_grid=param_grid,
                data=train_data
            )
            
            best_params = sweep_results.loc[sweep_results['sharpe'].idxmax()].to_dict()
            is_sharpe = best_params['sharpe']
            
            # Out-of-sample validation
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            strategy = strategy_class(**{
                k: v for k, v in best_params.items() 
                if k in param_grid
            })
            
            oos_result = await orchestrator.validate_strategy(
                strategy=strategy,
                data=test_data
            )
            
            results.append({
                'window': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params,
                'in_sample_sharpe': is_sharpe,
                'out_sample_sharpe': oos_result.metrics['sharpe_ratio'],
                'out_sample_return': oos_result.metrics['total_return'],
                'out_sample_max_dd': oos_result.metrics['max_drawdown']
            })
        
        return WalkForwardResult(
            windows=results,
            analysis=OverfittingDetector().walk_forward_efficiency(
                [r['in_sample_sharpe'] for r in results],
                [r['out_sample_sharpe'] for r in results]
            )
        )
    
    def _generate_windows(
        self,
        data: pd.DataFrame
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate train/test window pairs."""
        windows = []
        start = data.index.min()
        end = data.index.max()
        
        if self.anchored:
            # Anchored: training window expands
            train_start = start
            current = start + timedelta(days=self.train_window)
            
            while current + timedelta(days=self.test_window) <= end:
                windows.append((
                    train_start,
                    current,
                    current,
                    current + timedelta(days=self.test_window)
                ))
                current += timedelta(days=self.step_size)
        else:
            # Rolling: fixed-size windows
            current = start
            
            while current + timedelta(days=self.train_window + self.test_window) <= end:
                train_end = current + timedelta(days=self.train_window)
                test_end = train_end + timedelta(days=self.test_window)
                
                windows.append((current, train_end, train_end, test_end))
                current += timedelta(days=self.step_size)
        
        return windows

@dataclass
class WalkForwardResult:
    windows: list[dict]
    analysis: dict
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Walk-Forward Analysis Summary
=============================
Total Windows: {len(self.windows)}
Avg In-Sample Sharpe: {self.analysis['avg_in_sample_sharpe']:.2f}
Avg Out-Sample Sharpe: {self.analysis['avg_out_sample_sharpe']:.2f}
IS/OOS Correlation: {self.analysis['is_oos_correlation']:.2f}
Performance Degradation: {self.analysis['performance_degradation']:.1%}
OOS Consistency: {self.analysis['oos_consistency']:.1%}
Overfitting Probability: {self.analysis['overfitting_probability']:.1%}
        """
```

### 6.2 Cross-Validation Methods

```python
class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.
    Includes embargo period to prevent leakage.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01  # 1% embargo between train/test
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        data: pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test splits."""
        n_samples = len(data)
        embargo_size = int(n_samples * self.embargo_pct)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # Training indices: before test (with embargo) + after test
            train_before = np.arange(0, max(0, test_start - embargo_size))
            train_after = np.arange(min(n_samples, test_end + embargo_size), n_samples)
            train_idx = np.concatenate([train_before, train_after])
            
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    Tests all combinations of N-1 groups as training, 1 as test.
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        data: pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate all combinatorial splits."""
        n_samples = len(data)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Split data into groups
        groups = np.array_split(np.arange(n_samples), self.n_splits)
        
        # Generate all combinations of N-1 training groups
        from itertools import combinations
        
        for test_groups in combinations(range(self.n_splits), 1):
            train_groups = [i for i in range(self.n_splits) if i not in test_groups]
            
            train_idx = []
            test_idx = []
            
            for group_idx in train_groups:
                group = groups[group_idx]
                # Apply embargo: remove samples near test boundaries
                for test_group_idx in test_groups:
                    test_group = groups[test_group_idx]
                    test_min, test_max = test_group.min(), test_group.max()
                    
                    # Remove samples within embargo of test set
                    group = group[
                        (group < test_min - embargo_size) | 
                        (group > test_max + embargo_size)
                    ]
                
                train_idx.extend(group)
            
            for group_idx in test_groups:
                test_idx.extend(groups[group_idx])
            
            yield np.array(train_idx), np.array(test_idx)
```

### 6.3 Overfitting Warning Signs

| Indicator | Warning Threshold | Action |
|-----------|-------------------|--------|
| Sharpe Ratio | > 2.0 | Investigate with walk-forward |
| Win Rate | > 70% | Check for look-ahead bias |
| Profit Factor | > 3.0 | Verify with OOS testing |
| # Parameters | > 5 | Simplify strategy |
| IS vs OOS Sharpe | Degrades > 50% | Strategy likely overfit |
| OOS Consistency | < 60% windows profitable | Unstable strategy |
| Parameter Sensitivity | High gradient CV | Fragile to market changes |

---

## 7. Implementation Roadmap

### 7.1 Phase Overview

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **Phase 1** | Weeks 1-3 | Foundation | Data pipeline, core models |
| **Phase 2** | Weeks 4-6 | Backtesting | Vectorized + event-driven engines |
| **Phase 3** | Weeks 7-9 | Exchange Integration | Binance + Ethereal adapters |
| **Phase 4** | Weeks 10-11 | Risk & Analytics | Position sizing, metrics, dashboard |
| **Phase 5** | Weeks 12-13 | Paper Trading | Live data integration, state persistence |
| **Phase 6** | Weeks 14-15 | Overfitting Prevention | Walk-forward, cross-validation |
| **Phase 7** | Weeks 16-17 | Production Hardening | Testing, monitoring, deployment |

### 7.2 Detailed Phase Breakdown

#### Phase 1: Foundation (Weeks 1-3)

**Week 1: Project Setup**
- Initialize Python project with poetry/uv
- Set up type checking (mypy), linting (ruff), testing (pytest)
- Define core data models (Bar, Order, Position, Trade)
- Implement Parquet storage layer

**Week 2: Data Acquisition**
- Build Binance historical data downloader
- Implement data normalization pipeline
- Set up DuckDB for analytics queries
- Create data validation tests

**Week 3: Core Abstractions**
- Implement Strategy interface and registry
- Build Event and EventBus classes
- Create configuration management (YAML/Pydantic)
- Write unit tests for core components

#### Phase 2: Backtesting Engine (Weeks 4-6)

**Week 4: Vectorized Mode**
- Integrate VectorBT for parameter sweeps
- Build strategy-to-VectorBT signal adapter
- Implement parallel parameter optimization
- Create performance heatmap generator

**Week 5: Event-Driven Mode**
- Build ExecutionEngine with fee/slippage models
- Implement PortfolioManager for position tracking
- Create DataHandler for bar replay
- Wire up event flow between components

**Week 6: Trade Simulation**
- Implement leverage and margin calculations
- Add liquidation detection
- Build funding rate simulation
- Create stop-loss/take-profit handling

#### Phase 3: Exchange Integration (Weeks 7-9)

**Week 7: Binance Adapter**
- Implement REST API client with rate limiting
- Build WebSocket connection manager
- Create order placement and management
- Test on Binance testnet

**Week 8: Ethereal Adapter**
- Implement EIP-712 signing
- Build REST and WebSocket clients
- Handle Ethereal-specific order types
- Test on Ethereal testnet

**Week 9: Integration Testing**
- End-to-end tests with both exchanges
- Historical data backfill for both
- Verify data normalization consistency
- Benchmark API latency

#### Phase 4: Risk & Analytics (Weeks 10-11)

**Week 10: Risk Management**
- Implement position sizing strategies (Fixed, Kelly, ATR)
- Build RiskManager with pre-trade checks
- Add daily/weekly/monthly loss limits
- Create drawdown-based position reduction

**Week 11: Analytics Dashboard**
- Implement PerformanceAnalyzer with all metrics
- Build Streamlit dashboard layout
- Create interactive equity curve and drawdown charts
- Add trade log table with filters

#### Phase 5: Paper Trading (Weeks 12-13)

**Week 12: Live Data Integration**
- Connect WebSocket streams to backtest engine
- Implement simulated order execution with latency
- Build state persistence layer (PostgreSQL)
- Handle reconnection and recovery

**Week 13: Paper Trading Mode**
- Create unified backtest/paper/live mode switching
- Implement trade logging and audit trail
- Add real-time dashboard updates
- Test multi-week continuous operation

#### Phase 6: Overfitting Prevention (Weeks 14-15)

**Week 14: Walk-Forward Analysis**
- Implement WalkForwardAnalyzer
- Build rolling and anchored window modes
- Create IS/OOS comparison reports
- Add overfitting score calculation

**Week 15: Cross-Validation & Alerts**
- Implement PurgedKFold and CPCV
- Add Deflated Sharpe Ratio calculation
- Build parameter sensitivity analysis
- Create overfitting warning system

#### Phase 7: Production Hardening (Weeks 16-17)

**Week 16: Testing & Monitoring**
- Comprehensive integration test suite
- Load testing for concurrent strategies
- Set up logging and alerting (Prometheus/Grafana)
- Security audit for API key handling

**Week 17: Deployment**
- Docker containerization
- CI/CD pipeline setup
- Documentation and runbooks
- Final validation on testnet

### 7.3 MVP Milestones

| Milestone | Week | Description |
|-----------|------|-------------|
| **M1: Data Ready** | Week 3 | Historical data for BTC, ETH from Binance stored in Parquet |
| **M2: Backtest Working** | Week 6 | Can run SMA crossover strategy with fees/slippage |
| **M3: Exchanges Connected** | Week 9 | Live data streaming from both Binance and Ethereal |
| **M4: Dashboard Live** | Week 11 | Interactive performance analytics visible |
| **M5: Paper Trading** | Week 13 | 24/7 paper trading with state persistence |
| **M6: Overfitting Detection** | Week 15 | Walk-forward analysis with warnings |
| **M7: Production Ready** | Week 17 | Full system deployed and monitored |

---

## 8. Risk Assessment & Mitigation

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Ethereal API instability** | Medium | High | Abstract behind interface; fallback to Binance-only |
| **Slippage model inaccuracy** | High | High | Conservative 1-2% default; validate with paper trading |
| **Liquidation logic bugs** | Medium | Critical | Extensive unit tests; isolated margin only initially |
| **Data quality issues** | Medium | High | Data validation pipeline; multiple source verification |
| **WebSocket disconnections** | High | Medium | Exponential backoff reconnect; state checkpointing |
| **Memory limits for large backtests** | Medium | Medium | Chunked processing; DuckDB for large queries |

### 8.2 Financial Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Overfit strategy deployment** | High | Critical | Mandatory walk-forward analysis; staged capital allocation |
| **Leverage liquidation** | Medium | Critical | Max 10x initial; isolated margin; conservative sizing |
| **Exchange API changes** | Medium | High | Version pinning; monitoring for deprecations |
| **Flash crash** | Low | Critical | Circuit breakers; max daily loss limits |

### 8.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Server downtime** | Low | High | Cloud deployment with auto-restart; state recovery |
| **API key compromise** | Low | Critical | Encrypted storage; IP whitelisting; no withdrawal permissions |
| **Regulatory changes** | Low | Medium | Modular exchange adapters; jurisdiction monitoring |

---

## 9. Technology Recommendations

### 9.1 Final Stack Recommendation

| Component | Technology | Alternative |
|-----------|------------|-------------|
| **Language** | Python 3.11+ | TypeScript (for dashboard only) |
| **Rapid Backtesting** | VectorBT | Custom NumPy + Numba |
| **Event-Driven Engine** | Custom (NautilusTrader patterns) | NautilusTrader direct |
| **Data Storage** | Parquet + DuckDB | TimescaleDB |
| **Dashboard** | Streamlit | Plotly Dash |
| **Exchange Connectivity** | CCXT + Custom Ethereal | Direct API only |
| **State Persistence** | PostgreSQL | SQLite (development) |
| **Monitoring** | Prometheus + Grafana | ELK Stack |
| **Deployment** | Docker Compose | Kubernetes |

### 9.2 Development Environment

```bash
# Recommended project structure
trading-bot/
├── pyproject.toml
├── src/
│   ├── core/           # Data models, events, interfaces
│   ├── data/           # Data providers, storage
│   ├── strategies/     # Strategy implementations
│   ├── execution/      # Order execution, simulation
│   ├── risk/           # Position sizing, risk management
│   ├── analytics/      # Performance metrics, overfitting
│   ├── exchanges/      # Binance, Ethereal adapters
│   └── dashboard/      # Streamlit app
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/               # Parquet files (gitignored)
├── config/             # YAML configurations
└── scripts/            # Data download, deployment
```

### 9.3 Dependencies

```toml
# pyproject.toml (key dependencies)
[project]
dependencies = [
    # Core
    "numpy>=1.24",
    "pandas>=2.0",
    "polars>=0.20",        # Fast alternative to pandas
    
    # Backtesting
    "vectorbt>=0.25",
    "numba>=0.58",
    
    # Data
    "pyarrow>=14.0",
    "duckdb>=0.9",
    
    # Exchange
    "ccxt>=4.0",
    "websockets>=12.0",
    "aiohttp>=3.9",
    "eth-account>=0.10",   # For Ethereal EIP-712
    
    # Analytics
    "quantstats>=0.0.62",
    "empyrical>=0.5.5",
    "scipy>=1.11",
    
    # Dashboard
    "streamlit>=1.29",
    "plotly>=5.18",
    
    # Utilities
    "pydantic>=2.5",
    "loguru>=0.7",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "mypy>=1.7",
    "ruff>=0.1",
]
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ATR** | Average True Range - volatility indicator |
| **Calmar Ratio** | CAGR divided by maximum drawdown |
| **CPCV** | Combinatorial Purged Cross-Validation |
| **Drawdown** | Peak-to-trough decline in equity |
| **EIP-712** | Ethereum typed structured data signing standard |
| **Funding Rate** | Periodic payment between longs and shorts in perpetuals |
| **Kelly Criterion** | Optimal bet sizing formula |
| **Liquidation** | Forced position closure due to insufficient margin |
| **Mark Price** | Fair price used for liquidation (prevents manipulation) |
| **MMR** | Maintenance Margin Rate |
| **OOS** | Out-of-Sample (test data not seen during optimization) |
| **Perpetual** | Futures contract with no expiration date |
| **Sharpe Ratio** | Risk-adjusted return metric |
| **Slippage** | Difference between expected and actual execution price |
| **Sortino Ratio** | Sharpe variant using downside deviation only |
| **VaR** | Value at Risk - potential loss at confidence level |
| **Walk-Forward** | Rolling optimization/validation methodology |

---

## Appendix B: References

1. De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Chan, E. P. (2021). *Quantitative Trading*. Wiley.
3. Binance Futures API Documentation: https://binance-docs.github.io/apidocs/futures/en/
4. Ethereal Trading API: https://docs.ethereal.trade/developer-guides/trading-api
5. VectorBT Documentation: https://vectorbt.dev/
6. NautilusTrader Documentation: https://nautilustrader.io/

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Feb 2, 2026 | Claude | Initial specification |

