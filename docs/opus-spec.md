# Comprehensive algorithmic trading backtesting framework specification

Building a production-grade backtesting framework for perpetual futures requires integrating exchange APIs, realistic trade simulation, high-performance execution, and sophisticated risk management. This specification provides complete technical requirements for implementing a modular, extensible system supporting Binance and Ethereal exchanges with **20x leverage**, 1-minute to 1-day timeframes, and a web-based analytics dashboard.

The recommended architecture follows a **hybrid approach**: vectorized backtesting for rapid parameter screening (1000x faster) combined with event-driven simulation for final validation with realistic execution modeling. Python with Numba JIT compilation offers the optimal balance of development speed and execution performance.

---

## Exchange API integration specifications

### Binance Perpetual Futures API

Binance provides the most comprehensive perpetual futures API with extensive historical data access. The USDⓈ-M Futures API serves as the primary integration target.

**Base URLs and authentication:**
| Environment | REST API | WebSocket |
|-------------|----------|-----------|
| Production | `https://fapi.binance.com` | `wss://fstream.binance.com` |
| Testnet | `https://demo-fapi.binance.com` | `wss://fstream.binancefuture.com` |

Authentication requires **HMAC SHA256 signatures** with API keys passed via `X-MBX-APIKEY` header. Every signed request must include a `timestamp` parameter (milliseconds) and optional `recvWindow` (default 5000ms, max 60000ms).

**Rate limits to implement:**
- **2,400 request weight per minute** for REST endpoints
- **1,200 orders per minute** (account-based)
- **10 messages per second** per WebSocket connection
- Maximum **1,024 streams** per combined WebSocket connection

**Kline/candlestick data format (12-element array):**
```json
[1499040000000, "0.01634", "0.80000", "0.01575", "0.01577", "148976.11", 
 1499644799999, "2434.19", 308, "1756.87", "28.46", "17928899.62"]
```
Elements: open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore.

**Available intervals:** 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

**Historical data retrieval:** Maximum **1,500 candles per request** via `GET /fapi/v1/klines`. Pagination requires setting `startTime` to `last_candle_close_time + 1`. For bulk historical data, use `data.binance.vision` archive.

**WebSocket streams for real-time data:**
| Stream | Format | Update Speed |
|--------|--------|--------------|
| Kline | `<symbol>@kline_<interval>` | 250ms |
| Mark Price | `<symbol>@markPrice@1s` | 1s |
| Book Ticker | `<symbol>@bookTicker` | Real-time |
| Depth | `<symbol>@depth<levels>@100ms` | 100ms |

**Order types supported:** LIMIT, MARKET, STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET

**Time-in-force options:** GTC (Good Till Cancel), IOC (Immediate or Cancel), FOK (Fill or Kill), GTX (Post-Only), GTD (Good Till Date)

### Ethereal Exchange API

Ethereal operates as a **non-custodial Layer-3 exchange** on Arbitrum with unique authentication requirements.

**Base URLs:**
| Environment | REST API | WebSocket |
|-------------|----------|-----------|
| Mainnet | `https://api.ethereal.trade` | `wss://ws.ethereal.trade` |
| Testnet | `https://api.etherealtest.net` | `wss://ws.etherealtest.net` |

**Authentication:** Uses **EIP-712 cryptographic signatures** instead of API keys. Messages require wallet-native signing with domain configuration:
```json
{"name": "Ethereal", "version": "1", "chainId": 5064014}
```

Nonces use **nanosecond timestamps** (not sequential counters), enabling high-frequency trading without on-chain state tracking.

**Key differences from Binance:**
| Feature | Binance | Ethereal |
|---------|---------|----------|
| Auth method | API keys + HMAC | EIP-712 signatures |
| Custody | Custodial | Non-custodial |
| Settlement | USDT | USDe (yield-bearing) |
| Funding interval | 8 hours | 1 hour |
| Fee structure | 0.02%/0.04% | 0%/0.03% |

**Available products:** BTCUSD (20x max), ETHUSD (20x max), SOLUSD (10x max)

**Python SDK installation:**
```bash
pip install ethereal-sdk
```

**WebSocket streams:** BookDepth (100 levels, 200ms updates), MarketPrice (1s), OrderFill, OrderUpdate, SubaccountLiquidation

---

## Backtesting framework architecture

### Event-driven architecture design

The framework should implement a **message bus pattern** with discrete event types flowing through a central queue.

**Core event types:**
```python
class EventType(Enum):
    MARKET = "MARKET"      # New bar/tick data
    SIGNAL = "SIGNAL"      # Strategy trading signal
    ORDER = "ORDER"        # Order to submit
    FILL = "FILL"          # Order execution
    RISK = "RISK"          # Risk management events
```

**Event flow diagram:**
```
DataHandler → [MARKET] → Strategy → [SIGNAL] → Portfolio → [ORDER] → 
ExecutionHandler → [FILL] → Portfolio (update positions)
```

**Main event loop structure:**
```python
while data_handler.continue_backtest:
    data_handler.update_bars()      # Emit MARKET events
    while events.process_next():    # Process all generated events
        pass
```

### Strategy pattern implementation

Strategies should implement a **protocol/interface** for easy swapping:

```python
from typing import Protocol

class IStrategy(Protocol):
    def on_bar(self, bar_data: dict) -> list[Signal]: ...
    def on_tick(self, tick_data: dict) -> list[Signal]: ...
    def initialize(self, context: BacktestContext) -> None: ...
```

**Strategy registry with decorator:**
```python
@StrategyRegistry.register("moving_average_cross")
class MovingAverageCrossover(Strategy):
    def on_bar(self, bar_data):
        # Strategy logic
        pass
```

### Multi-exchange data normalization

Create an **adapter layer** to normalize exchange-specific formats:

```python
@dataclass
class NormalizedBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str

class BinanceAdapter(ExchangeAdapter):
    def normalize_bar(self, raw: list) -> NormalizedBar:
        return NormalizedBar(
            timestamp=datetime.fromtimestamp(raw[0] / 1000),
            open=float(raw[1]), high=float(raw[2]),
            low=float(raw[3]), close=float(raw[4]),
            volume=float(raw[5])
        )
```

### Modular component separation

Follow the **LEAN framework** modular design:

| Module | Responsibility |
|--------|----------------|
| DataHandler | Data loading, normalization, bar emission |
| Strategy | Signal generation only (no position management) |
| Portfolio | Position sizing, risk limits, target generation |
| ExecutionHandler | Order routing, fill simulation |
| RiskManager | Pre/post-trade risk checks |

---

## Trade simulation mechanics

### Perpetual futures P&L calculation

**Long position unrealized P&L:**
```
Unrealized_PnL = (Mark_Price - Entry_Price) × Position_Size
```

**Short position unrealized P&L:**
```
Unrealized_PnL = (Entry_Price - Mark_Price) × Position_Size
```

**Position averaging formula:**
```
New_Entry = (Old_Position × Old_Entry + New_Position × New_Entry) / Total_Position
```

### Fee calculation model

**Binance fee structure:**
| Fee Type | Maker | Taker |
|----------|-------|-------|
| Default tier | 0.02% | 0.04% |
| VIP 1 | 0.016% | 0.04% |

**Fee calculation (on notional, not margin):**
```python
def calculate_fee(position_notional, is_maker=False):
    rate = 0.0002 if is_maker else 0.0004
    return position_notional * rate
```

### Slippage modeling

Implement **volume-based slippage** with the square-root market impact model:

```python
def calculate_slippage(order_size, avg_volume, volatility, spread):
    participation_rate = order_size / avg_volume
    base_slippage = spread / 2
    market_impact = 0.1 * volatility * math.sqrt(participation_rate)
    return base_slippage + market_impact
```

**Configurable parameters:**
- `base_slippage`: 1 bps minimum
- `volume_impact_factor`: 0.1 coefficient
- `random_factor`: 0-2 bps Gaussian noise

### Funding rate simulation

**Funding payment (every 8 hours for Binance, 1 hour for Ethereal):**
```
Funding_Fee = Position_Notional × Funding_Rate
```

**Payment direction:**
- Positive rate: Longs pay shorts
- Negative rate: Shorts pay longs

**Accrual implementation:**
```python
def calculate_funding_accrual(position_value, funding_rate, hours_held, interval=8):
    num_periods = hours_held // interval
    return position_value * funding_rate * num_periods
```

### Leverage and liquidation mechanics

**Initial margin calculation:**
```
Initial_Margin = Position_Notional / Leverage
```

**Maintenance margin (tiered by position size):**
| Position Size (USDT) | MMR | Maintenance Amount |
|---------------------|-----|-------------------|
| 0 - 50,000 | 0.4% | 0 |
| 50,000 - 250,000 | 0.5% | 50 |
| 250,000 - 1,000,000 | 1.0% | 1,300 |

**Simplified liquidation price (isolated margin):**
```
Long: Liq_Price = Entry × (1 - 1/Leverage + MMR)
Short: Liq_Price = Entry × (1 + 1/Leverage - MMR)
```

**Margin ratio trigger:**
```
Margin_Ratio = Maintenance_Margin / Margin_Balance
Liquidation occurs when Margin_Ratio ≥ 100%
```

### Stop loss and take profit simulation

**Order trigger mechanics:**
```python
def check_stop_trigger(position, mark_price, trigger_type='mark'):
    if position.side == 'long':
        if mark_price <= position.stop_loss:
            return 'stop_triggered'
        if mark_price >= position.take_profit:
            return 'tp_triggered'
```

**Trailing stop implementation:**
```python
class TrailingStop:
    def update(self, current_price, position_side):
        if position_side == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.stop_price = current_price * (1 - self.trail_percent)
        return self.stop_price
```

**OCO (One-Cancels-Other) handling:** When both SL and TP could trigger on same bar, assume SL triggers first (pessimistic approach).

### Network latency simulation

```python
class LatencySimulator:
    def get_latency(self, base_ms=50, jitter_ms=20, spike_prob=0.01):
        latency = base_ms + random.gauss(0, jitter_ms)
        if random.random() < spike_prob:
            latency *= 10  # Occasional spikes
        return max(1, latency)
```

---

## Performance optimization strategies

### Vectorized vs event-driven comparison

| Aspect | Vectorized | Event-Driven |
|--------|-----------|--------------|
| Speed | **10-1000x faster** | Slower, sequential |
| Realism | Lower (fills at bar close) | Higher (models execution) |
| Look-ahead bias risk | High | Low |
| Code reuse (live trading) | Different code needed | Same code |

**Recommended hybrid approach:**
1. **Stage 1 (Vectorized):** Screen thousands of parameter combinations with VectorBT
2. **Stage 2 (Event-driven):** Validate top 5-10 candidates with realistic execution

### Numba JIT optimization

```python
from numba import njit

@njit(parallel=True, cache=True)
def calculate_returns(prices):
    n = len(prices)
    returns = np.empty(n-1, dtype=np.float64)
    for i in range(n-1):
        returns[i] = (prices[i+1] - prices[i]) / prices[i]
    return returns
```

**Typical speedup:** 10-100x over pure Python

### Data storage recommendations

| Scale | Storage | Processing |
|-------|---------|------------|
| <1GB | Parquet files | Pandas + Numba |
| 1-100GB | DuckDB + Parquet | VectorBT |
| >100GB | QuestDB/TimescaleDB | Dask distributed |

**Parquet advantages:**
- Columnar storage (faster column selection)
- Excellent compression (Snappy or Brotli)
- Native S3 support for cloud deployment

**DuckDB for local analytics:**
- Query Parquet files directly without loading
- Processes datasets larger than RAM
- ASOF joins for time-series alignment

### Memory optimization techniques

```python
# Data type optimization (50-80% memory savings)
df['symbol'] = df['symbol'].astype('category')
df['close'] = df['close'].astype('float32')

# Chunked processing for large datasets
for chunk in pd.read_parquet('data.parquet', chunksize=100000):
    process(chunk)
```

### Parallel backtesting with Ray

```python
import ray

@ray.remote
def backtest(params, data_ref):
    data = ray.get(data_ref)  # Zero-copy access
    return run_backtest(data, params)

# Store data once, share across workers
data_ref = ray.put(price_data)
futures = [backtest.remote(p, data_ref) for p in params]
results = ray.get(futures)
```

**Ray vs Dask:** Ray offers ~10% faster execution with zero-copy NumPy array sharing via Plasma object store.

---

## Paper trading implementation

### Architecture differences from backtesting

| Aspect | Backtesting | Paper Trading |
|--------|------------|---------------|
| Data source | Historical files | Live WebSocket streams |
| Time model | Compressed replay | Real-time progression |
| Execution | Instantaneous fills | Simulated latency |
| State | Recreatable from data | Must persist |

### Live data integration pattern

```python
class LiveDataHandler:
    async def connect(self):
        self.ws = await websockets.connect('wss://fstream.binance.com/stream')
        await self.subscribe(['btcusdt@kline_1m', 'btcusdt@bookTicker'])
    
    async def on_message(self, message):
        normalized = self.normalize(json.loads(message))
        self.event_bus.publish(MarketEvent(normalized))
```

**Reconnection strategy with exponential backoff:**
```python
async def reconnect(attempt=0):
    delay = min(1000 * (2 ** attempt), 30000)  # Max 30s
    jitter = random.random() * 1000
    await asyncio.sleep((delay + jitter) / 1000)
```

### Simulated order execution

**Realistic fill simulation components:**
1. **Latency:** 50-200ms base + jitter
2. **Slippage:** Volume-based market impact
3. **Partial fills:** Walk through order book levels
4. **Queue position:** For limit orders, estimate position based on volume ahead

```python
def simulate_market_fill(order, order_book):
    fills = []
    remaining = order.quantity
    for price, qty in order_book.asks:  # For buy order
        filled = min(remaining, qty)
        fills.append((price, filled))
        remaining -= filled
        if remaining <= 0:
            break
    vwap = sum(p * q for p, q in fills) / order.quantity
    return vwap
```

### State persistence

**Critical data to persist:**
- Account balances and margin
- Open positions with entry details
- Pending orders
- Trade history with timestamps

**Checkpoint strategy:**
- Periodic snapshots every 5 minutes
- Event sourcing for complete audit trail
- Recovery: Load snapshot → replay events since snapshot

---

## Statistics and metrics implementation

### Core trading metrics with formulas

**Sharpe Ratio (annualized):**
```python
def sharpe_ratio(returns, rf=0.01, periods=252):
    excess = returns.mean() * periods - rf
    vol = returns.std() * np.sqrt(periods)
    return excess / vol
```

**Sortino Ratio (downside deviation only):**
```python
def sortino_ratio(returns, rf=0.01, periods=252):
    excess = returns.mean() * periods - rf
    downside = returns[returns < 0].std() * np.sqrt(periods)
    return excess / downside
```

**Maximum Drawdown:**
```python
def max_drawdown(equity_curve):
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()
```

**Profit Factor:**
```
Profit_Factor = Gross_Profits / Gross_Losses
```
Interpretation: >1.0 profitable, >1.75 strong, >2.0 excellent

**Expectancy per trade:**
```
E = (Win% × Avg_Win) - (Loss% × Avg_Loss)
```

### Advanced risk metrics

**Value at Risk (95% confidence):**
```python
def var_95(returns):
    return np.percentile(returns, 5)
```

**Conditional VaR (Expected Shortfall):**
```python
def cvar_95(returns):
    var = np.percentile(returns, 5)
    return returns[returns <= var].mean()
```

**Calmar Ratio:**
```
Calmar = CAGR / |Max_Drawdown|
```

### Dashboard visualization recommendations

**Essential charts:**
| Chart | Purpose | Library |
|-------|---------|---------|
| Equity curve | Cumulative performance | Plotly line |
| Drawdown | Risk visualization | Area chart (red fill) |
| Monthly returns heatmap | Seasonal patterns | Seaborn heatmap |
| Trade distribution | Win/loss analysis | Histogram |
| Rolling Sharpe | Consistency | Rolling line chart |

**Dashboard layout structure:**
```
┌─────────────────────────────────────────────────────┐
│  KPIs: Return | Sharpe | MDD | Win Rate | PF       │
├─────────────────────────────────────────────────────┤
│  [Date Range] [Strategy Filter] [Asset Filter]      │
├─────────────────────────────────────────────────────┤
│  Equity Curve          │  Trade Distribution        │
├─────────────────────────────────────────────────────┤
│  Drawdown Chart        │  Monthly Heatmap           │
├─────────────────────────────────────────────────────┤
│  Trade Log Table (sortable, filterable)            │
└─────────────────────────────────────────────────────┘
```

**Framework recommendations:**
- **Rapid prototyping:** Streamlit (Python)
- **Production dashboard:** Plotly Dash or React + Plotly.js
- **Real-time updates:** WebSocket + Lightweight Charts

---

## Portfolio management strategies

### Kelly Criterion implementation

**Basic Kelly formula:**
```
f* = W - (1-W)/R
```
Where W = win rate, R = avg_win/avg_loss

**Fractional Kelly (recommended):**
```python
def kelly_position_size(win_rate, avg_win, avg_loss, fraction=0.5):
    R = avg_win / abs(avg_loss)
    kelly_full = win_rate - ((1 - win_rate) / R)
    if kelly_full <= 0:
        return 0  # No edge
    return kelly_full * fraction  # Half-Kelly recommended
```

### Fixed fractional position sizing

```python
def fixed_fractional_size(equity, risk_percent, entry, stop_loss):
    risk_amount = equity * risk_percent
    trade_risk = abs(entry - stop_loss)
    return risk_amount / trade_risk
```

**Standard risk percentages:** 1-2% per trade for conservative, 2-5% for aggressive

### ATR-based volatility sizing

```python
def atr_position_size(equity, risk_percent, atr, atr_multiple=2):
    risk_amount = equity * risk_percent
    stop_distance = atr * atr_multiple
    return risk_amount / stop_distance
```

### Risk management integration

**Portfolio heat monitoring:**
```python
def portfolio_heat(positions, correlations, individual_risks):
    # Account for correlated positions
    portfolio_variance = weights.T @ cov_matrix @ weights
    return np.sqrt(portfolio_variance)
```

**Drawdown-based position reduction:**
```python
def drawdown_adjustment(base_position, current_dd, max_dd=0.20):
    if current_dd >= max_dd:
        return 0  # Stop trading
    reduction = 1 - (current_dd / max_dd)
    return base_position * reduction
```

**Daily loss limits:**
- Daily limit: 2% of equity
- Weekly limit: 5% of equity
- Monthly limit: 10% of equity

### Leverage management for perpetuals

**Optimal leverage (Kelly-based):**
```
Optimal_Leverage = Expected_Return / Variance
```

**Volatility decay consideration:**
```
Geometric_Return ≈ Arithmetic_Return - (Leverage² × Variance) / 2
```

**Dynamic leverage adjustment:**
```python
def adjust_leverage(current_vol, target_vol=0.15, base_leverage=1.0):
    vol_ratio = target_vol / current_vol
    return max(0.5, min(2.0, base_leverage * vol_ratio))
```

---

## Implementation technology stack

### Recommended Python stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Core framework | Custom event-driven | Backtesting engine |
| Fast computation | Numba, NumPy | Performance-critical paths |
| Data storage | Parquet + DuckDB | Efficient data management |
| Screening | VectorBT | Rapid parameter optimization |
| Visualization | Plotly, Seaborn | Charts and dashboards |
| Dashboard | Streamlit/Dash | Web interface |
| Metrics | empyrical, quantstats | Performance analytics |
| Parallelization | Ray | Distributed computing |

### TypeScript/Node.js considerations

For TypeScript implementation:
- **Worker threads** for CPU-intensive backtests
- **SharedArrayBuffer** for zero-copy data sharing
- **Piscina** for worker pool management
- **WebAssembly** (Rust-compiled) for numerical hotspots

```typescript
import Piscina from 'piscina';

const pool = new Piscina({
  filename: './backtest-worker.js',
  maxThreads: os.cpus().length - 1
});

const results = await Promise.all(
  paramSets.map(params => pool.run(params))
);
```

### Data pipeline architecture

```
Exchange APIs → Data Normalizers → Parquet Storage →
              ↓
DuckDB (queries) → Backtesting Engine → Results DB →
              ↓
Dashboard (Plotly/React) ← WebSocket (live updates)
```

### Testing strategy

1. **Unit tests:** Order matching, position calculations, margin requirements
2. **Integration tests:** WebSocket reconnection, data normalization, exchange adapters
3. **Simulation tests:** Run 10,000+ random trades through paper trading system
4. **Shadow testing:** Compare paper trades vs small real trades on testnet

---

## Key implementation priorities

For building this framework, prioritize in this order:

1. **Data layer:** Parquet storage + exchange adapters for Binance/Ethereal
2. **Core engine:** Event-driven backtesting with strategy interface
3. **Simulation accuracy:** Realistic fees, slippage, funding, liquidation
4. **Performance:** Numba optimization + vectorized screening
5. **Paper trading:** Live WebSocket integration with simulated execution
6. **Dashboard:** Statistics calculation + Plotly/Streamlit visualization
7. **Risk management:** Position sizing + drawdown controls

This specification provides complete technical requirements for a production-grade algorithmic trading backtesting framework supporting perpetual futures with leverage, extensible architecture for spot trading, and comprehensive analytics capabilities.