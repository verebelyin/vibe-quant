# Extending Vibe-Quant: New Algorithms & Indicators Research

> **Date**: 2026-02-28
> **Purpose**: Comprehensive research on extending the vibe-quant system with new algorithms, indicators, and techniques to improve discovery and backtesting results.

---

## Table of Contents

1. [Current System Inventory](#1-current-system-inventory)
2. [New Indicators to Implement](#2-new-indicators-to-implement)
   - 2.1 [Trend & Momentum](#21-trend--momentum-indicators)
   - 2.2 [Volatility & Regime Detection](#22-volatility--regime-detection)
   - 2.3 [Volume & Microstructure](#23-volume--microstructure-indicators)
   - 2.4 [Information-Theoretic](#24-information-theoretic-indicators)
   - 2.5 [Statistical & Fractal](#25-statistical--fractal-indicators)
3. [Porting TradingView/Pine Script Indicators](#3-porting-tradingviewpine-script-indicators)
4. [Crypto-Specific Signals](#4-crypto-specific-signals)
5. [Machine Learning Integration](#5-machine-learning-integration)
6. [Advanced Strategy Archetypes](#6-advanced-strategy-archetypes)
7. [Discovery Pipeline Improvements](#7-discovery-pipeline-improvements)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [References & Sources](#9-references--sources)

---

## 1. Current System Inventory

### Currently Supported Indicators (20 total)

| Category | Indicators | NT Native | pandas-ta-classic |
|----------|-----------|-----------|-------------------|
| **Trend** (6) | SMA, EMA, WMA, DEMA, TEMA, Ichimoku | 5 | 6 |
| **Momentum** (6) | RSI, MACD, STOCH, CCI, WILLR, ROC | 5 | 6 |
| **Volatility** (4) | ATR, BBANDS, KC, DONCHIAN | 4 | 4 |
| **Volume** (4) | OBV, VWAP, MFI, VOLSMA | 3 | 4 |

### Discovery Genome Pool (Currently Active)

Only indicators with absolute-scale thresholds are usable in the genetic algorithm:
- **RSI** (period 5-50, threshold 20-80)
- **MACD** (fast 8-21, slow 21-50, signal 5-13, threshold -0.01 to 0.01)
- **ATR** (period 5-30, threshold 0.001-0.05)
- **STOCH** (k 5-21, d 3-9, threshold 20-80)
- **CCI** (threshold -100 to 100)
- **WILLR** (threshold -80 to -20)
- **ROC** (threshold -5 to 5)

**Excluded** from genome: EMA, SMA, WMA, DEMA, TEMA (price-relative, threshold=0 produces no trades), BBANDS, KC (require indicator-vs-indicator comparison not yet in genome).

### Key Limitations

1. **No indicator-vs-indicator genome support** — Can't evolve EMA crossovers or BBANDS breakouts
2. **No composite/derived indicators** — No SuperTrend, Squeeze, or custom combinations
3. **No market regime detection** — Strategies are static, no adaptive behavior
4. **No volume-based signals in genome** — OBV, MFI, VWAP excluded from discovery
5. **No microstructure data** — No order flow, funding rate, or on-chain metrics
6. **No statistical indicators** — No Hurst exponent, entropy, or GARCH-based volatility
7. **Limited fitness function** — Sharpe/DD/PF/Return only; no stability or regime metrics

---

## 2. New Indicators to Implement

### 2.1 Trend & Momentum Indicators

#### SuperTrend
- **What**: ATR-based trend-following indicator that plots a trailing stop above/below price
- **Why**: Simple, clean signals with built-in volatility adaptation. One of the most popular TradingView indicators. Excellent for trend-following strategies in crypto where trends are strong.
- **Signal**: Binary (bullish/bearish) — perfect for genome threshold comparison
- **Implementation**: Available in `pandas-ta-classic` as `df.ta.supertrend()`. Parameters: `period` (7-21), `multiplier` (1.0-4.0)
- **Genome fit**: Excellent — output is +1/-1, threshold at 0
- **Priority**: **HIGH**
- **Source**: [pandas-ta-classic docs](https://xgboosted.github.io/pandas-ta-classic/indicators.html)

#### Aroon Oscillator
- **What**: Measures trend strength and direction based on time since highest high / lowest low
- **Why**: Unique time-based perspective on trend strength. Built into NautilusTrader (Rust).
- **Signal**: -100 to +100 range, ideal for threshold-based genome
- **Implementation**: NautilusTrader native `AroonOscillator`. Parameter: `period` (10-50)
- **Genome fit**: Excellent — bounded range, absolute scale
- **Priority**: **HIGH**

#### Chande Momentum Oscillator (CMO)
- **What**: Unsmoothed momentum oscillator using up/down sum ratio
- **Why**: More responsive than RSI for short-term momentum detection. Bounded -100 to +100.
- **Signal**: -100 to +100, overbought/oversold interpretation
- **Implementation**: Available in `pandas-ta-classic` as `df.ta.cmo()`
- **Genome fit**: Excellent — bounded absolute range
- **Priority**: MEDIUM

#### Klinger Volume Oscillator (KVO)
- **What**: Volume-based oscillator that measures money flow direction
- **Why**: Combines volume and price to detect accumulation/distribution. Better than OBV for divergence detection.
- **Signal**: Oscillates around zero — crossovers and divergences
- **Implementation**: Available in `pandas-ta-classic` as `df.ta.kvo()`. Matches TradingView's implementation.
- **Genome fit**: Good — zero-cross threshold works
- **Priority**: MEDIUM
- **Source**: [pandas-ta-classic docs](https://xgboosted.github.io/pandas-ta-classic/indicators.html)

#### Parabolic SAR
- **What**: Trailing stop-and-reverse indicator (dots above/below price)
- **Why**: Generates clear entry/exit signals. Popular in trend-following systems.
- **Signal**: Binary direction change (price above/below SAR)
- **Implementation**: `pandas-ta-classic` `df.ta.psar()`. Parameters: `af0` (0.01-0.03), `af` (0.01-0.03), `max_af` (0.1-0.3)
- **Genome fit**: Moderate — needs price-relative comparison (SAR vs close)
- **Priority**: MEDIUM

---

### 2.2 Volatility & Regime Detection

#### GARCH/EGARCH Volatility Forecast
- **What**: Econometric model that forecasts conditional volatility using past returns and past variance
- **Why**: Crypto markets exhibit extreme volatility clustering. EGARCH captures the asymmetric leverage effect where negative shocks generate more volatility than positive ones. Half-life analysis shows crypto volatility shocks take 2-4 weeks to dissipate — invaluable for position sizing.
- **Signal**: Forecasted volatility (continuous), can be compared to realized volatility for regime signals
- **Implementation**: Python `arch` library (`arch_model`). EGARCH(1,1) is recommended as best-performing for crypto. Compute on rolling window (e.g., 252 bars).
- **Genome fit**: Can threshold on volatility ratio (forecast/realized) — values >1.5 = expanding vol regime
- **Integration pattern**: Custom NautilusTrader indicator wrapping `arch` library
- **Priority**: **HIGH** — Fundamental for regime-adaptive strategies
- **References**:
  - [GARCH volatility clustering across asset classes (2015-2026)](https://jonathankinlay.com/2026/02/garch-volatility-clustering-asset-classes/)
  - [GARCH vs EGARCH vs GJR-GARCH for Bitcoin](https://www.ijltemas.in/submission/index.php/online/article/view/3574)

#### Realized Volatility (RV) Ratio
- **What**: Ratio of short-term realized volatility to long-term realized volatility
- **Why**: Simple proxy for volatility regime detection. When RV_short >> RV_long, market is in a high-volatility regime.
- **Signal**: Ratio (continuous), >1 = expanding vol, <1 = contracting vol
- **Implementation**: Pure calculation — `std(returns, short_window) / std(returns, long_window)`. Parameters: `short_window` (5-20), `long_window` (50-200)
- **Genome fit**: Excellent — absolute ratio, threshold around 1.0
- **Priority**: **HIGH** — Simple to implement, powerful signal

#### Squeeze Momentum (John Carter's TTM Squeeze)
- **What**: Detects when Bollinger Bands contract inside Keltner Channels (squeeze), then fires when momentum breaks out
- **Why**: Extremely popular on TradingView. Identifies consolidation periods before explosive moves — crypto's bread and butter.
- **Signal**: Binary squeeze state + momentum histogram direction
- **Implementation**: Available in `pandas-ta-classic` as `df.ta.squeeze()` (also has Lazybear variant with `lazybear=True`)
- **Genome fit**: Good — squeeze on/off (binary) + momentum value (continuous)
- **Priority**: **HIGH**
- **Source**: [pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic)

#### Average Directional Index (ADX)
- **What**: Measures trend strength regardless of direction (0-100 scale)
- **Why**: Already proven in research — MACD+ADX combination significantly outperformed individual indicators in crypto. ADX >25 = trending, <20 = ranging.
- **Signal**: 0-100 bounded range — perfect for threshold
- **Implementation**: NautilusTrader native (`AverageDirectionIndex`). Parameters: `period` (7-30)
- **Genome fit**: Excellent — bounded absolute scale
- **Priority**: **HIGH** — Strong research backing, already in NautilusTrader
- **Reference**: [Technical Analysis Meets ML: Bitcoin Evidence (arXiv 2025)](https://arxiv.org/html/2511.00665v1)

---

### 2.3 Volume & Microstructure Indicators

#### VPIN (Volume-Synchronized Probability of Informed Trading)
- **What**: Measures order flow toxicity — the probability that informed traders are present in the market. Updated in volume-time (after each volume bucket), not clock-time.
- **Why**: Academic research shows VPIN significantly predicts future price jumps in Bitcoin. Cornell/Easley et al. found that microstructure-driven predictability (VPIN + Roll measure) enables trend-following exploitation. One of the most important crypto microstructure indicators.
- **Signal**: 0-1 bounded probability — perfect for thresholds. High VPIN (>0.7) = toxic flow, prepare for volatility.
- **Implementation**: Custom implementation required. Key steps:
  1. Classify volume into buy/sell using Bulk Volume Classification (BVC) — NOT tick rule
  2. Aggregate into volume buckets (e.g., 1/50th of daily volume per bucket)
  3. Compute VPIN over rolling window of N buckets: `VPIN = sum(|V_buy - V_sell|) / (N * V_bucket)`
- **Genome fit**: Excellent — bounded 0-1, absolute scale
- **Priority**: **HIGH** — Strong academic backing for crypto
- **References**:
  - [VPIN & Bitcoin price jumps (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
  - [Cornell/Easley: Microstructure in Crypto Markets](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)
  - [From PIN to VPIN (QuantResearch.org)](https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf)

#### Order Book Imbalance (OBI)
- **What**: Ratio of bid volume to ask volume at the top N levels of the order book
- **Why**: Leading indicator of short-term price direction. Research shows order book features dominate (81.3% of selected features in ML models).
- **Signal**: -1 to +1 (normalized) — buy pressure vs sell pressure
- **Implementation**: Requires L2 order book data. `OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)` at top N levels.
- **Genome fit**: Excellent — bounded range
- **Priority**: MEDIUM (requires order book data infrastructure)
- **Reference**: [hftbacktest: Market Making with OBI](https://hftbacktest.readthedocs.io/en/latest/tutorials/Market%20Making%20with%20Alpha%20-%20Order%20Book%20Imbalance.html)

#### Cumulative Volume Delta (CVD)
- **What**: Running total of (buy volume - sell volume), showing aggressive buying vs selling pressure
- **Why**: Reveals divergences between price and actual buying/selling pressure. When price rises but CVD falls, it signals weak buying.
- **Signal**: Trend of the delta — can apply SMA/EMA and use crossover
- **Implementation**: Requires trade-level data with aggressor side. `CVD += buy_vol - sell_vol` per bar.
- **Genome fit**: Moderate — needs derived signal (e.g., CVD rate of change)
- **Priority**: MEDIUM

#### Taker Buy/Sell Ratio
- **What**: Ratio of taker buy volume to total taker volume
- **Why**: Direct measure of market sentiment. Available from Binance API as a pre-computed metric.
- **Signal**: 0-1 bounded, >0.5 = net buying, <0.5 = net selling
- **Implementation**: Fetch from exchange API or compute from trade-level data
- **Genome fit**: Excellent — bounded, absolute scale
- **Priority**: MEDIUM

---

### 2.4 Information-Theoretic Indicators

#### Shannon Entropy of Returns
- **What**: Measures the randomness/predictability of price movements over a rolling window. Converts price changes to binary patterns and computes information content.
- **Why**: Low entropy = more predictable market = better edge for trading strategies. High entropy = random/noise, don't trade. Research shows entropy peaks correlate with trend reversals in BTC/USDT.
- **Signal**: Continuous (0 to log(n)). Low values (<0.7) indicate edge; high values (~1.0) indicate randomness.
- **Implementation**: Custom indicator:
  1. Convert returns to binary (up/down) sequences
  2. Count frequency of each n-bit pattern (e.g., 3-bit: UUU, UUD, UDU, ...)
  3. Compute Shannon entropy: `H = -sum(p * log2(p))`
  4. Normalize: `H_norm = H / log2(n_patterns)`
- **Genome fit**: Excellent — bounded 0-1 when normalized, absolute scale
- **Priority**: **HIGH** — Unique signal, powerful for regime detection
- **References**:
  - [Shannon Entropy & Market Randomness (Robot Wealth)](https://robotwealth.com/shannon-entropy/)
  - [TradingView Shannon Entropy Indicator](https://www.tradingview.com/script/90gGxKtX-Shannon-Entropy-V2/)
  - [Entropy for Trend Detection in BTC/USDT (Superalgos)](https://medium.com/superalgos/entropy-as-a-calculation-basis-for-market-trend-highlighting-advanced-trend-indicator-9569111e3b0a)

#### Transfer Entropy
- **What**: Measures directional information flow between two time series (e.g., BTC → ETH)
- **Why**: Captures non-linear, non-symmetric statistical relationships. Reveals which assets lead and which follow — useful for pairs/cross-asset strategies.
- **Signal**: Continuous, higher = more information transfer
- **Implementation**: Complex — requires `pyinform` or custom implementation with binning
- **Genome fit**: Poor (requires multi-asset context)
- **Priority**: LOW (future enhancement for cross-asset strategies)
- **Reference**: [The Flow of Information in Trading: An Entropy Approach (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7597144/)

#### Approximate Entropy (ApEn)
- **What**: Measures the regularity/complexity of a time series. Lower ApEn = more regular/predictable.
- **Why**: Can detect when markets transition from chaotic to ordered states. More robust than Shannon entropy for noisy financial data.
- **Signal**: Continuous, lower = more predictable
- **Implementation**: Available in various Python libraries (`antropy`, `nolds`). Parameters: `m` (embedding dimension, 2), `r` (tolerance, 0.2 * std)
- **Genome fit**: Good — continuous, can threshold
- **Priority**: MEDIUM

---

### 2.5 Statistical & Fractal Indicators

#### Hurst Exponent (Moving Hurst)
- **What**: Measures long-range dependence. H > 0.5 = trending/persistent, H < 0.5 = mean-reverting, H = 0.5 = random walk.
- **Why**: Fundamental tool for strategy selection: tells you whether to use momentum or mean-reversion. Bitcoin at high frequency (10-second intervals) consistently shows H > 0.7 (strong trending). Academic research proved the "Moving Hurst" indicator outperforms MACD. Crypto pairs with H < 0.5 in spreads revert to mean significantly faster — validated in pairs trading backtests.
- **Signal**: 0-1 bounded — perfect threshold at 0.5 (trend vs mean-revert boundary)
- **Implementation**: Custom indicator using Rescaled Range (R/S) analysis or DFA (Detrended Fluctuation Analysis). Rolling window of 100-500 bars.
  ```python
  def hurst_rs(series, min_window=10, max_window=None):
      """Rescaled Range method for Hurst exponent."""
      # Divide series into subseries of varying lengths
      # For each length: compute R/S statistic
      # Fit log(R/S) vs log(n) — slope = H
  ```
- **Genome fit**: Excellent — bounded 0-1, absolute scale, clear interpretation
- **Priority**: **HIGH** — Uniquely valuable for crypto, strong academic support
- **References**:
  - [Hurst Exponent in Crypto Pairs Trading (MDPI)](https://www.mdpi.com/2227-7390/12/18/2911)
  - [Detecting Trends and Mean Reversion (Macrosynergy)](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/)
  - [Hurst Exponent and Trading Signals (SciTePress)](https://www.scitepress.org/papers/2018/66670/66670.pdf)

#### Fractal Dimension
- **What**: Related to Hurst exponent (D = 2 - H). Measures the roughness/complexity of a price series.
- **Why**: D close to 1 = smooth trend, D close to 2 = choppy/noise. Surges in Hurst (drops in D) can flag breakout transitions.
- **Signal**: 1-2 bounded — threshold around 1.5
- **Implementation**: Derived from Hurst or computed independently via box-counting method
- **Genome fit**: Excellent — bounded, absolute scale
- **Priority**: MEDIUM (can derive from Hurst)

#### Z-Score of Spread (for Pairs/Mean-Reversion)
- **What**: Standardized deviation of a spread from its rolling mean
- **Why**: Core signal for pairs trading and mean-reversion strategies. Z > 2 = overextended short, Z < -2 = overextended long. Academic research (Springer 2025) confirms cointegration-based z-score trading outperforms buy-and-hold in crypto.
- **Signal**: Unbounded but typically -3 to +3
- **Implementation**: `z = (spread - mean(spread, window)) / std(spread, window)`. Parameters: `window` (20-100)
- **Genome fit**: Good — threshold-based (±1.5 to ±2.5)
- **Priority**: MEDIUM (requires multi-asset architecture extension)
- **References**:
  - [Copula-based Trading of Cointegrated Crypto Pairs (Springer 2025)](https://link.springer.com/article/10.1186/s40854-024-00702-7)
  - [Stat Arb Models 2025 Deep Dive](https://coincryptorank.com/blog/stat-arb-models-deep-dive)

---

## 3. Porting TradingView/Pine Script Indicators

### Why Port from TradingView?

TradingView hosts 150,000+ community scripts, many with proven backtesting results. The Pine Script ecosystem represents the largest collection of crowd-sourced trading indicators. Porting the best ones gives access to strategies validated by thousands of traders.

### Conversion Approach

1. **Automated parsing**: Use [pyine](https://pypi.org/project/pyine/) for simple Pine Script → Python conversion
2. **AI-assisted**: Use LLMs (Claude, GPT) for complex indicator conversion with validation
3. **Manual porting**: For critical indicators, hand-convert with numerical validation against TradingView charts
4. **Validation**: Always compare Python output vs Pine Script output on identical historical data

**Key technical difference**: Pine Script uses reverse-indexed arrays (latest = index 0) while Python uses standard indexing (latest = index -1).

### Top TradingView Indicators Worth Porting

| Indicator | TradingView Name | Category | Genome Fit | Effort |
|-----------|-----------------|----------|------------|--------|
| **SuperTrend** | Built-in | Trend | Excellent | Low (in pandas-ta) |
| **Squeeze Momentum** | TTM_SQUEEZE | Volatility | Good | Low (in pandas-ta) |
| **Hurst Momentum Oscillator** | AlphaNatt | Fractal | Excellent | Medium |
| **Advanced Fractal & Hurst (AFHI)** | PuzzlerTrades | Fractal | Excellent | Medium |
| **Shannon Entropy V2** | kocurekc | Information | Excellent | Medium |
| **LuxAlgo Volume Profile** | LuxAlgo | Volume | Poor | High |
| **ICT Killzones** | Community | Time | N/A | Medium |
| **Market Cipher** | Community | Composite | Good | High |
| **Ichimoku Cloud** | Built-in | Trend | Moderate | Low (in pandas-ta) |
| **Linear Regression Channel** | Built-in | Trend | Moderate | Medium |

### Integration Pattern for Custom Indicators

```python
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar

class CustomIndicator(Indicator):
    """Template for porting TradingView indicators to NautilusTrader."""

    def __init__(self, period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self._values = []
        self.value = 0.0

    def handle_bar(self, bar: Bar) -> None:
        self._values.append(float(bar.close))
        if len(self._values) > self.period:
            self._values.pop(0)
        if len(self._values) >= self.period:
            self.value = self._compute()
            self._set_initialized(True)

    def _compute(self) -> float:
        # Implement indicator logic here
        raise NotImplementedError

    def reset(self) -> None:
        self._values.clear()
        self.value = 0.0
        self._set_initialized(False)
```

### References
- [pyine: Pine Script to Python converter](https://pypi.org/project/pyine/)
- [Pine Script to Python Guide (Pineify 2026)](https://pineify.app/resources/blog/converting-pine-script-to-python-a-comprehensive-guide)
- [awesome-pinescript GitHub collection](https://github.com/pAulseperformance/awesome-pinescript)
- [TradingView Community Scripts](https://www.tradingview.com/scripts/)

---

## 4. Crypto-Specific Signals

### 4.1 Funding Rate Signals

**Background**: Perpetual futures use funding rates to anchor price to spot. Funding rates are paid every 8 hours (Binance, most exchanges). Positive rate = longs pay shorts (bullish sentiment); negative = shorts pay longs (bearish).

**Research findings**:
- Funding rate arbitrage consistently offers superior risk-adjusted returns vs HODL ([ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2096720925000818))
- Non-correlated with HODL returns — valuable for diversification
- CEX dominates price discovery with 61% higher integration than DEX ([MDPI 2026](https://www.mdpi.com/2227-7390/14/2/346))
- Only 40% of top arbitrage opportunities generate positive returns after costs ([BIS Working Paper](https://www.bis.org/publ/work1087.pdf))

**Implementation signals**:

| Signal | Formula | Threshold | Usage |
|--------|---------|-----------|-------|
| **Funding Rate** | Direct from exchange | ±0.01% (neutral) | Sentiment indicator |
| **Funding Rate Z-Score** | `(FR - mean) / std` | ±2.0 | Extreme sentiment |
| **Cumulative Funding** | `sum(FR, 24h)` | ±0.1% | Sustained bias |
| **Funding Divergence** | `FR_exchange_A - FR_exchange_B` | ±0.02% | Cross-exchange arb |
| **Basis** | `(perp_price - spot_price) / spot` | ±0.5% | Carry signal |

**Priority**: **HIGH** — unique to perps, strong alpha signal

### 4.2 Open Interest Signals

- **OI Change Rate**: `(OI_t - OI_{t-1}) / OI_{t-1}` — surge = new positions, decline = liquidations
- **OI / Volume Ratio**: High ratio = crowded trade, low = healthy flow
- **OI-weighted Funding**: `funding_rate * OI` — dollar-value of funding exposure

**Priority**: **HIGH** — available from most exchange APIs

### 4.3 Liquidation Cascade Detection

- Monitor large liquidation events (>$1M) as contrarian signals
- Liquidation clusters often mark local extremes
- Available via exchange WebSocket feeds (Binance, Bybit)

**Priority**: MEDIUM

### 4.4 On-Chain Metrics (Longer-Term Signals)

These are primarily useful for longer timeframes (daily/weekly) and position sizing, not intraday signals.

| Metric | What It Measures | Signal |
|--------|-----------------|--------|
| **MVRV Z-Score** | Market cap vs realized cap | >7 = overheated, <0 = undervalued |
| **SOPR** | Spending output profit ratio | <1 = selling at loss (capitulation) |
| **NUPL** | Net unrealized profit/loss | >0.75 = euphoria, <0 = capitulation |
| **Exchange Flows** | Net BTC in/out of exchanges | Net inflow = selling pressure |
| **STH-SOPR** | Short-term holder spending | Reset to 1 = healthy correction |

**Data sources**: [Glassnode](https://glassnode.com), [CryptoQuant](https://cryptoquant.com), [Coin Metrics](https://coinmetrics.io)

**Priority**: LOW for intraday strategies, MEDIUM for position sizing / risk filters

**References**:
- [MVRV Z-Score (Bitcoin Magazine Pro)](https://www.bitcoinmagazinepro.com/charts/mvrv-zscore/)
- [On-Chain Metrics for Price Prediction (Nansen)](https://www.nansen.ai/post/onchain-metrics-key-indicators-for-cryptocurrency-price-prediction)

---

## 5. Machine Learning Integration

### 5.1 Feature Engineering for ML Models

Research consistently shows that **feature engineering matters more than model complexity** for crypto trading. Key feature categories ranked by importance:

1. **Order book features** (81.3% of selected features in ML models — [Springer 2025](https://link.springer.com/article/10.1007/s44163-025-00519-y))
2. **Technical indicator features** (RSI, MACD, BBands position, momentum)
3. **Statistical features** (returns, log returns, rolling std, z-scores)
4. **Volume features** (volume change, VWAP deviation, volume profile)
5. **Sentiment features** (funding rate, social media NLP — lower priority)

**Best practice**: Compute 20-40 well-chosen features rather than hundreds. Use XGBoost feature importance or SHAP values to prune.

### 5.2 Model Architectures for Crypto

| Model | Best For | Pros | Cons | Crypto Evidence |
|-------|----------|------|------|-----------------|
| **XGBoost / LightGBM** | Next-bar direction | Fast, interpretable, handles tabular data well | No sequential memory | Baseline model, consistently competitive |
| **LSTM** | Multi-step forecasting | Captures temporal dependencies, 65% cumulative return on BTC | Overfits on small datasets, slow to train | [arXiv 2025](https://arxiv.org/html/2511.00665v1) |
| **Bidirectional LSTM** | Volatility forecasting | Outperformed GARCH for 7-day RV forecast | Requires more data | [GitHub](https://github.com/chibui191/bitcoin_volatility_forecasting) |
| **PPO / A2C (RL)** | Portfolio allocation | Adapts to regime changes, end-to-end optimization | Sample-inefficient, unstable training | [MDPI 2025](https://www.mdpi.com/2076-3417/15/17/9400) |
| **DQN (RL)** | Strategy selection | Can switch between strategies (RSI, SMA, BBANDS, VWAP) | Doesn't generalize well to unseen regimes | [Tandfonline 2025](https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2594873) |

### 5.3 Integration Architecture with NautilusTrader

```
┌─────────────────────────────────────────────────┐
│                NautilusTrader Strategy            │
│                                                   │
│  on_bar(bar):                                     │
│    features = compute_features(bar, indicators)   │
│    signal = ml_model.predict(features)            │
│    if signal > threshold:                         │
│        submit_order(...)                          │
│                                                   │
│  Indicators:  [RSI, MACD, ATR, Hurst, Entropy]   │
│  ML Model:    [XGBoost / LSTM / RL Agent]         │
│  Features:    [indicator values + derived stats]  │
└─────────────────────────────────────────────────┘
```

**Key design decisions**:
1. **Offline training, online inference**: Train models on historical data, deploy as read-only inference in NautilusTrader strategy callbacks
2. **Feature computation in indicators**: Use NautilusTrader indicator system for feature computation (leverages Rust performance)
3. **Model persistence**: Serialize trained models (joblib/ONNX), load on strategy initialization
4. **Retraining pipeline**: Periodic retraining with walk-forward validation to prevent model decay

### 5.4 Reinforcement Learning Integration

While no turnkey NautilusTrader + Gym integration exists, the building blocks are available:

- **[FinRL](https://github.com/AI4Finance-Foundation/FinRL)**: Open-source DRL library with crypto environments, Stable-Baselines3 integration
- **[gym-trading-env](https://gym-trading-env.readthedocs.io/)**: Gymnasium environment for RL trading agents
- **Architecture**: Train RL agent in Gym environment → export policy → deploy as NautilusTrader strategy

**Recommended approach**: Start with **DQN for strategy selection** (choose between RSI, MACD, Bollinger strategies based on regime) rather than end-to-end RL for order generation. This is more robust and interpretable.

**Reference**: [FinRL Contests 2024-2025 (Wiley)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/aie2.12004)

---

## 6. Advanced Strategy Archetypes

### 6.1 Regime-Adaptive Strategies

**Problem**: Current strategies are static — they apply the same logic in trending and ranging markets. Research shows crypto alternates between distinct regimes (trending, mean-reverting, chaotic).

**Solution**: Use regime detection to switch between strategy modes:

```yaml
# Proposed DSL extension
name: adaptive_regime_strategy
regime_detector:
  type: hurst_entropy
  hurst_period: 100
  entropy_period: 50
  trending_threshold: 0.6    # H > 0.6 = trending
  ranging_threshold: 0.45    # H < 0.45 = mean-reverting

strategies:
  trending_mode:
    indicators: [ema_fast, ema_slow, adx]
    entry: "ema_fast crosses_above ema_slow"
    filter: "adx > 25"
  ranging_mode:
    indicators: [rsi, bbands]
    entry: "rsi < 30"
    filter: "close < bbands.lower"
```

### 6.2 Funding Rate Carry Strategy

**Concept**: Systematically harvest funding rate payments by taking the opposite side of crowded trades.

```
When funding_rate > 0.05%:
  → Go SHORT perp (receive funding)
  → Hedge with LONG spot (delta-neutral)
  Expected return: funding_rate * leverage - costs

When funding_rate < -0.05%:
  → Go LONG perp (receive funding)
  → Hedge with SHORT spot (if available)
```

**Research backing**: [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2096720925000818) — consistently superior risk-adjusted returns vs HODL with non-correlated returns.

**Caveat**: [BIS Working Paper](https://www.bis.org/publ/work1087.pdf) warns about liquidation risk from managing legs separately.

### 6.3 Statistical Arbitrage / Pairs Trading

**Concept**: Find cointegrated crypto pairs, trade the mean-reverting spread.

**Implementation steps**:
1. Test all crypto pair combinations for cointegration (Johansen test)
2. Estimate half-life using Ornstein-Uhlenbeck calibration
3. Compute z-score of spread
4. Enter when |z| > 2, exit when |z| < 0.5
5. Monitor cointegration stability — exit if it breaks

**Research**: [Dynamic Cointegration Pairs Trading in Crypto (arXiv)](https://arxiv.org/pdf/2109.10662) — outperforms buy-and-hold on Bitmex with reasonably low max drawdown.

**Enhancement**: Use Hurst exponent on spread — pairs with H < 0.5 revert faster ([MDPI 2024](https://www.mdpi.com/2227-7390/12/18/2911)).

### 6.4 Avellaneda-Stoikov Market Making

**Concept**: Optimal bid/ask quoting with inventory management. The model calculates a "reservation price" that adjusts based on inventory position.

**Key formula**:
```
reservation_price = mid_price - q * γ * σ² * (T - t)
spread = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
```
Where: q = inventory, γ = risk aversion, σ = volatility, k = order arrival intensity

**Available implementations**:
- [Hummingbot](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/) — full open-source implementation with perpetual exchange connectors
- [hftbacktest](https://hftbacktest.readthedocs.io/) — GLFT model variant with Binance Futures backtesting
- [GitHub implementations](https://github.com/fedecaccia/avellaneda-stoikov) — reference Python code

**Priority**: LOW for discovery pipeline (requires tick-level data), but HIGH value for live trading

### 6.5 Multi-Timeframe Confirmation with ML Filter

**Concept**: Combine traditional technical signals with ML confidence filtering.

```
Signal Generation (Traditional):
  4h: EMA trend direction
  1h: MACD momentum confirmation
  15m: RSI entry trigger

ML Filter (XGBoost):
  Features: [hurst, entropy, vol_ratio, funding_rate, oi_change]
  Output: probability of signal success
  Filter: only take signals with P(success) > 0.6
```

This hybrid approach addresses the key finding that **simple models often outperform complex ones** in crypto — the traditional signals generate candidates, ML filters for quality.

---

## 7. Discovery Pipeline Improvements

### 7.1 Expand the Genome Pool

**Current limitation**: Only 7 indicators in the genome pool (RSI, MACD, ATR, STOCH, CCI, WILLR, ROC). Adding more expands the strategy search space exponentially.

**Recommended additions** (in priority order):

| Indicator | Genome Parameters | Why |
|-----------|------------------|-----|
| **ADX** | period (7-30), threshold (15-40) | Trend strength filter, strong research backing |
| **SuperTrend** | period (7-21), multiplier (1-4), threshold (0) | Clean binary signal, very popular |
| **Shannon Entropy** | period (20-100), threshold (0.3-0.9) | Regime filter — avoid trading in random markets |
| **Hurst Exponent** | period (50-200), threshold (0.3-0.7) | Strategy mode selector (trend vs mean-revert) |
| **Squeeze** | bbands_period (10-30), kc_period (10-30), threshold (0) | Breakout detection |
| **CMO** | period (5-30), threshold (-50 to 50) | Bounded momentum, more responsive than RSI |
| **Aroon Oscillator** | period (10-50), threshold (-50 to 50) | Time-based trend strength |
| **Realized Vol Ratio** | short_period (5-20), long_period (50-200), threshold (0.5-2.0) | Volatility regime filter |
| **VPIN** | bucket_size, window, threshold (0.3-0.8) | Microstructure toxicity filter |

### 7.2 Enable Indicator-vs-Indicator Comparisons in Genome

**Current gap**: EMA crossovers, BBANDS breakouts, and price-vs-indicator signals can't be evolved because the genome only supports indicator-vs-threshold.

**Proposed gene extension**:
```python
@dataclass
class StrategyGene:
    indicator_type: str
    parameters: dict[str, float]
    condition: ConditionType
    # Current: single threshold
    threshold: float
    # NEW: optional second indicator for comparison
    compare_indicator: str | None = None
    compare_parameters: dict[str, float] | None = None
    compare_sub_value: str | None = None
```

This enables discovering strategies like:
- `ema_8 crosses_above ema_21`
- `close > bbands.upper`
- `rsi_14 > rsi_28` (multi-period RSI comparison)

### 7.3 Add Regime-Aware Fitness Function

**Current fitness**: 35% Sharpe + 25% (1-MaxDD) + 20% PF + 20% Return

**Proposed enhancement**: Add stability and regime metrics:

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Sharpe Ratio | 25% | Risk-adjusted return (reduced from 35%) |
| Max Drawdown | 20% | Tail risk (reduced from 25%) |
| Profit Factor | 15% | Win/loss ratio (reduced from 20%) |
| Total Return | 10% | Absolute return (reduced from 20%) |
| **Calmar Ratio** | 10% | Return / MaxDD — penalizes strategies with deep drawdowns |
| **Win Rate Consistency** | 10% | Std of rolling win rate — rewards consistent strategies |
| **Regime Robustness** | 10% | Performance across detected regimes (trend + ranging) |

### 7.4 Multi-Asset Discovery

**Current limitation**: Discovery runs on a single instrument at a time.

**Proposed enhancement**: Run discovery across multiple instruments simultaneously and select strategies that work across >60% of tested instruments. This dramatically reduces overfitting to a single asset's idiosyncrasies.

### 7.5 Improved Overfitting Prevention

**Current**: WFA, DSR, Purged K-Fold (all available but some optional)

**Proposed additions**:

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Combinatorial Purged Cross-Validation (CPCV)** | Bailey/López de Prado method — tests all possible train/test splits | [Advances in Financial ML, Ch. 12](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) |
| **Monte Carlo Permutation** | Shuffle returns and compare strategy vs random — statistical significance test | Randomly permute bar returns 1000x, check if real Sharpe > 95th percentile |
| **White's Reality Check / SPA Test** | Controls for data snooping across all tested strategies | Adjusts p-values for multiple comparisons |
| **Regime-Conditional WFA** | Run WFA separately for trending vs ranging periods | Split data by Hurst > 0.5 / < 0.5, run WFA on each |

### 7.6 Walk-Forward Optimization Improvements

**Current research** (2025) recommends:

- **Rolling-window reoptimization**: Reoptimize every 30 trading days ([arXiv 2025](https://arxiv.org/html/2510.07943v1))
- **Genetic switch mechanism**: Simultaneously optimize which indicators to use AND their parameters ([ADBAS Scientific 2025](https://journals.adbascientific.com/iteb/article/view/126))
- **Fitness function evolution**: Use Calmar-like ratio with explicit drawdown penalization
- **Population diversity maintenance**: Monitor fitness variance; increase mutation when diversity drops
- **Sortino Ratio as alternative metric**: Penalizes only downside volatility (1.98 vs 1.21 Sortino vs buy-and-hold in research)

---

## 8. Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
*Low effort, high impact — indicators already in pandas-ta-classic or NautilusTrader*

- [ ] Add **SuperTrend** to indicator registry and genome pool
- [ ] Add **ADX** (already in NT!) to indicator registry and genome pool
- [ ] Add **Squeeze Momentum** to indicator registry
- [ ] Add **Aroon Oscillator** (already in NT) to genome pool
- [ ] Add **CMO** to indicator registry and genome pool
- [ ] Add **Realized Volatility Ratio** as custom indicator
- [ ] Update frontend indicator catalog to include new indicators

### Phase 2: Custom Indicators (2-4 weeks)
*Medium effort — custom implementations needed*

- [ ] Implement **Shannon Entropy** indicator (custom NautilusTrader indicator)
- [ ] Implement **Hurst Exponent** indicator (Moving Hurst, rolling R/S method)
- [ ] Implement **VPIN** indicator (Bulk Volume Classification + volume buckets)
- [ ] Implement **GARCH/EGARCH** volatility forecast indicator (wrapping `arch` library)
- [ ] Add all Phase 2 indicators to genome pool with appropriate parameter ranges

### Phase 3: Genome & Fitness Enhancements (2-3 weeks)
*Architectural changes to the discovery pipeline*

- [ ] Enable **indicator-vs-indicator comparisons** in genome (EMA crossovers, price vs bands)
- [ ] Implement **regime-aware fitness function** (Calmar, consistency, regime robustness)
- [ ] Add **Monte Carlo permutation test** to overfitting prevention
- [ ] Add **multi-instrument discovery** (test strategy across multiple pairs)
- [ ] Improve WFA with rolling reoptimization every N days

### Phase 4: Crypto-Specific Signals (3-4 weeks)
*Exchange API integration required*

- [ ] Integrate **funding rate** data from exchange APIs (Binance, Bybit)
- [ ] Implement **funding rate indicators** (z-score, cumulative, basis)
- [ ] Integrate **open interest** data and compute OI-based signals
- [ ] Implement **taker buy/sell ratio** indicator
- [ ] Add funding/OI signals to genome pool

### Phase 5: ML Integration (4-6 weeks)
*Machine learning pipeline*

- [ ] Build **feature engineering pipeline** from indicators to ML features
- [ ] Implement **XGBoost signal filter** (predict signal quality from regime features)
- [ ] Create **ML-filtered strategy template** in DSL
- [ ] Implement **model retraining pipeline** with walk-forward validation
- [ ] Evaluate **RL-based strategy selection** (DQN choosing between strategies)

### Phase 6: Advanced Strategies (4-6 weeks)
*New strategy archetypes beyond single-instrument directional*

- [ ] Implement **pairs trading framework** (cointegration testing, z-score signals, spread trading)
- [ ] Implement **funding rate carry strategy** (delta-neutral with spot hedge)
- [ ] Implement **regime-adaptive strategy** DSL extension (switch logic by Hurst/entropy)
- [ ] Research and prototype **Avellaneda-Stoikov market making** (requires tick data)

---

## 9. References & Sources

### Academic Papers

1. He, Manela, Ross, & von Wachter. ["Fundamentals of Perpetual Futures"](https://arxiv.org/html/2212.06888v5). arXiv, 2024.
2. ["Technical Analysis Meets Machine Learning: Bitcoin Evidence"](https://arxiv.org/html/2511.00665v1). arXiv, Nov 2025.
3. ["Anti-Persistent Values of the Hurst Exponent Anticipate Mean Reversion in Pairs Trading: The Cryptocurrencies Market"](https://www.mdpi.com/2227-7390/12/18/2911). MDPI Mathematics, 2024.
4. ["Bitcoin wild moves: Evidence from order flow toxicity and price jumps"](https://www.sciencedirect.com/science/article/pii/S0275531925004192). ScienceDirect, 2025.
5. Easley et al. ["Microstructure and Market Dynamics in Crypto Markets"](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf). Cornell, 2024.
6. ["Exploring Risk and Return Profiles of Funding Rate Arbitrage on CEX and DEX"](https://www.sciencedirect.com/science/article/pii/S2096720925000818). ScienceDirect, 2025.
7. ["The Two-Tiered Structure of Cryptocurrency Funding Rate Markets"](https://www.mdpi.com/2227-7390/14/2/346). MDPI, Jan 2026.
8. ["Crypto Carry"](https://www.bis.org/publ/work1087.pdf). BIS Working Papers No. 1087.
9. ["Adaptive Multi-Asset Trading Strategy Optimization via GAs with Walk-Forward"](https://journals.adbascientific.com/iteb/article/view/126). ITEB, 2025.
10. ["Agent-Based Genetic Algorithm for Crypto Trading Strategy Optimization"](https://arxiv.org/html/2510.07943v1). arXiv, Oct 2025.
11. ["Cryptocurrency Futures Portfolio Trading System Using Reinforcement Learning"](https://www.mdpi.com/2076-3417/15/17/9400). MDPI Applied Sciences, 2025.
12. ["Copula-based Trading of Cointegrated Cryptocurrency Pairs"](https://link.springer.com/article/10.1186/s40854-024-00702-7). Springer Financial Innovation, 2025.
13. ["Designing a cryptocurrency trading system with deep RL utilizing LSTM and XGBoost"](https://www.sciencedirect.com/science/article/abs/pii/S1568494625003400). ScienceDirect, Mar 2025.
14. ["Forecasting and Trading Cryptocurrencies with ML Under Changing Markets"](https://link.springer.com/chapter/10.1007/978-981-96-6839-7_10). Springer, 2025.
15. ["Machine learning-driven feature selection and anomaly detection for Bitcoin"](https://www.sciencedirect.com/science/article/pii/S1568494625016953). ScienceDirect, 2025.
16. ["High-frequency dynamics of Bitcoin futures: Market microstructure"](https://www.sciencedirect.com/science/article/pii/S2214845025001188). ScienceDirect, 2025.
17. ["The Flow of Information in Trading: An Entropy Approach to Market Regimes"](https://pmc.ncbi.nlm.nih.gov/articles/PMC7597144/). PMC, 2020.
18. ["Shannon Entropy: An Econophysical Approach to Cryptocurrency Portfolios"](https://www.mdpi.com/1099-4300/24/11/1583). MDPI Entropy, 2022.

### Tools & Libraries

19. [NautilusTrader Indicators API](https://docs.nautilustrader.io/api_reference/indicators.html)
20. [nautilus-indicators Rust crate](https://docs.rs/nautilus-indicators)
21. [pandas-ta-classic (200+ indicators)](https://github.com/xgboosted/pandas-ta-classic)
22. [hftbacktest (HFT backtesting with order book)](https://hftbacktest.readthedocs.io/)
23. [Hummingbot (market making bot)](https://hummingbot.org/)
24. [FinRL (RL for trading)](https://github.com/AI4Finance-Foundation/FinRL)
25. [pyine (Pine Script → Python)](https://pypi.org/project/pyine/)
26. [arch library (GARCH models)](https://arch.readthedocs.io/)

### Practitioner Resources

27. [Shannon Entropy & Market Randomness (Robot Wealth)](https://robotwealth.com/shannon-entropy/)
28. [Detecting Trends with Hurst Exponent (Macrosynergy)](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/)
29. [From PIN to VPIN (QuantResearch.org)](https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf)
30. [Stat Arb Models 2025 Deep Dive](https://coincryptorank.com/blog/stat-arb-models-deep-dive)
31. [GARCH Volatility Clustering Across Asset Classes (2026)](https://jonathankinlay.com/2026/02/garch-volatility-clustering-asset-classes/)
32. [Walk-Forward Analysis Deep Dive (IBKR)](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)
33. [Funding Rate Arbitrage Guide (Amberdata)](https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata)
34. [Avellaneda-Stoikov Practical Guide (Algotron)](https://algotron.medium.com/avellaneda-stoikov-market-making-strategy-a-practical-guide-for-crypto-traders-d42d0682c6d1)
35. [On-Chain Metrics for Crypto (Nansen)](https://www.nansen.ai/post/onchain-metrics-key-indicators-for-cryptocurrency-price-prediction)
36. [Algoindex: Institutional Crypto Microstructure Intelligence](https://algoindex.org/)

### TradingView Indicators

37. [TradingView Community Scripts](https://www.tradingview.com/scripts/)
38. [Hurst Momentum Oscillator (AlphaNatt)](https://www.tradingview.com/script/Sc9Ls8Kx-Hurst-Momentum-Oscillator-AlphaNatt/)
39. [Advanced Fractal and Hurst Indicator (PuzzlerTrades)](https://www.tradingview.com/script/zJUoUcxr-Advanced-Fractal-and-Hurst-Indicator/)
40. [Shannon Entropy V2 (kocurekc)](https://www.tradingview.com/script/90gGxKtX-Shannon-Entropy-V2/)
