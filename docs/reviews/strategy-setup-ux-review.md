# Strategy Setup UX Review & Improvement Plan

**Date:** 2026-02-08
**Scope:** Strategy Management tab, Backtest Launch tab, DSL editor, form-based editor
**Goal:** Make vibe-quant's strategy setup best-in-class, surpassing Composer.trade, TradingView, QuantConnect, NinjaTrader, and 3Commas

---

## Executive Summary

vibe-quant has a functional dual-mode strategy editor (raw YAML + form-based) with solid Pydantic validation. However, the current UX has significant gaps compared to leading platforms. This review identifies **13 major improvement areas** with specific, actionable recommendations drawn from competitive analysis of 10+ platforms and modern UX research.

The core thesis: **the best strategy setup UX is one where users never need to read documentation to build their first strategy.** Every leading platform that achieves high adoption (Composer, TradingView, 3Commas) does so through progressive disclosure, templates, and inline guidance -- not through better documentation.

---

## Current State Analysis

### What Works Well

1. **Robust validation pipeline** -- Pydantic v2 with field-level, cross-field, and condition-level validation catches errors before they reach the backtester
2. **Clean separation of concerns** -- Strategy definition (DSL) is cleanly separated from backtest configuration (sweep ranges, dates, symbols, latency)
3. **Dual-mode editing** -- Having both YAML and form modes is the right architectural decision (matches Azure DevOps, Home Assistant, Buddy CI/CD pattern)
4. **Comprehensive DSL** -- Supports multi-timeframe, 18 indicator types, 6 condition operators, multiple stop-loss/take-profit types, time filters, and sweep parameters

### What Needs Improvement

The UX falls short in 4 fundamental dimensions:

| Dimension | Current State | Industry Best Practice |
|-----------|--------------|----------------------|
| **Discoverability** | Users must know YAML syntax, indicator names, condition operators | Searchable indicator catalog, autocomplete, inline docs |
| **Feedback speed** | Errors only on Validate/Save click | Real-time validation, live preview, instant parameter feedback |
| **Onboarding** | Blank template with 100-line YAML | Guided wizard, one-click templates, progressive disclosure |
| **Visual clarity** | Flat text areas for conditions | Visual condition builder, flow diagrams, chart preview |

---

## Detailed Findings & Recommendations

### 1. Strategy Template Library (Priority: Critical)

**Current:** New strategy opens with a generic RSI template. No categorized templates, no community strategies.

**Problem:** Users must build strategies from scratch every time. 3Commas, Composer, Coinrule, and Cryptohopper all ship with 50-150+ categorized templates.

**Recommendation:**
- Ship 10-15 strategy templates across categories: **Momentum** (RSI mean reversion, MACD crossover), **Trend** (EMA ribbon, Donchian breakout), **Volatility** (Bollinger squeeze, KC breakout), **Multi-TF** (HTF trend + LTF entry)
- One-click "Use Template" button that populates the editor
- Each template includes: name, description, **difficulty level** (beginner/intermediate/advanced), expected market conditions, and recommended instruments
- Template selector as a card grid with visual previews (strategy logic summary diagram)

**Competitive edge over:** QuantConnect (limited templates), NinjaTrader (no templates), Backtrader (zero templates)

### 2. Guided Strategy Wizard (Priority: Critical)

**Current:** No step-by-step creation flow. Users face the full DSL complexity immediately.

**Problem:** Jakob Nielsen's progressive disclosure principle shows that hiding advanced features behind toggles can increase conversion by 67%. MT5's Strategy Tester is notoriously hard to navigate because it shows everything at once.

**Recommendation:**
- Multi-step wizard for new strategies with progress indicator:
  1. **Choose template or start blank** (card grid)
  2. **Configure indicators** (searchable catalog with descriptions)
  3. **Define entry/exit rules** (visual condition builder)
  4. **Set risk parameters** (stop-loss, take-profit with visual zones)
  5. **Configure sweep parameters** (slider ranges or value lists)
  6. **Review & save** (full YAML preview + validation summary)
- Each step has inline help, example values, and "Why this matters" tooltips
- "Skip to YAML" escape hatch at any step for power users
- Progress auto-saves to session state

**Competitive edge over:** All competitors (none offer a full guided wizard for YAML-based DSL strategies)

### 3. Indicator Catalog with Search (Priority: High)

**Current:** Form editor uses flat `st.selectbox` with 18 unsorted options. No descriptions, no parameter hints, no category grouping.

**Problem:** TradingView's searchable modal with hierarchical categories (Trend, Momentum, Volatility, Volume) is the industry standard. As the indicator library grows beyond 18 types, a flat dropdown becomes unusable.

**Recommendation:**
- Replace flat selectbox with a **searchable indicator catalog modal/expander**:
  - Categories: Trend, Momentum, Volatility, Volume (matching DSL schema)
  - Each indicator shows: name, description, default parameters, typical use case
  - Search filters by name and description
  - "Popular" / "Recently used" quick-access section
- When an indicator is selected, **auto-populate its parameters** with smart defaults based on the selected timeframe (e.g., RSI period 14 for 1h but 21 for 5m)
- Show parameter descriptions inline (not just field names)
- For multi-parameter indicators (MACD: fast/slow/signal, BBANDS: period/std_dev), show **only the relevant fields** for the selected type

**Competitive edge over:** NinjaTrader (dropdown only), MT5 (tree view but no search), QuantConnect (code-only)

### 4. Visual Condition Builder (Priority: High)

**Current:** Entry/exit conditions are free-form text areas where users type condition strings like `"rsi_14 < 30"`. No autocomplete, no syntax highlighting, errors only at save time.

**Problem:** This is the single biggest UX gap. Users must memorize condition syntax, indicator names, and operators. Typos in indicator names aren't caught until validation. Composer and 3Commas solve this with visual builders.

**Recommendation:**
- **Row-based condition builder** with dropdowns:
  - Column 1: Indicator selector (dropdown of defined indicators + `close`, `open`, `high`, `low`)
  - Column 2: Operator (`<`, `>`, `<=`, `>=`, `crosses_above`, `crosses_below`, `between`)
  - Column 3: Value/Indicator (number input or indicator dropdown)
  - Column 4: [+] Add condition, [-] Remove condition
- **AND logic** between conditions (matching current DSL semantics)
- Each row generates the condition string automatically
- "Advanced: raw condition" toggle for each row to allow free-text for edge cases
- Real-time condition validation as users build (green checkmarks per row)
- Option to toggle between visual builder and raw text mode

**Competitive edge over:** QuantConnect (code-only), Backtrader (code-only), Zipline (code-only)

### 5. Enhanced YAML Editor (Priority: High)

**Current:** Plain `st.text_area` with no syntax highlighting, no autocompletion, no inline error markers. 500px fixed height.

**Problem:** Streamlit's text_area is the wrong component for code editing. Leading platforms use Monaco (VS Code engine) or CodeMirror 6 with schema-based validation, hover docs, and code folding.

**Recommendation:**
- Replace `st.text_area` with **`streamlit-monaco`** (or `streamlit-code-editor`):
  - YAML syntax highlighting
  - Schema-based autocompletion (indicator types, timeframes, operators)
  - Inline error markers (red squiggly underlines on validation errors)
  - Hover documentation (show field descriptions from Pydantic model docstrings)
  - Code folding for indicators, conditions, time_filters sections
  - Line numbers
- **Real-time validation** (debounced 500ms) showing errors inline rather than only on button click
- **Side-by-side preview** panel showing the parsed strategy summary (indicators, conditions, risk params) updating live as YAML changes
- **Snippet insertion** buttons for common patterns (add indicator, add condition, add session filter)

**Competitive edge over:** TradingView Pine Editor (language-specific but no YAML schema support), NinjaTrader (C# editor)

### 6. Side-by-Side Editor Mode (Priority: Medium)

**Current:** YAML and form modes are mutually exclusive. A toggle switches between them, and unsaved changes are lost on switch.

**Problem:** Users often want to see the form summary while editing YAML, or vice versa. Home Assistant learned this lesson and now supports both views simultaneously.

**Recommendation:**
- **Split-pane layout**: YAML editor on left, live form preview on right (or vice versa)
- Changes in either pane sync to the other in real-time
- Clicking a form field scrolls the YAML editor to that section
- Clicking a YAML section highlights the corresponding form field
- Mobile/narrow: stack vertically with tabs (current behavior as fallback)

**Competitive edge over:** All competitors (none offer synchronized dual-pane YAML + form editing)

### 7. Sweep Parameter UX Overhaul (Priority: High)

**Current:** Sweep params are configured in the strategy DSL but edited as comma-separated text on the backtest launch page. The backtest page re-parses text inputs. No visual feedback on parameter space size.

**Problem:** Users must understand two different locations for sweep configuration. Comma-separated text inputs are error-prone. No visualization of how parameter combinations interact.

**Recommendation:**
- **In the strategy editor:**
  - Replace raw YAML sweep textarea with a **structured sweep builder**
  - For each sweepable parameter: min/max/step sliders or explicit value list
  - Show total combination count with warnings at >1000 combinations
  - "Quick sweep" presets: Narrow (3 values), Medium (5 values), Wide (10 values)
  - Visual parameter space heatmap preview (2D for top 2 params)
- **In the backtest launch page:**
  - Keep the editable sweep values but add range sliders
  - Show estimated backtest duration based on combination count
  - "Add parameter" button to sweep additional parameters not in the DSL
  - Warn when total combinations exceed reasonable limits (>10,000)

**Competitive edge over:** MT5 (basic grid only), NinjaTrader (limited to brute force), QuantConnect (code-only optimization)

### 8. Risk Parameter Panel with Live Feedback (Priority: Medium)

**Current:** Stop-loss and take-profit configuration shows all parameter fields simultaneously regardless of selected type. No visual feedback on risk/reward ratio.

**Problem:** TradingView and MetaTrader's Position Sizer show risk parameters updating in real-time. The current form shows irrelevant fields (e.g., ATR multiplier when fixed_pct is selected).

**Recommendation:**
- **Conditional field visibility**: Only show relevant parameters for the selected type:
  - `fixed_pct` -> percent field only
  - `atr_fixed` -> ATR multiplier + indicator dropdown (pre-populated from defined ATR indicators)
  - `atr_trailing` -> ATR multiplier + indicator dropdown
  - `risk_reward` -> ratio slider only
- **Live risk/reward visualization**:
  - Show a simple bar chart: entry -> stop loss distance vs entry -> take profit distance
  - Risk/reward ratio calculated and displayed as a metric
  - Warning colors when ratio < 1.0
- **Indicator auto-detection**: If an ATR indicator is defined, auto-populate the stop_loss/take_profit indicator field
- **Quick presets**: "Conservative" (1% SL, 2:1 RR), "Moderate" (2% SL, 1.5:1 RR), "Aggressive" (3% SL, 1:1 RR)

**Competitive edge over:** NinjaTrader (no live feedback), QuantConnect (code-only), Backtrader (code-only)

### 9. Strategy Validation Summary Panel (Priority: Medium)

**Current:** Validation is binary (success/error). No summary of what was validated, no warnings for suboptimal configurations.

**Problem:** Users don't know if their strategy is good, only if it's syntactically valid. MT5 and NinjaTrader provide optimization hints.

**Recommendation:**
- **Validation summary card** after successful validation showing:
  - Number of indicators, conditions, sweep params
  - Risk/reward ratio
  - Estimated backtest combinations
  - Warnings for common issues:
    - "No exit conditions defined for short" (partial coverage)
    - "Sweep has >5000 combinations" (may be slow)
    - "No time filters" (trades 24/7 including weekends)
    - "Stop loss > 5%" (high risk per trade)
    - "Only 1 entry condition" (potentially noisy signals)
  - Strategy complexity score (simple/moderate/complex)
- **"Ready to backtest" checklist** showing what's configured vs missing

**Competitive edge over:** All competitors (none provide a pre-backtest strategy health check)

### 10. Improved Add/Remove Indicator Flow (Priority: High)

**Current:** Form editor renders existing indicators as expanders but has **no way to add new indicators or remove existing ones**. Users must switch to YAML mode.

**Problem:** This is a critical gap that forces users out of form mode for common operations.

**Recommendation:**
- **"+ Add Indicator" button** below the indicator list:
  - Opens the indicator catalog (see item 3)
  - Generates a unique name suggestion (e.g., `rsi_14`, `ema_50`)
  - Adds a new expander with the selected indicator type pre-configured
- **"Remove" button** on each indicator expander
- **"Duplicate" button** to copy an indicator with a new name (useful for sweep variations)
- **Drag-and-drop reordering** of indicators (cosmetic but improves scanability)
- **Indicator dependency visualization**: Show which conditions reference which indicators (highlight orphan indicators)

**Competitive edge over:** 3Commas (limited indicator set), Composer (no custom indicators)

### 11. Session/Time Filter Visual Editor (Priority: Low)

**Current:** Blocked days use multiselect. Sessions have no form representation (only in YAML). Funding avoidance is a checkbox + two number inputs.

**Problem:** Time filters are critical for crypto strategies (funding every 8h, weekend behavior) but the UI treats them as an afterthought.

**Recommendation:**
- **Visual weekly schedule**: 7-day x 24-hour grid where users can paint allowed/blocked trading windows
- **Session quick presets**: "Asia session", "London session", "US session", "24/7"
- **Funding time overlay**: Show the 8h funding schedule on the grid with the avoidance buffer zones
- **Time zone selector** with live clock showing "current exchange time"
- Move from the bottom of the form to a dedicated collapsible section with a clear label

### 12. Backtest Launch UX Improvements (Priority: Medium)

**Current:** The backtest launch page is a long vertical scroll of sections. Strategy details are collapsed by default.

**Recommendation:**
- **Sticky strategy summary header**: Show the selected strategy name, timeframe, and indicator count at the top as users scroll down
- **Collapsible sections** with completion indicators (green check when configured)
- **"Quick backtest" mode**: Pre-fill everything with defaults, one-click to run
- **Parameter combination calculator**: Show real-time count as sweep values change, with estimated duration
- **Date range presets**: "Last 30 days", "Last 90 days", "Last year", "Full history", "Custom"
- **Symbol search with market cap/volume info**: Help users choose relevant trading pairs
- **Pre-flight checklist**: Before running, show a summary dialog: strategy, symbols, date range, combinations, estimated duration, filters enabled

### 13. Anti-Overfitting UX Integration (Priority: Medium)

**Current:** Three checkboxes for overfitting filters (DSR, WFA, Purged K-Fold) with tooltips but no further context.

**Problem:** Research shows 100% performance gap between in-sample and out-of-sample. Users need to understand overfitting risk, not just toggle checkboxes.

**Recommendation:**
- **Always-visible overfitting risk indicator** based on:
  - Number of parameter combinations tested
  - Ratio of in-sample to out-of-sample data
  - Strategy complexity (number of conditions and indicators)
- **Educational tooltips** explaining each filter with visual diagrams
- **"Why this matters" expandable section** showing the Deflated Sharpe formula and walk-forward methodology
- **Results integration**: After backtest, show in-sample vs out-of-sample metrics side by side with visual comparison
- **Automatic warnings**: "You tested 500 combinations on 6 months of data. There is a 64% chance of finding a false positive with 20+ comparisons."

---

## Implementation Priority Matrix

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| P0 | Strategy Template Library | High | Medium |
| P0 | Guided Strategy Wizard | High | High |
| P1 | Indicator Catalog with Search | High | Medium |
| P1 | Visual Condition Builder | High | High |
| P1 | Enhanced YAML Editor (Monaco) | High | Medium |
| P1 | Add/Remove Indicator Flow | High | Low |
| P1 | Sweep Parameter UX Overhaul | Medium | Medium |
| P2 | Risk Parameter Panel | Medium | Medium |
| P2 | Strategy Validation Summary | Medium | Low |
| P2 | Side-by-Side Editor | Medium | High |
| P2 | Backtest Launch UX Improvements | Medium | Medium |
| P2 | Anti-Overfitting UX | Medium | Medium |
| P3 | Time Filter Visual Editor | Low | High |

---

## Competitive Positioning

After implementing these improvements, vibe-quant would surpass competitors in:

| Platform | Vibe-Quant Advantage |
|----------|---------------------|
| TradingView | YAML DSL for version control + visual builder; anti-overfitting built-in |
| QuantConnect | No-code entry path; guided wizard; much lower learning curve |
| NinjaTrader | Web-based; real-time validation; template library |
| Composer.trade | Full YAML escape hatch; multi-timeframe; custom indicators; parameter sweeps |
| 3Commas | Overfitting detection; walk-forward validation; full backtesting pipeline |
| MT5 | Modern web UI; YAML version control; progressive disclosure done right |
| Backtrader | Complete GUI; no Python coding required for strategy definition |

The key differentiator would be: **the only platform that offers both a world-class visual strategy builder AND a version-controllable YAML DSL, with seamless sync between the two, plus built-in anti-overfitting protection.**
