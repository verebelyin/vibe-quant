# Code Review: `vibe_quant/ethereal` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-11
**Scope:** Full module review -- all 8 source files + 5 test files

---

## Module Overview

The `ethereal` module implements DEX integration for the Ethereal exchange (Phase 7 of SPEC.md).

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 134 | Public API re-exports |
| `__main__.py` | 126 | CLI entry point |
| `archive.py` | 312 | SQLite raw data archive |
| `data_client.py` | 574 | WebSocket data client (Socket.IO) |
| `execution_client.py` | 585 | REST execution + EIP-712 signing |
| `ingestion.py` | 543 | Historical data ingestion |
| `instruments.py` | 175 | NT instrument definitions |
| `venue.py` | 153 | Venue config for backtest/paper |
| **Total source** | **2602** | |
| **Total tests** | **2170** | 5 test files |

---

## Findings

### CRITICAL (2)

#### C-1: Missing timezone on datetime.fromtimestamp in get_fills
**File:** `execution_client.py`, line 571

`datetime.fromtimestamp(int(f["timestamp"]) / 1000)` creates naive datetime using system local timezone. Every other timestamp in the module correctly uses `tz=UTC`.

**Fix:** Add `tz=UTC` parameter.

#### C-2: Nonce generation has collision risk under concurrent use
**File:** `execution_client.py`, lines 249-251

`generate_nonce()` docstring claims "timestamp + random bits" but uses only `int(time.time() * 1000)` -- no random bits. `_nonce_counter` resets to 0 on new client instances. For a non-custodial exchange, collision could cause order rejection or replay.

**Fix:** Use `time.time_ns()` + `secrets.token_bytes`.

### HIGH (5)

#### H-1: base_currency set to USDE instead of actual base asset
**File:** `instruments.py`, line 108

`EtherealInstrumentConfig` defines base_currency as "BTC"/"ETH"/"SOL" but `create_ethereal_instrument` sets `base_currency=USDE` for all. Affects position value and PnL calculations.

**Fix:** `base_currency=Currency.from_str(config.base_currency)`

#### H-2: Private key exposed in repr() output
**File:** `execution_client.py`, lines 134-146

`EtherealConfig` auto-generated `__repr__` displays the private key in logs/debuggers.

**Fix:** Add `repr=False` to private_key field or override `__repr__`.

#### H-3: starting_balance_usdt naming but Ethereal settles in USDe
**File:** `venue.py`, lines 59, 69

Currency mismatch between field name and actual settlement.

**Fix:** Rename to `starting_balance_usde`.

#### H-4: HTTP errors silently swallowed during data download
**File:** `ingestion.py`, lines 156-160, 221-224

Non-404 errors (429, 500) silently skipped, creating data gaps.

**Fix:** Log errors, add retry logic, validate bar counts.

#### H-5: Reconnect loop creates orphaned tasks
**File:** `data_client.py`, lines 252-255

Multiple `on_disconnect` fires create parallel `_reconnect_loop` tasks without cancelling previous ones.

**Fix:** Cancel existing `_reconnect_task` before creating new one.

### MEDIUM (9)

**M-1:** `EtherealArchive` lacks `__enter__`/`__exit__` context manager (`archive.py`).

**M-2:** `EtherealExecutionClient` lacks `__aenter__`/`__aexit__` async context manager.

**M-3:** `wallet_address` property re-derives key every access via `Account.from_key()` (`execution_client.py` lines 326-329).

**M-4:** Ethereal latency presets set non-zero operation-specific latencies that are additive with base (`venue.py` lines 37-49). Likely double-counting.

**M-5:** `_safe_years_ago` uses private naming but is imported externally (`archive.py` line 60).

**M-6:** `cancel_order` EIP-712 domain omits `verifyingContract` (`execution_client.py` lines 444-465).

**M-7:** `EtherealVenueConfig` has no `to_backtest_venue_config()` method.

**M-8:** `klines_to_bars` sets `ts_event=open_time`, `ts_init=close_time`. NT convention may expect `ts_event=close_time`.

**M-9:** `download_bars` does not log non-404 HTTP errors (`ingestion.py` lines 156-157).

### LOW (8)

**L-1:** `verifying_contract` defaults to zero address with no validation.

**L-2:** `EtherealOrder` uses "BTC-USD" format but data client uses "BTCUSD". No conversion.

**L-3:** `ingest_ethereal` returns `len(klines)` not actual inserted count.

**L-4:** `download_funding_rates` logger format has 4 placeholders but 3 arguments. Will crash at runtime.

**L-5:** `tuple[Any, ...]` used for kline types instead of concrete types.

**L-6:** `get_ethereal_status` loads all funding rates into memory just to count them.

**L-7:** Duplicate `ETHEREAL_FUNDING_INTERVAL` constant in `instruments.py` and `venue.py`.

**L-8:** `ingest_ethereal_funding` returns `len(rates)` not actual inserted count.

### INFO (5)

**I-1:** Callbacks are synchronous but called from async handlers. Will block event loop.

**I-2:** No NautilusTrader `LiveDataClient`/`LiveExecutionClient` subclassing. Cannot plug into TradingNode.

**I-3:** Test coverage gaps: no tests for CLI, `ingest_all_ethereal`, `get_ethereal_status`, non-404 errors, `cancel_order` signature.

**I-4:** `archive_to_catalog` does not write funding rates to catalog.

**I-5:** Module only imported within itself and tests. Integration not yet wired.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH | 5 |
| MEDIUM | 9 |
| LOW | 8 |
| INFO | 5 |
| **Total** | **29** |

## Recommendations

**Priority 1 (before live/paper use):** C-1, C-2, H-1, H-2, H-4

**Priority 2 (before backtest validation):** M-4, M-6, M-7, M-8, L-4

**Priority 3 (improvement):** M-1, M-2, M-3, H-3, I-4, I-2, L-5, I-3
