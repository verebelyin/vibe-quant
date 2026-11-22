# Code Review: `vibe_quant/data` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** All files in vibe_quant/data/ and related tests

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 15 | Module exports |
| `__main__.py` | 7 | CLI entry point |
| `archive.py` | 458 | SQLite raw data archive |
| `downloader.py` | 279 | Binance data download |
| `catalog.py` | 411 | NautilusTrader catalog mgmt |
| `ingest.py` | 900 | Orchestration pipeline |
| `verify.py` | 155 | Data quality checks |
| **Total** | **~2225** | |

---

## Findings

### HIGH (2)

#### H-1: Incorrect base_currency semantics in catalog.py
**File:** `catalog.py:95`

`base_currency=USDT` for USDT-M perpetuals. NautilusTrader's `CryptoPerpetual.base_currency` represents the traded asset (BTC, ETH, SOL), not settlement. Current code sets all instruments to USDT as base.

**Impact:** Incorrect position value and PnL calculations throughout NautilusTrader's internal accounting.

**Fix:**
```python
base_currency=Currency.from_str(config["base_currency"]),  # BTC for BTCUSDT
quote_currency=USDT,
settlement_currency=USDT,
```

#### H-2: Type error in _print_summary call
**File:** `ingest.py:509`

`_print_summary(results, archive._db_path, ...)` passes `Path | None` to function expecting `Path`. If archive initialized with `db_path=None`, `_print_summary` line 571 calls `.stat()` on None.

**Impact:** Runtime TypeError when archive uses default path.

**Fix:** `archive._db_path or DEFAULT_ARCHIVE_PATH`

### MEDIUM (2)

**M-1:** Duplicate WAL connection logic between `archive.py:86-96` and `db/connection.py:11-34`. Both implement identical PRAGMA configuration. Violates DRY.

**Fix:** Refactor `RawDataArchive` to use `db.get_connection()`.

**M-2:** Optional client typing in `downloader.py:58,99` causes nullable call/close errors under strict mypy. Non-None client used without narrowing.

### LOW (2)

**L-1:** Duplicate monthly-range generation utility with ethereal module (`downloader.py:102` vs `ethereal/ingestion.py:68`).

**L-2:** Catalog helper methods at `catalog.py:341,386` appear unreferenced from app code.

### INFO (2)

**I-1:** Bar aggregation is well tested with known-result checks. Precision and timestamp conversions explicitly verified.

**I-2:** Archive-first design matches SPEC section 4 reproducibility requirements. Catalog is rebuildable from archive.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 2 |
| MEDIUM | 2 |
| LOW | 2 |
| INFO | 2 |
| **Total** | **8** |

## Recommendations

**Priority 1:** Fix base_currency field in catalog.py -- affects all instrument modeling and PnL.

**Priority 2:** Fix _print_summary type error, consolidate WAL connection logic.

**Priority 3:** Deduplicate monthly-range utility, clean up unreferenced helpers.
