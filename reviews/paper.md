# Paper Module Code Review

## Module Overview

The `paper` module implements paper trading functionality for the vibe-quant algorithmic trading engine. It wraps NautilusTrader's `TradingNode` for simulated execution against Binance testnet with real-time WebSocket data.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 35 | Public API exports |
| `cli.py` | 212 | CLI entry point for background subprocess |
| `config.py` | 270 | Configuration dataclasses and factory |
| `errors.py` | 369 | Error classification, retry logic, halt-and-alert |
| `node.py` | 497 | PaperTradingNode state machine and lifecycle |
| `persistence.py` | 356 | SQLite checkpoint save/restore |
| **Total** | **1,739** | |

**Test files:** 5 files, 1,837 lines total.

**Consumers:** Dashboard (`paper_trading.py`), tests (5 files).

---

## Findings

### CRITICAL

#### C-1: `_trading_node` is a placeholder dict, not actual NautilusTrader TradingNode
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, lines 302-308
- **Description:** `_initialize()` sets `self._trading_node` to a plain dictionary instead of instantiating a real `TradingNode`. The `_run_loop()` at line 310-330 just awaits a shutdown event forever. No actual trading occurs -- no WebSocket connections, no order execution, no position tracking. The entire module is a scaffold.
- **Impact:** Paper trading is completely non-functional. The SPEC Section 11 requires real WebSocket data feeds and simulated execution against Binance testnet. None of that exists.
- **Suggested fix:** Implement actual `TradingNode` instantiation with `BinanceDataClientConfig` and `BinanceExecClientConfig`. The `_run_loop` should delegate to `TradingNode.run()`.

#### C-2: No periodic state persistence in PaperTradingNode
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`
- **Description:** The `StatePersistence` class exists with `start_periodic_checkpointing()` but `PaperTradingNode` never instantiates or uses `StatePersistence`. The SPEC Section 11 requires "State persisted to SQLite (positions, orders, balance) for crash recovery" and "Periodic checkpoint every 60 seconds."
- **Impact:** If the paper trading process crashes, all state is lost. No crash recovery is possible.
- **Suggested fix:** In `_initialize()`, create a `StatePersistence` instance and start periodic checkpointing. In `_shutdown()`, save a final checkpoint and close.

#### C-3: No Telegram alert integration
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, line 173
- **Description:** `_on_error_alert()` only writes an event log entry. The SPEC Section 11 requires "Telegram alerts active for errors and circuit breakers." There is no Telegram bot client, no alert dispatch.
- **Impact:** Operators receive no push notifications for fatal errors or circuit breaker triggers during paper trading.
- **Suggested fix:** Integrate with the `vibe_quant.alerts` module (or create it) to send Telegram messages from `_on_error_alert()`.

---

### HIGH

#### H-1: `_on_error_halt` does not cancel open orders
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, lines 160-171
- **Description:** SPEC Section 11 error handling requires: "1. Cancel all open orders for the affected strategy." The `_on_error_halt` callback simply stores previous state and calls `halt()`. No order cancellation occurs.
- **Impact:** On fatal error, open orders remain active on the exchange (or testnet), potentially leading to unintended fills.
- **Suggested fix:** Before transitioning to HALTED, issue cancel-all for the strategy's orders via the execution client.

#### H-2: `halt()` sets shutdown event, preventing recovery
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, line 393
- **Description:** `halt()` calls `self._shutdown_event.set()`, which causes `_run_loop()` to exit and triggers `_shutdown()` in `start()`. After `_shutdown()`, the `StateManager` is closed and the event writer is closed. `resume_from_halt()` changes the state enum but cannot restart the actual trading loop because the node has already shut down.
- **Impact:** `resume_from_halt()` is misleading -- it changes state fields but cannot actually resume trading. The process would need to be restarted externally.
- **Suggested fix:** Either (a) redesign `halt()` to pause the trading loop without triggering full shutdown, or (b) document that `resume_from_halt()` only works when called before the event loop exits (e.g., from within the strategy).

#### H-3: `save_config_to_json` writes API credentials to disk
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/cli.py`, lines 108-139
- **Description:** `save_config_to_json()` serializes `config.binance.api_key` and `config.binance.api_secret` to JSON on disk (lines 111-112). The conventions document states: "API keys and secrets in environment variables only, never in code or config files."
- **Impact:** Security violation. API credentials could leak via config files in logs, backups, or git.
- **Suggested fix:** Exclude `api_key` and `api_secret` from the serialized output, matching how `_create_paper_config_file()` in the dashboard correctly omits them.

#### H-4: `load_config_from_json` creates config with empty API keys
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/cli.py`, lines 57-62
- **Description:** When the JSON config file omits `api_key`/`api_secret` (the correct behavior per conventions), `load_config_from_json` defaults them to empty strings. This creates a `BinanceTestnetConfig` with empty credentials that will fail at runtime without a clear error at config load time.
- **Impact:** Error manifests late at connection time instead of at config validation time.
- **Suggested fix:** After loading from JSON, fall back to environment variables: `api_key = binance_data.get("api_key") or os.getenv("BINANCE_API_KEY", "")`. Then validate that credentials are non-empty in `PaperTradingConfig.validate()`.

#### H-5: `run_with_heartbeat` return value discarded
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/cli.py`, line 169
- **Description:** `run_with_heartbeat(run_id, config.db_path)` returns a `BacktestJobManager` but the return value is ignored. The heartbeat thread runs `while True` with no stop mechanism.
- **Impact:** Minor resource leak concern, but the heartbeat thread is daemon so it dies with the process.
- **Suggested fix:** Store the return value and use it for cleanup during shutdown.

#### H-6: No validation of sizing `method` field
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/config.py`, lines 147-173
- **Description:** `PaperTradingConfig.validate()` checks `risk_per_trade`, `max_leverage`, and `max_drawdown_pct` bounds but never validates that `sizing.method` is one of the supported methods (`fixed_fractional`, `kelly`, `atr`).
- **Impact:** Bad config passes validation, fails deep in the trading pipeline.
- **Suggested fix:** Add `if self.sizing.method not in ("fixed_fractional", "kelly", "atr"): errors.append(...)`.

---

### MEDIUM

#### M-1: `Any` type used for `_trading_node` and `_compile_strategy` return
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, lines 136, 221
- **Description:** `self._trading_node: Any = None` and `def _compile_strategy(...) -> Any:` violate the project convention "Never use the `any` type."
- **Suggested fix:** Use a protocol or `object` type.

#### M-2: `Any` type used for `state_callback` in persistence
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/persistence.py`, lines 281, 305
- **Description:** `state_callback: Any` should be typed as `Callable[[], StateCheckpoint | None] | Callable[[], Awaitable[StateCheckpoint | None]]`.

#### M-3: `persistence.py` silently swallows checkpoint loop exceptions
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/persistence.py`, lines 329-332
- **Description:** The `except Exception: pass` in `_checkpoint_loop` silently discards all errors including database corruption, disk full, etc.
- **Impact:** Critical persistence failures go undetected.
- **Suggested fix:** At minimum, add `logging.exception("Checkpoint failed")`.

#### M-4: `__init__.py` does not export persistence classes
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/__init__.py`
- **Description:** `StateCheckpoint`, `StatePersistence`, and `recover_state` are not in `__all__` or imported in `__init__.py`. External consumers import directly from submodule.

#### M-5: `classify_error` checks transient patterns before fatal
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/errors.py`, lines 84-107
- **Description:** An error containing both "connection" and "authentication failed" would be classified as TRANSIENT because transient patterns are checked first.
- **Example:** `ConnectionError("authentication failed via connection")` -- classified as TRANSIENT, but should be FATAL.
- **Suggested fix:** Check fatal patterns first, or check both and prefer fatal when both match.

#### M-6: No max checkpoint retention or disk space management
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/persistence.py`
- **Description:** `start_periodic_checkpointing` saves a checkpoint every `checkpoint_interval` seconds indefinitely. `delete_old_checkpoints` exists but is never called automatically.
- **Impact:** SQLite database grows unbounded over long paper trading sessions.

#### M-7: `EventWriter` not flushed before shutdown
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, lines 332-350
- **Description:** `_shutdown()` calls `self._event_writer.close()` but there is no explicit `flush()` call before the final shutdown event is written.
- **Suggested fix:** Call `self._event_writer.flush()` after writing the stop event and before `close()`.

#### M-8: `_create_paper_config_file` uses `/tmp` with no cleanup
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/dashboard/pages/paper_trading.py`, line 139
- **Description:** Config files are written to `/tmp/paper_{trader_id}.json` with no cleanup mechanism.

#### M-9: Dashboard opens new DB connections per function call
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/dashboard/pages/paper_trading.py`, lines 55-99
- **Description:** `_get_validated_strategies` and `_get_active_paper_jobs` each create and close their own connections per call.
- **Impact:** Inefficiency. Not a correctness issue due to WAL mode.

#### M-10: `resume_from_halt` does not clear `_shutdown_event`
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, lines 425-463
- **Description:** `resume_from_halt()` changes state but does not clear `self._shutdown_event`. Combined with H-2, `resume_from_halt()` has no practical effect.

---

### LOW

#### L-1: `_decimal_from_str` has unnecessary lazy import
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/cli.py`, lines 30-34
- **Suggested fix:** Move the `Decimal` import to module level.

#### L-2: `ConfigurationError` uses unnecessary `pass`
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/config.py`, lines 23-27

#### L-3: `signal_handler` parameter unused
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, lines 242-245
- **Description:** `def signal_handler(sig: int) -> None:` receives `sig` but never uses it.

#### L-4: `ErrorHandler._transient_counts` never cleaned up for completed operations
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/errors.py`, line 192

#### L-5: `Decimal` serialization preserves trailing zeros
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/cli.py`, lines 118-124
- **Impact:** No functional impact.

#### L-6: `asyncio.get_event_loop()` deprecated since Python 3.10
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, line 240
- **Suggested fix:** Replace with `asyncio.get_running_loop()`.

#### L-7: Float PnL values not rounded in serialization
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/paper/node.py`, lines 76-80

#### L-8: Dashboard hardcodes sizing/risk defaults for paper trading
- **File:** `/Users/verebelyin/projects/vibe-quant/vibe_quant/dashboard/pages/paper_trading.py`, lines 122-137
- **Description:** `_create_paper_config_file` hardcodes sizing/risk values instead of reading from user's configured presets.

---

### INFO

#### I-1: Module is largely scaffolding
- The paper module is structurally complete but the core `_run_loop` and `_initialize` are placeholders. Consistent with Phase 6 in SPEC.

#### I-2: Good test coverage for what exists
- Test count (1,837 lines) exceeds source (1,739 lines). Well-structured with proper fixtures and edge cases.

#### I-3: Error classification is well-designed
- The callback-based halt/alert system is extensible.

#### I-4: Config uses `frozen=True` appropriately for BinanceTestnetConfig
- Prevents accidental credential modification. Good practice.

#### I-5: Dashboard page calls `render_paper_trading_tab()` at module level
- Expected pattern for Streamlit's `st.navigation` API.

#### I-6: `_get_validated_strategies` uses COALESCE for overfitting metrics
- Good defensive SQL allowing strategies without overfitting metrics to be promoted.

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| CRITICAL | 3 |
| HIGH | 6 |
| MEDIUM | 10 |
| LOW | 8 |
| INFO | 6 |
| **Total** | **33** |

---

## Recommendations

### Immediate (before paper trading goes live)

1. **Implement actual TradingNode integration** (C-1). Module is non-functional without it.
2. **Wire up StatePersistence in PaperTradingNode** (C-2). Persistence class is built but unused.
3. **Remove API credential serialization from `save_config_to_json`** (H-3). Security issue.
4. **Fix `load_config_from_json` to read env vars as fallback** (H-4).
5. **Fix error classification priority** (M-5). Check fatal before transient.

### Short-term

6. **Redesign halt/resume mechanism** (H-2, M-10).
7. **Add order cancellation on halt** (H-1). Required by SPEC.
8. **Add Telegram alert dispatch** (C-3). Required by SPEC.
9. **Add validation for sizing method** (H-6). Fail fast on invalid config.
10. **Fix silent exception swallowing in checkpoint loop** (M-3).

### Longer-term

11. **Replace `Any` types** (M-1, M-2). Define proper protocols/types per project conventions.
12. **Add automatic checkpoint retention management** (M-6).
13. **Add integration tests for actual NT TradingNode** once implemented.
14. **Connect dashboard sizing/risk config to paper trading** (L-8).
