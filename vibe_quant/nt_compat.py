"""Compatibility helpers for NautilusTrader >= 1.223.

NT 1.223+ hard-aborts the process (Rust panic: "attempted to set a logger
after the logging system was already initialized") when a BacktestEngine or
BacktestNode is disposed and a new one is created in the same process: dispose
drops the kernel ``LogGuard`` which tears down the logging subsystem, but the
Rust ``log`` crate's global logger can only be set once per process, so the
next kernel's ``init_logging`` panics.

Retaining the first kernel's ``LogGuard`` for the life of the process keeps
``is_logging_initialized()`` true, so subsequent kernels skip re-init and log
through the existing subsystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine

_LOG_GUARDS: list[object] = []


def retain_log_guard(engine: BacktestEngine) -> None:
    """Keep ``engine``'s kernel LogGuard alive for the life of the process.

    Call before ``dispose()``. Only the first engine in a process actually
    owns a guard (later kernels see logging already initialized and hold
    None), so the retained list stays effectively size one.
    """
    guard = engine.kernel.get_log_guard()
    if guard is not None:
        _LOG_GUARDS.append(guard)
