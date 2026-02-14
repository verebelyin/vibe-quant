"""Error handling for paper trading.

Provides error classification, handling, and retry logic for transient errors.
Supports halt-and-alert for fatal errors and strategy state management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ErrorCategory(StrEnum):
    """Categories of errors for handling decisions."""

    TRANSIENT = "transient"  # Retry-able (network, timeout, rate limit)
    FATAL = "fatal"  # Halt strategy, alert, require manual intervention
    STRATEGY = "strategy"  # Strategy-specific error, halt strategy only


# Known transient error patterns
_TRANSIENT_PATTERNS = frozenset(
    {
        "timeout",
        "connection reset",
        "connection refused",
        "connection closed",
        "rate limit",
        "too many requests",
        "service unavailable",
        "gateway timeout",
        "temporarily unavailable",
        "network unreachable",
        "dns resolution",
        "socket error",
        "ssl error",
        "eof occurred",
        "broken pipe",
        "reconnect",
        "retry",
    }
)

# Known fatal error patterns
_FATAL_PATTERNS = frozenset(
    {
        "authentication failed",
        "invalid api key",
        "invalid signature",
        "account suspended",
        "insufficient balance",
        "margin insufficient",
        "liquidation",
        "position not found",
        "order rejected",
        "invalid order",
        "permission denied",
        "access denied",
        "forbidden",
    }
)


def classify_error(exception: BaseException) -> ErrorCategory:
    """Classify an exception into an error category.

    Args:
        exception: The exception to classify.

    Returns:
        ErrorCategory based on exception type and message.
    """
    error_msg = str(exception).lower()
    exc_type = type(exception).__name__.lower()

    # Check fatal patterns FIRST so e.g. "authentication failed" isn't
    # caught by the "connection" transient pattern.
    for pattern in _FATAL_PATTERNS:
        if pattern in error_msg:
            return ErrorCategory.FATAL

    # Check for transient patterns
    for pattern in _TRANSIENT_PATTERNS:
        if pattern in error_msg or pattern in exc_type:
            return ErrorCategory.TRANSIENT

    # Check specific exception types for transient errors
    transient_types = (
        "timeout",
        "connection",
        "socket",
        "network",
        "ssl",
        "http",
    )
    if any(t in exc_type for t in transient_types):
        return ErrorCategory.TRANSIENT

    # Default to strategy error (safe default - halts strategy but doesn't alert)
    return ErrorCategory.STRATEGY


@dataclass
class ErrorContext:
    """Context information for error handling.

    Attributes:
        error: The exception that occurred.
        category: Classified error category.
        timestamp: When error occurred.
        operation: What operation was being performed.
        symbol: Symbol involved (if applicable).
        retry_count: Number of retries attempted.
        metadata: Additional context data.
    """

    error: BaseException
    category: ErrorCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    operation: str = ""
    symbol: str = ""
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum retry attempts.
        base_delay_ms: Base delay between retries in milliseconds.
        max_delay_ms: Maximum delay cap in milliseconds.
        exponential_base: Base for exponential backoff.
    """

    max_retries: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    exponential_base: float = 2.0

    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for given attempt number.

        Args:
            attempt: Retry attempt number (0-indexed).

        Returns:
            Delay in milliseconds with exponential backoff.
        """
        delay = self.base_delay_ms * (self.exponential_base**attempt)
        return min(int(delay), self.max_delay_ms)


class ErrorHandler:
    """Handler for paper trading errors.

    Provides centralized error handling with:
    - Transient errors: log + retry with backoff
    - Fatal errors: halt strategy, log critical, prepare alert
    - Strategy errors: halt strategy only, log warning

    Attributes:
        retry_config: Configuration for retry behavior.
        on_halt: Callback when strategy should halt.
        on_alert: Callback when alert should be sent.
    """

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        on_halt: Callable[[str, str], None] | None = None,
        on_alert: Callable[[str, ErrorContext], None] | None = None,
    ) -> None:
        """Initialize error handler.

        Args:
            retry_config: Retry configuration (uses defaults if None).
            on_halt: Callback(reason, message) when halt is needed.
            on_alert: Callback(alert_type, context) when alert is needed.
        """
        self._retry_config = retry_config or RetryConfig()
        self._on_halt = on_halt
        self._on_alert = on_alert
        self._transient_counts: dict[str, int] = {}

    @property
    def retry_config(self) -> RetryConfig:
        """Get retry configuration."""
        return self._retry_config

    def handle_error(
        self,
        error: BaseException,
        operation: str = "",
        symbol: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ErrorContext:
        """Handle an error based on its category.

        Args:
            error: The exception that occurred.
            operation: Description of operation being performed.
            symbol: Symbol involved (if applicable).
            metadata: Additional context data.

        Returns:
            ErrorContext with handling results.
        """
        category = classify_error(error)
        context = ErrorContext(
            error=error,
            category=category,
            operation=operation,
            symbol=symbol,
            metadata=metadata or {},
        )

        if category == ErrorCategory.TRANSIENT:
            self._handle_transient(context)
        elif category == ErrorCategory.FATAL:
            self._handle_fatal(context)
        else:  # STRATEGY
            self._handle_strategy(context)

        return context

    def _handle_transient(self, context: ErrorContext) -> None:
        """Handle transient error with retry logic.

        Args:
            context: Error context.
        """
        key = f"{context.operation}:{context.symbol}"
        current_count = self._transient_counts.get(key, 0) + 1
        self._transient_counts[key] = current_count
        context.retry_count = current_count

        delay_ms = self._retry_config.get_delay_ms(current_count - 1)

        logger.warning(
            "Transient error (attempt %d/%d): %s [%s] - retry in %dms",
            current_count,
            self._retry_config.max_retries,
            context.operation,
            context.error,
            delay_ms,
            extra={
                "error_category": context.category.value,
                "operation": context.operation,
                "symbol": context.symbol,
                "retry_count": current_count,
                "delay_ms": delay_ms,
            },
        )

        # If max retries exceeded, escalate to fatal
        if current_count >= self._retry_config.max_retries:
            logger.error(
                "Max retries exceeded for %s - escalating to fatal",
                context.operation,
            )
            context.category = ErrorCategory.FATAL
            self._handle_fatal(context)

    def _handle_fatal(self, context: ErrorContext) -> None:
        """Handle fatal error with halt and alert.

        Args:
            context: Error context.
        """
        logger.critical(
            "Fatal error - halting strategy: %s [%s]",
            context.operation,
            context.error,
            extra={
                "error_category": context.category.value,
                "operation": context.operation,
                "symbol": context.symbol,
                "metadata": context.metadata,
            },
        )

        # Trigger halt callback
        if self._on_halt is not None:
            reason = f"fatal_error:{type(context.error).__name__}"
            message = f"{context.operation}: {context.error}"
            self._on_halt(reason, message)

        # Trigger alert callback
        if self._on_alert is not None:
            self._on_alert("fatal_error", context)

    def _handle_strategy(self, context: ErrorContext) -> None:
        """Handle strategy error with halt only (no alert).

        Args:
            context: Error context.
        """
        logger.warning(
            "Strategy error - halting: %s [%s]",
            context.operation,
            context.error,
            extra={
                "error_category": context.category.value,
                "operation": context.operation,
                "symbol": context.symbol,
            },
        )

        # Trigger halt callback but no alert
        if self._on_halt is not None:
            reason = f"strategy_error:{type(context.error).__name__}"
            message = f"{context.operation}: {context.error}"
            self._on_halt(reason, message)

    def reset_retry_count(self, operation: str = "", symbol: str = "") -> None:
        """Reset retry count for an operation.

        Called after successful operation to clear transient error state.

        Args:
            operation: Operation identifier.
            symbol: Symbol identifier.
        """
        key = f"{operation}:{symbol}"
        if key in self._transient_counts:
            del self._transient_counts[key]

    def should_retry(self, context: ErrorContext) -> bool:
        """Check if error should be retried.

        Args:
            context: Error context from handle_error.

        Returns:
            True if retry should be attempted.
        """
        return (
            context.category == ErrorCategory.TRANSIENT
            and context.retry_count < self._retry_config.max_retries
        )

    def get_retry_delay_ms(self, context: ErrorContext) -> int:
        """Get retry delay for error context.

        Args:
            context: Error context.

        Returns:
            Delay in milliseconds before retry.
        """
        return self._retry_config.get_delay_ms(context.retry_count - 1)


__all__ = [
    "ErrorCategory",
    "ErrorContext",
    "ErrorHandler",
    "RetryConfig",
    "classify_error",
]
