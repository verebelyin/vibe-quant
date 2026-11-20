"""Funding-carry run spec, decision schedule and error classes (design §1.4, §2).

The book is generic in N symbols and trades one pooled account once a week;
this module owns the frozen run parameters, the Monday 00:05:00 UTC decision
schedule both executors share, and the carry error classes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Sequence

DECISION_OFFSET_NS = 5 * 60 * 10**9
SIGNAL_SAFETY_NS = 60 * 10**9
LOOKBACK_NS = 7 * 86_400 * 10**9
PRICE_WINDOW_NS = 60 * 60 * 10**9
MIN_LEG_STEPS = 10
MIN_LEG_NOTIONAL_MULTIPLE = 2
VENUE_LEVERAGE = 10
DAY_NS = 86_400 * 10**9

_WEEK_NS = 7 * DAY_NS
_DECISION_CUTOFF_NS = PRICE_WINDOW_NS + SIGNAL_SAFETY_NS
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
WEIGHTINGS = ("equal", "inverse_vol")
LIQUIDATION_MODES = ("off", "cross")
VOL_LOOKBACK_RANGE = (2, 365)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class CarryDataError(DataUnavailableError):
    """Input a carry run needs is missing (funding, MARK price, sizing tick)."""


class CarryEquityTooSmall(ValueError):
    """The book is below the tradable size of a leg (fail closed)."""

    def __init__(
        self,
        symbol: str,
        decision_ns: int | None,
        leg_notional: float,
        required: float,
    ) -> None:
        self.symbol = symbol
        self.decision_ns = decision_ns
        self.leg_notional = leg_notional
        self.required = required
        super().__init__(
            f"equity too small for {symbol} at {decision_ns}: "
            f"leg notional {leg_notional!r} < required {required!r}"
        )


class CarryLiquidated(CarryEquityTooSmall):
    """The cross-margin account went bust on the MARK path (a ``liquidated`` trial).

    Raised by the liquidation overlay (``portfolio/liquidation.py``) with
    ``decision_ns`` = the bust time, ``symbols`` = the sorted legs held then, and the
    account-wide ``margin_balance`` <= ``maint_margin`` of the trigger minute.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        bust_ts: int,
        margin_balance: float,
        maint_margin: float,
    ) -> None:
        self.symbols = tuple(symbols)
        self.margin_balance = margin_balance
        self.maint_margin = maint_margin
        legs = ",".join(self.symbols)
        super().__init__(legs, bust_ts, margin_balance, maint_margin)
        self.args = (
            f"account bust (cross) at {bust_ts}: legs {legs} margin balance "
            f"{margin_balance!r} <= maintenance {maint_margin!r}",
        )


class CarryExecutionError(RuntimeError):
    """The engine did not execute the planned book (stale, unfilled, drift)."""


def _date_ns(value: str) -> int:
    """``YYYY-MM-DD`` at 00:00 UTC -> epoch ns, never local time (R20, R46)."""
    if not isinstance(value, str) or _DATE_RE.fullmatch(value) is None:
        raise ValueError(f"date must be YYYY-MM-DD, got {value!r}")
    delta = datetime.fromisoformat(value).replace(tzinfo=UTC) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1_000


def decision_times(start_ns: int, end_ns: int) -> list[int]:
    """Every Monday 00:05:00 UTC with ``start_ns <= D <= end_ns - 61 min``."""
    cutoff_ns = end_ns - _DECISION_CUTOFF_NS
    first = start_ns - (start_ns % DAY_NS) + DECISION_OFFSET_NS
    if first < start_ns:
        first += DAY_NS
    days = first // DAY_NS
    first += ((7 - (days + 3) % 7) % 7) * DAY_NS  # 1970-01-01 was a Thursday
    times: list[int] = []
    at = first
    while at <= cutoff_ns:
        times.append(at)
        at += _WEEK_NS
    return times


def _as_float(value: Any, name: str) -> float:
    """Finite float or ``ValueError``; bool/str and huge ints are refused (R42, R45)."""
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


@dataclass(frozen=True)
class CarrySpec:
    """Frozen run parameters; ``__post_init__`` enforces every R8d/R45 range."""

    symbols: tuple[str, ...]
    start_ns: int
    end_ns: int
    initial_balance: float = 10_000.0
    legs_divisor: int = 3
    gross: float = 1.0
    band: float = 0.10
    weighting: str = "equal"
    vol_lookback_days: int = 30
    min_quote_volume: float | None = None
    liquidation_mode: str = "off"

    def __post_init__(self) -> None:
        if self.liquidation_mode not in LIQUIDATION_MODES:
            raise ValueError(
                f"liquidation_mode must be one of {LIQUIDATION_MODES}, got {self.liquidation_mode!r}"
            )
        if self.min_quote_volume is not None:
            min_quote_volume = _as_float(self.min_quote_volume, "min_quote_volume")
            if min_quote_volume <= 0.0:
                raise ValueError(f"min_quote_volume must be > 0, got {self.min_quote_volume!r}")
            object.__setattr__(self, "min_quote_volume", min_quote_volume)
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {self.weighting!r}")
        lookback = self.vol_lookback_days
        low, high = VOL_LOOKBACK_RANGE
        if isinstance(lookback, bool) or not isinstance(lookback, int) or not low <= lookback <= high:
            raise ValueError(f"vol_lookback_days must be an int in [{low}, {high}], got {lookback!r}")
        if self.weighting != "inverse_vol" and lookback != 30:
            raise ValueError("vol_lookback_days needs weighting='inverse_vol'")
        if isinstance(self.legs_divisor, bool) or not isinstance(self.legs_divisor, int):
            raise ValueError(f"legs_divisor must be an int >= 2, got {self.legs_divisor!r}")
        if self.legs_divisor < 2:
            raise ValueError(f"legs_divisor must be an int >= 2, got {self.legs_divisor!r}")
        if isinstance(self.start_ns, bool) or not isinstance(self.start_ns, int):
            raise ValueError(f"start_ns must be an int ns, got {self.start_ns!r}")
        if isinstance(self.end_ns, bool) or not isinstance(self.end_ns, int):
            raise ValueError(f"end_ns must be an int ns, got {self.end_ns!r}")
        gross = _as_float(self.gross, "gross")
        band = _as_float(self.band, "band")
        initial_balance = _as_float(self.initial_balance, "initial_balance")
        if not 0.0 < gross <= 2.0:
            raise ValueError(f"gross must be in (0, 2.0], got {self.gross!r}")
        if not 0.0 <= band < 1.0:
            raise ValueError(f"band must be in [0.0, 1.0), got {self.band!r}")
        if initial_balance <= 0.0:
            raise ValueError(f"initial_balance must be > 0, got {self.initial_balance!r}")
        symbols = tuple(self.symbols)
        if any(not isinstance(s, str) or not s for s in symbols):
            raise ValueError("symbols must be non-empty strings")
        if len(set(symbols)) != len(symbols) or len(symbols) < 2:
            raise ValueError("symbols must be at least 2 distinct symbols")
        if self.start_ns % DAY_NS != 0 or self.end_ns % DAY_NS != 0:
            raise ValueError("start_ns and end_ns must be 00:00 UTC")
        if self.start_ns >= self.end_ns:
            raise ValueError("start_ns must be before end_ns")
        if not decision_times(self.start_ns, self.end_ns):
            raise ValueError(f"no decision in window {self.start_ns}..{self.end_ns}")
        object.__setattr__(self, "symbols", tuple(sorted(symbols)))
        object.__setattr__(self, "gross", gross)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "initial_balance", initial_balance)

    @classmethod
    def from_dates(cls, symbols: Sequence[str], start: str, end: str, **kw: Any) -> CarrySpec:
        """Build from ``YYYY-MM-DD`` dates, parsed as UTC (R20, R46)."""
        return cls(symbols=tuple(symbols), start_ns=_date_ns(start), end_ns=_date_ns(end), **kw)

    def to_json(self) -> dict[str, Any]:
        """The 7 base keys; a variant adds its own keys only when on (default hash kept)."""
        payload: dict[str, Any] = {
            "symbols": list(self.symbols),
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "initial_balance": self.initial_balance,
            "legs_divisor": self.legs_divisor,
            "gross": self.gross,
            "band": self.band,
        }
        if self.weighting == "inverse_vol":
            payload["weighting"] = self.weighting
            payload["vol_lookback_days"] = self.vol_lookback_days
        if self.min_quote_volume is not None:
            payload["min_quote_volume"] = self.min_quote_volume
        if self.liquidation_mode == "cross":
            payload["liquidation_mode"] = self.liquidation_mode
        return payload

    def spec_hash(self) -> str:
        payload = json.dumps(self.to_json(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
