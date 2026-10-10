"""Numpy-backed OHLCV buffer for the compute_fn (pandas) indicator path.

Generated strategies used to append every bar to five Python lists per
timeframe and build a :class:`pandas.DataFrame` from them on each recompute.
:class:`PtaBuffer` keeps the same data in preallocated ``float64`` arrays and
reproduces the legacy semantics exactly:

* the same length after every append and the same trim schedule -- once
  ``len > cap + cap // 4`` the buffer keeps only the last ``cap`` rows
  (``cap == 0`` never trims);
* bit-identical ``float64`` values, so a ``prefix_memo.RecurrenceMemo`` keyed
  on these arrays hits/misses exactly as before;
* :meth:`frame` builds the same column names/order/dtypes as the old
  ``pd.DataFrame`` from the lists.

Trimming/growing allocates fresh arrays instead of shifting in place, so a
DataFrame handed out by an earlier :meth:`frame` never observes later writes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PtaBuffer:
    """Per-timeframe OHLCV buffer with the legacy rolling-cap trim schedule.

    Args:
        cap: Rolling cap in bars; ``0`` means unbounded (cumulative indicators).
        with_close_ns: Also buffer the integer nanosecond bar close time and
            expose it as ``frame().attrs["bar_close_ns"]`` (needs_context path).
    """

    __slots__ = (
        "_cap",
        "_with_close_ns",
        "_n",
        "_open",
        "_high",
        "_low",
        "_close",
        "_volume",
        "_close_ns",
        "_df",
    )

    def __init__(self, cap: int, with_close_ns: bool = False) -> None:
        self._cap = int(cap)
        self._with_close_ns = bool(with_close_ns)
        self._n = 0
        capacity = self._cap + self._cap // 4 + 1 if self._cap else 1024
        self._open = np.empty(capacity, dtype=np.float64)
        self._high = np.empty(capacity, dtype=np.float64)
        self._low = np.empty(capacity, dtype=np.float64)
        self._close = np.empty(capacity, dtype=np.float64)
        self._volume = np.empty(capacity, dtype=np.float64)
        self._close_ns: np.ndarray | None = (
            np.empty(capacity, dtype=np.int64) if self._with_close_ns else None
        )
        self._df: pd.DataFrame | None = None

    def __len__(self) -> int:
        return self._n

    def append(
        self,
        o: float,
        h: float,
        low: float,
        c: float,
        v: float,
        close_ns: int | None = None,
    ) -> None:
        """Append one bar, trimming to the cap once it exceeds it by 25%.

        ``close_ns`` is required when the buffer was created ``with_close_ns``
        and must be integer nanoseconds (it is stored in an ``int64`` array).
        """
        if self._n == self._open.shape[0]:
            self._grow()
        i = self._n
        self._open[i] = o
        self._high[i] = h
        self._low[i] = low
        self._close[i] = c
        self._volume[i] = v
        if self._with_close_ns:
            if close_ns is None:
                msg = "PtaBuffer(with_close_ns=True) needs a close_ns on append"
                raise ValueError(msg)
            ns = self._close_ns
            assert ns is not None
            ns[i] = close_ns
        self._n = i + 1
        self._df = None
        cap = self._cap
        if cap and self._n > cap + (cap // 4):
            self._trim()

    def frame(self) -> pd.DataFrame:
        """The buffered bars as a DataFrame, cached until the next append.

        Column names/order/dtypes match the legacy DataFrame built from the
        lists; ``with_close_ns`` buffers also carry ``attrs["bar_close_ns"]``
        (int64). The returned object is only rebuilt after an :meth:`append`.
        """
        if self._df is None:
            n = self._n
            df = pd.DataFrame(
                {
                    "open": self._open[:n],
                    "high": self._high[:n],
                    "low": self._low[:n],
                    "close": self._close[:n],
                    "volume": self._volume[:n],
                }
            )
            ns = self._close_ns
            if ns is not None:
                df.attrs["bar_close_ns"] = ns[:n]
            self._df = df
        return self._df

    def _trim(self) -> None:
        """Keep the last ``cap`` rows in FRESH arrays (old frame views survive)."""
        cap = self._cap
        src = slice(self._n - cap, self._n)
        capacity = self._open.shape[0]
        o = np.empty(capacity, dtype=np.float64)
        h = np.empty(capacity, dtype=np.float64)
        low = np.empty(capacity, dtype=np.float64)
        c = np.empty(capacity, dtype=np.float64)
        v = np.empty(capacity, dtype=np.float64)
        o[:cap] = self._open[src]
        h[:cap] = self._high[src]
        low[:cap] = self._low[src]
        c[:cap] = self._close[src]
        v[:cap] = self._volume[src]
        self._open, self._high, self._low, self._close, self._volume = o, h, low, c, v
        ns = self._close_ns
        if ns is not None:
            new_ns = np.empty(capacity, dtype=np.int64)
            new_ns[:cap] = ns[src]
            self._close_ns = new_ns
        self._n = cap

    def _grow(self) -> None:
        """Double the capacity (unbounded path) into fresh arrays."""
        n = self._n
        capacity = self._open.shape[0] * 2
        o = np.empty(capacity, dtype=np.float64)
        h = np.empty(capacity, dtype=np.float64)
        low = np.empty(capacity, dtype=np.float64)
        c = np.empty(capacity, dtype=np.float64)
        v = np.empty(capacity, dtype=np.float64)
        o[:n] = self._open[:n]
        h[:n] = self._high[:n]
        low[:n] = self._low[:n]
        c[:n] = self._close[:n]
        v[:n] = self._volume[:n]
        self._open, self._high, self._low, self._close, self._volume = o, h, low, c, v
        ns = self._close_ns
        if ns is not None:
            new_ns = np.empty(capacity, dtype=np.int64)
            new_ns[:n] = ns[:n]
            self._close_ns = new_ns
