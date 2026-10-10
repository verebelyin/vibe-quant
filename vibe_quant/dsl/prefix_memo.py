"""Exact recurrence memo for compute_fn indicators recomputed once per bar.

Generated strategies re-run each indicator over a rolling OHLCV buffer every
bar, so a recurrence (e.g. Wilder RMA) is recomputed from scratch although its
input only grew by one element. :class:`RecurrenceMemo` caches the output of a
PURE recurrence keyed on its actual input arrays: a lookup hits only when every
input equals a stored input plus exactly one appended element, compared
bitwise (uint64 view, so NaN payloads and +-0 are distinguished). The extended
result comes from the same step helper as the full recompute, so it is
bit-identical by construction.

Never key on a DataFrame or close prefix: indicator outputs are not
prefix-stable (window-wide eps guards flip earlier values), only the recurrence
on its own inputs is.

Kill switch: ``VIBE_QUANT_PTA_MEMO=0``, read at import (and by :func:`refresh_enabled`).

Single-threaded by design: the global memo is shared by every strategy instance in
the process (one engine runs one instance per symbol, same key, sequentially)
and is not locked. Per process it holds at most ``MAX_BYTES`` (x workers).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

# 16 slots: ADX keeps 4 live series per symbol under one key (atr, +dm, -dm, dx
# share ("rma", length) across all symbols in a process), so 16 covers 4 symbols.
# Beyond that the LRU thrashes and the miss path costs ~= the unmemoized path.
SLOTS_PER_KEY = 16
MAX_KEYS = 32
MAX_BYTES = 13 * 1024 * 1024
MAX_LEN = 16384  # above this (unbounded cumulative buffers) bypass the memo

_Entry = tuple[tuple[np.ndarray, ...], np.ndarray, Any]


_enabled = os.environ.get("VIBE_QUANT_PTA_MEMO", "1") != "0"


def memo_enabled() -> bool:
    return _enabled


def refresh_enabled() -> None:
    """Re-read ``VIBE_QUANT_PTA_MEMO`` (tests flip it at runtime)."""
    global _enabled
    _enabled = os.environ.get("VIBE_QUANT_PTA_MEMO", "1") != "0"


def _freeze(a: np.ndarray) -> np.ndarray:
    c = np.array(a, dtype=np.float64, copy=True)
    c.flags.writeable = False
    return c


def _nbytes(entry: _Entry) -> int:
    return sum(a.nbytes for a in entry[0]) + entry[1].nbytes


class RecurrenceMemo:
    """Global LRU: ``slots`` entries per key, ``max_keys`` keys, ``max_bytes`` total."""

    def __init__(
        self,
        slots: int = SLOTS_PER_KEY,
        max_keys: int = MAX_KEYS,
        max_bytes: int = MAX_BYTES,
    ) -> None:
        self.slots = slots
        self.max_keys = max_keys
        self.max_bytes = max_bytes
        self._d: OrderedDict[Hashable, list[_Entry]] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._d.clear()
        self.bytes = 0
        self.hits = self.misses = 0

    @property
    def n_keys(self) -> int:
        return len(self._d)

    def n_slots(self, key: Hashable) -> int:
        return len(self._d.get(key, ()))

    def lookup(self, key: Hashable, inputs: Sequence[np.ndarray]) -> tuple[np.ndarray, Any] | None:
        """Pop and return ``(out, state)`` of the entry that is ``inputs`` minus its last element."""
        lst = self._d.get(key)
        if lst is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        n = len(inputs[0])
        for i, (pins, out, state) in enumerate(lst):
            if len(pins[0]) != n - 1:
                continue
            # cheap reject before the full compare: last prefix element, bitwise
            if inputs[0][-2:-1].view(np.uint64)[0] != pins[0][-1:].view(np.uint64)[0]:
                continue
            if all(
                np.array_equal(x[:-1].view(np.uint64), p.view(np.uint64))
                for x, p in zip(inputs, pins, strict=True)
            ):
                # Pop the matched slot: the extended entry replaces it, so the
                # slots hold distinct live series instead of stale prefixes.
                self.bytes -= _nbytes(lst.pop(i))
                self.hits += 1
                return out, state
        self.misses += 1
        return None

    def store(
        self, key: Hashable, inputs: Sequence[np.ndarray], out: np.ndarray, state: Any
    ) -> None:
        entry: _Entry = (tuple(_freeze(x) for x in inputs), _freeze(out), state)
        lst = self._d.setdefault(key, [])
        self._d.move_to_end(key)
        lst.insert(0, entry)
        self.bytes += _nbytes(entry)
        while len(lst) > self.slots:
            self.bytes -= _nbytes(lst.pop())
        while len(self._d) > self.max_keys or (self.bytes > self.max_bytes and len(self._d) > 1):
            _, dropped = self._d.popitem(last=False)
            self.bytes -= sum(_nbytes(e) for e in dropped)
        while self.bytes > self.max_bytes and len(lst) > 1:  # one key left: shed its oldest slots
            self.bytes -= _nbytes(lst.pop())


MEMO = RecurrenceMemo()


def memo_run(
    key: Hashable,
    inputs: Sequence[np.ndarray],
    full: Callable[[Sequence[np.ndarray]], tuple[np.ndarray, Any]],
    step: Callable[[Any, Sequence[np.ndarray], int], tuple[float, Any]],
) -> np.ndarray:
    """Run a recurrence: extend a memoized prefix by one step, else ``full(inputs)``.

    ``full`` returns ``(out, state)``; ``step(state, inputs, i)`` returns
    ``(value_i, state)``. Both must use one shared step helper.
    """
    if not memo_enabled() or len(inputs[0]) > MAX_LEN:
        return full(inputs)[0]
    hit = MEMO.lookup(key, inputs)
    if hit is not None:
        out_prev, state = hit
        value, state = step(state, inputs, len(inputs[0]) - 1)
        out = np.append(out_prev, value)
    else:
        out, state = full(inputs)
    MEMO.store(key, inputs, out, state)
    return out
