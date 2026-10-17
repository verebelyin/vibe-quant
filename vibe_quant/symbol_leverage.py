"""The ONE rule for a symbol's Binance USDT-M maximum leverage (bead vibe-quant-4oo1i).

A launch must never run at a leverage the venue refuses. The cap is the COIN MAXIMUM:
the max over a symbol's brackets of ``max_leverage`` (the first bracket's value) in the
bracket snapshot :mod:`vibe_quant.liquidation` also loads for its tier tables.

Stdlib-only on purpose. The API schemas import this module on EVERY launch request, and a
mode-off run must never import :mod:`vibe_quant.liquidation` (import isolation), so the
snapshot is read by path rather than through ``importlib.resources.files(...)``, which
would import the package.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "liquidation" / "binance_um_brackets.json"


class SymbolLeverageError(ValueError):
    """A launch's symbol is missing from the leverage snapshot, or its leverage exceeds the cap."""


@cache
def _table() -> tuple[str, dict[str, int]]:
    """``(snapshot date, {symbol: coin maximum leverage})`` from the bracket snapshot."""
    raw = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    fetched_at = str(raw["fetched_at"])[:10]
    maxima: dict[str, int] = {}
    for symbol, entry in raw["symbols"].items():
        caps = [
            bracket["max_leverage"]
            for bracket in entry["brackets"]
            if bracket.get("max_leverage") is not None
        ]
        if caps:
            maxima[symbol] = int(max(caps))
    return fetched_at, maxima


def max_leverage(symbol: str) -> int:
    """The coin maximum leverage of ``symbol`` (fail closed when the snapshot lacks it)."""
    fetched_at, maxima = _table()
    if symbol not in maxima:
        msg = f"no Binance leverage bracket for {symbol} (snapshot {fetched_at})"
        raise SymbolLeverageError(msg)
    return maxima[symbol]


def check_leverage(symbols: Iterable[str], leverage: float) -> None:
    """Refuse ``leverage`` for the FIRST symbol (in given order) whose coin maximum is lower.

    ``leverage == max`` is allowed: the venue's own cap is launchable.
    """
    for symbol in symbols:
        maximum = max_leverage(symbol)
        if maximum < leverage:
            msg = f"leverage {leverage:g}x exceeds the Binance maximum for {symbol} ({maximum}x)"
            raise SymbolLeverageError(msg)
