"""Screening fails loudly when a requested bar type has no catalog data (vibe-quant-c8va4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.screening.nt_runner import require_bars_in_window

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_bar_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"BTCUSDT-PERP\.BINANCE-1-DAY-LAST-EXTERNAL") as ei:
        require_bars_in_window(str(tmp_path), "BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL", "2024-01-01", "2024-06-01")
    assert "2024-01-01" in str(ei.value)
