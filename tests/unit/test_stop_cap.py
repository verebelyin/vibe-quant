"""The stop cap: the GA's widest stop with liquidation off, inside the isolated liquidation distance.

Contract rows R1-R11 and R28b of arch-c29-1. The distance comes from the production
``tiers.isolated_liq_price`` at the run leverage on the 1 000 USDT reference notional; the cap is
95 % of the worst (symbol, side) distance floored to 0.01, and ``None`` where that is the
unchanged 10.0 (see ``docs/claude/discovery.md`` comparability breaks).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from vibe_quant.discovery.stop_cap import StopCap, StopCapLaunchError, run_stop_cap, stop_cap
from vibe_quant.liquidation_settings import LiquidationSettings


def test_btc_10x_cap() -> None:
    """R1: BTCUSDT at 10x is capped at 9.08, bound by the short side (9.5618 %)."""
    assert stop_cap(["BTCUSDT"], 10.0) == StopCap(
        max_pct=9.08,
        leverage=10.0,
        symbol="BTCUSDT",
        side="short",
        liquidation_distance_pct=9.561752988047814,
        tier_snapshot="2026-10-17",
    )


def test_note_keys() -> None:
    """The persisted note carries the bound's provenance, margin and reference notional."""
    cap = stop_cap(["BTCUSDT"], 10.0)
    assert cap is not None
    assert cap.to_note() == {
        "max_pct": 9.08,
        "leverage": 10.0,
        "symbol": "BTCUSDT",
        "side": "short",
        "liquidation_distance_pct": 9.561752988047814,
        "margin": 0.95,
        "reference_notional_usdt": 1000.0,
        "tier_snapshot": "2026-10-17",
    }


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("ETHUSDT", 9.08),
        ("SOLUSDT", 8.98),
        ("BNBUSDT", 8.98),
        ("ZECUSDT", 8.46),
        ("HYPEUSDT", 8.46),
        ("NEARUSDT", 8.46),
        ("SUIUSDT", 8.46),
    ],
)
def test_cap_per_symbol_10x(symbol: str, expected: float) -> None:
    """R2: the 10x cap of every supported symbol."""
    cap = stop_cap([symbol], 10.0)
    assert cap is not None
    assert cap.max_pct == expected


@pytest.mark.parametrize("symbols", [["BTCUSDT", "SOLUSDT"], ["SOLUSDT", "BTCUSDT"]])
def test_multi_symbol_worst(symbols: list[str]) -> None:
    """R3: one genome trades every symbol, so the worst symbol binds, in either order."""
    cap = stop_cap(symbols, 10.0)
    assert cap is not None
    assert (cap.max_pct, cap.symbol, cap.side) == (8.98, "SOLUSDT", "short")


def test_tie_goes_to_the_first_sorted_symbol() -> None:
    """SOL and BNB share a tier table shape: the binding symbol is the first in sorted order."""
    first = stop_cap(["SOLUSDT", "BNBUSDT"], 10.0)
    second = stop_cap(["BNBUSDT", "SOLUSDT"], 10.0)
    assert first == second
    assert first is not None
    assert first.symbol == "BNBUSDT"


@pytest.mark.parametrize(
    ("symbol", "leverage"),
    [("BTCUSDT", 9.0), ("SOLUSDT", 9.0), ("BTCUSDT", 5.0), ("BTCUSDT", 2.0), ("BTCUSDT", 1.0)],
)
def test_no_cap_below_threshold(symbol: str, leverage: float) -> None:
    """R4: where 95 % of the distance is at least 10 % (BTC up to 9x) the range is unchanged."""
    assert stop_cap([symbol], leverage) is None


def test_zec_9x_is_capped() -> None:
    """R5: ZECUSDT starts narrowing one leverage step earlier than BTC (9x -> 9.51)."""
    cap = stop_cap(["ZECUSDT"], 9.0)
    assert cap is not None
    assert cap.max_pct == 9.51


@pytest.mark.parametrize(
    ("symbol", "leverage", "distance"),
    [("BTCUSDT", 107.0, "0.53"), ("ZECUSDT", 65.0, "0.53"), ("SOLUSDT", 97.0, "0.53")],
)
def test_refusal_edge(symbol: str, leverage: float, distance: str) -> None:
    """R6/R7: 106x BTC still draws (cap 0.51); 107x refuses, naming the liquidation distance."""
    allowed = stop_cap(["BTCUSDT"], 106.0)
    assert allowed is not None
    assert allowed.max_pct == 0.51

    with pytest.raises(StopCapLaunchError) as caught:
        stop_cap([symbol], leverage)
    text = str(caught.value)
    assert text.startswith(
        f"--leverage {leverage:g} with liquidation off liquidates a {symbol} short "
    )
    assert text.endswith(
        "% from entry; a 95% stop cap (0.50%) leaves no room above the smallest stop discovery "
        "draws (0.5%). Turn liquidation on (--liquidation-mode isolated or cross) or lower "
        "--leverage."
    )
    assert distance in text


def test_refusal_text_exact() -> None:
    """R7: the one full text the CLI and the API print."""
    with pytest.raises(StopCapLaunchError) as caught:
        stop_cap(["BTCUSDT"], 107.0)
    assert str(caught.value) == (
        "--leverage 107 with liquidation off liquidates a BTCUSDT short 0.53% from entry; a 95% "
        "stop cap (0.50%) leaves no room above the smallest stop discovery draws (0.5%). Turn "
        "liquidation on (--liquidation-mode isolated or cross) or lower --leverage."
    )


def test_empty_symbols() -> None:
    """R8: a cap needs the run's symbols."""
    with pytest.raises(ValueError, match="stop cap needs the run's symbols"):
        stop_cap([], 10.0)


def test_unknown_symbol() -> None:
    """R9: no tier table, no bound: refuse rather than draw unbounded stops."""
    with pytest.raises(StopCapLaunchError) as caught:
        stop_cap(["XYZUSDT"], 10.0)
    assert str(caught.value) == (
        "no Binance tier table for XYZUSDT (snapshot 2026-10-17): cannot bound the stop range "
        "with liquidation off"
    )


@pytest.mark.parametrize(
    ("mode", "leverage", "grid", "port", "capped"),
    [
        ("off", 10.0, False, False, True),
        ("isolated", 10.0, False, False, False),
        ("cross", 10.0, False, False, False),
        ("isolated", 107.0, False, False, False),
        ("off", 10.0, True, False, False),
        ("off", 10.0, False, True, False),
        ("off", 107.0, False, True, False),
    ],
)
def test_run_stop_cap_scope(
    mode: str, leverage: float, grid: bool, port: bool, capped: bool
) -> None:
    """R10/R11: only a plain liquidation-off run is capped; liquidation on, grid and port are not."""
    settings = LiquidationSettings(mode=mode, leverage=leverage)  # type: ignore[arg-type]
    cap = run_stop_cap(["BTCUSDT"], settings, grid=grid, port=port)
    assert (cap is not None) is capped
    if capped:
        assert cap is not None
        assert cap.max_pct == 9.08


@pytest.mark.slow(reason="a fresh interpreter imports the API router and the discovery driver")
def test_import_isolation() -> None:
    """R28b: importing the launch paths loads no liquidation package; using the cap does."""
    probe = (
        "import sys\n"
        "import vibe_quant.api.routers.discovery, vibe_quant.discovery.run\n"
        "from vibe_quant.discovery.stop_cap import stop_cap\n"
        "def loaded():\n"
        "    return sorted(m for m in sys.modules if m.startswith('vibe_quant.liquidation')\n"
        "                  and m != 'vibe_quant.liquidation_settings')\n"
        "before = loaded()\n"
        "stop_cap(['BTCUSDT'], 10.0)\n"
        "print(before, 'vibe_quant.liquidation.tiers' in loaded())\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    assert done.stdout.strip() == "[] True"
