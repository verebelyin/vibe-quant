#!/usr/bin/env python3
"""Fetch Binance USDT-M maintenance-margin brackets into a versioned snapshot.

Keyless mode (default) POSTs the public, UNOFFICIAL endpoint
``https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets``,
which returns the *current* brackets for every listed symbol. This is a
snapshot tool: the engine must never call the endpoint at runtime — it loads
the committed JSON instead.

    python scripts/fetch_leverage_brackets.py \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,SUIUSDT,ZECUSDT,NEARUSDT,HYPEUSDT \
        --out vibe_quant/liquidation/binance_um_brackets.json

Only the keyless snapshot is supported (no signed ``leverageBracket`` mode).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx

from vibe_quant.liquidation.tiers import TierTableError, load_tier_tables

BRACKETS_ENDPOINT = "https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,SUIUSDT,ZECUSDT,NEARUSDT,HYPEUSDT"
DEFAULT_OUT = "vibe_quant/liquidation/binance_um_brackets.json"
NOTE = "snapshot as of fetch date; historical bracket changes not modelled"
CLEARANCE_FEE_SOURCE = (
    "Binance trading rules; rate not public -> null = forfeit all remaining margin (conservative)"
)


def _bracket_record(raw: dict[str, object]) -> dict[str, object]:
    leverage = raw.get("maxOpenPosLeverage")
    return {
        "floor": float(raw["bracketNotionalFloor"]),  # type: ignore[arg-type]
        "cap": float(raw["bracketNotionalCap"]),  # type: ignore[arg-type]
        "mmr": float(raw["bracketMaintenanceMarginRate"]),  # type: ignore[arg-type]
        "cum": float(raw["cumFastMaintenanceAmount"]),  # type: ignore[arg-type]
        "max_leverage": int(leverage) if leverage is not None else None,  # type: ignore[arg-type]
    }


def fetch_keyless(symbols: list[str]) -> dict[str, dict[str, object]]:
    """Fetch current brackets for ``symbols`` from the public endpoint."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            BRACKETS_ENDPOINT, headers={"Content-Type": "application/json"}, json={}
        )
        response.raise_for_status()
        payload = response.json()
    rows = payload.get("data", {}).get("brackets")
    if not isinstance(rows, list):
        raise SystemExit(f"unexpected bracket response shape: {payload!r:.200}")
    by_symbol = {str(row["symbol"]): row for row in rows}
    missing = [symbol for symbol in symbols if symbol not in by_symbol]
    if missing:
        raise SystemExit(f"endpoint returned no brackets for: {', '.join(missing)}")
    out: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        risk = by_symbol[symbol]["riskBrackets"]
        out[symbol] = {
            "brackets": [_bracket_record(raw) for raw in risk],
            "clearance_fee_rate": None,
            "clearance_fee_source": CLEARANCE_FEE_SOURCE,
        }
    return out


def build_snapshot(symbols: dict[str, dict[str, object]]) -> dict[str, object]:
    now = dt.datetime.now(dt.UTC)
    return {
        "version": now.date().isoformat(),
        "fetched_at": now.isoformat(),
        "source": BRACKETS_ENDPOINT,
        "note": NOTE,
        "symbols": symbols,
    }


def write_snapshot_atomic(snapshot: dict[str, object], out_path: Path) -> None:
    """Validate ``snapshot`` then atomically replace ``out_path`` with it.

    The JSON is written to a temp file beside the target, validated with
    :func:`vibe_quant.liquidation.tiers.load_tier_tables` and only then moved
    onto the target with ``os.replace``. A snapshot the loader rejects raises
    ``TierTableError`` and leaves the target byte-identical and the directory
    free of any temp file, so a bad fetch never corrupts the live snapshot.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=f".{out_path.name}.tmp-", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, indent=2) + "\n")
        load_tier_tables(Path(tmp_name))
        os.replace(tmp_name, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fetched = fetch_keyless(symbols)
    snapshot = build_snapshot(fetched)

    out_path = Path(args.out)
    try:
        write_snapshot_atomic(snapshot, out_path)
    except TierTableError as exc:
        print(f"refusing to replace {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(symbols)} symbols -> {out_path}")
    for symbol in symbols:
        first = fetched[symbol]["brackets"][0]  # type: ignore[index]
        print(f"  {symbol}: {len(fetched[symbol]['brackets'])} brackets, tier1={first}")  # type: ignore[arg-type,index]
    return 0


if __name__ == "__main__":
    sys.exit(main())
