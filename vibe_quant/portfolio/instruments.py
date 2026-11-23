"""Frozen research instrument snapshot for the 1h tier (arch-c9-2 R45, R46, R50; C9-14).

The research coins have no catalog instrument, so a 1h run takes every
instrument from the committed ``research_instruments.json``. It is built
OFFLINE -- this module has no fetch path -- by :func:`build_snapshot` from a
saved Binance exchangeInfo payload plus the archive's 1h closes before a
cutoff, and records its own provenance: the payload's SHA-256 and capture time,
the archive cutoff, the derivation rule and the effective filters. Same inputs
give the same bytes; rows at or after the cutoff are never read, so appending
data never changes the snapshot or its fingerprint.

Rebuild (from the repo root)::

    python -m vibe_quant.portfolio.instruments \\
        --payload data/research/instruments/binance-um-exchangeinfo-20261122.json \\
        --archive data/archive/raw_data.db --cutoff 2026-11-22 \\
        --universe vibe_quant/data/universe_top20.txt \\
        --fetched-at "2026-11-22T16:34:11Z (payload file mtime)"

``captured_at`` is the payload's own ``serverTime`` (the venue's clock);
``fetched_at`` is the local fetch time, recorded verbatim (R-c9t3-3). The payload's
SHA-256 is its identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from vibe_quant.fees import FeeSchedule
from vibe_quant.portfolio.spec import CarryDataError, _date_ns

if TYPE_CHECKING:
    from collections.abc import Sequence

SNAPSHOT_PATH = Path(__file__).with_name("research_instruments.json")
FEE_SOURCE = "FeeSchedule VIP0 (no BNB discount)"
DERIVATION = (
    "price_increment = PRICE_FILTER tickSize when every archived 1h close with open_time < "
    "cutoff is a multiple of it, else 10^-p with p the most decimal places of tickSize and "
    "those closes; size_increment = LOT_SIZE stepSize; min_quantity = LOT_SIZE minQty; "
    "min_notional = MIN_NOTIONAL notional; margin_init / margin_maint = requiredMarginPercent "
    "/ maintMarginPercent divided by 100; maker/taker fee = " + FEE_SOURCE
)
ASSUMPTIONS = (
    "today's filters apply to all history (step, min quantity and min notional were never "
    "finer before); a close at or after the cutoff off this grid fails a run at load "
    "(rebuild the snapshot with a later cutoff)"
)
_VENUE = Venue("BINANCE")
_NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class ResearchSnapshot:
    """A loaded snapshot: its file, the SHA-256 of its bytes and the parsed entries."""

    path: Path
    sha256: str
    symbols: dict[str, dict[str, Any]]


def load_snapshot(path: Path | None = None) -> ResearchSnapshot:
    """Read a snapshot (default: the committed :data:`SNAPSHOT_PATH`)."""
    target = SNAPSHOT_PATH if path is None else path
    raw = target.read_bytes()
    return ResearchSnapshot(target, hashlib.sha256(raw).hexdigest(), json.loads(raw)["symbols"])


def price_increment(snapshot: ResearchSnapshot, symbol: str) -> Decimal:
    """The snapshot's effective price grid of ``symbol`` (R41's close check)."""
    return Decimal(_entry(snapshot, symbol)["price_increment"])


def onboard_ms(snapshot: ResearchSnapshot, symbol: str) -> int | None:
    """The snapshot's Binance ``onboardDate`` of ``symbol`` (ms), the listing evidence."""
    value = _entry(snapshot, symbol)["onboard_ms"]
    return None if value is None else int(value)


def _entry(snapshot: ResearchSnapshot, symbol: str) -> dict[str, Any]:
    entry = snapshot.symbols.get(symbol)
    if entry is None:
        raise CarryDataError(f"{symbol}: not in {snapshot.path.name} (sha {snapshot.sha256})")
    return entry


def research_instrument(snapshot: ResearchSnapshot, symbol: str) -> CryptoPerpetual:
    """The NT instrument of a snapshot entry (R45: absent -> ``CarryDataError``)."""
    entry = _entry(snapshot, symbol)
    quote = Currency.from_str(entry["quote"])
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(f"{symbol}-PERP"), _VENUE),
        raw_symbol=Symbol(symbol),
        base_currency=Currency.from_str(entry["base"]),
        quote_currency=quote,
        settlement_currency=quote,
        is_inverse=False,
        price_precision=entry["price_precision"],
        size_precision=entry["size_precision"],
        price_increment=Price.from_str(entry["price_increment"]),
        size_increment=Quantity.from_str(entry["size_increment"]),
        max_quantity=None,
        min_quantity=Quantity.from_str(entry["min_quantity"]),
        max_notional=None,
        min_notional=Money(Decimal(entry["min_notional"]), quote),
        max_price=None,
        min_price=None,
        margin_init=Decimal(entry["margin_init"]),
        margin_maint=Decimal(entry["margin_maint"]),
        maker_fee=Decimal(entry["maker_fee"]),
        taker_fee=Decimal(entry["taker_fee"]),
        ts_event=0,
        ts_init=0,
    )


# ---------------------------------------------------------------------------
# Offline builder
# ---------------------------------------------------------------------------


def _plain(value: Decimal) -> str:
    """``value`` without trailing zeros or an exponent (``0.10`` -> ``0.1``, ``1E+1`` -> ``10``)."""
    return format(value.normalize(), "f")


def _places(value: Decimal) -> int:
    exponent = value.normalize().as_tuple().exponent
    assert isinstance(exponent, int)  # finite by construction
    return max(0, -exponent)


def _positive(symbol: str, field: str, raw: object) -> Decimal:
    """A filter value as a finite Decimal > 0, else ``ValueError`` naming symbol and field."""
    try:
        value = Decimal(str(raw))
    except InvalidOperation:
        value = Decimal("NaN")
    if raw is None or not value.is_finite() or value <= 0:
        raise ValueError(f"{symbol}: {field} must be a finite number > 0, got {raw!r}")
    return value


def _filters(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["filterType"]: f for f in entry.get("filters", [])}


def _closes(conn: sqlite3.Connection, symbol: str, cutoff_ms: int) -> list[tuple[int, float]]:
    return [
        (int(open_ms), float(close))
        for open_ms, close in conn.execute(
            "SELECT open_time, close FROM raw_klines WHERE symbol = ? AND interval = '1h' "
            "AND open_time < ? ORDER BY open_time",
            (symbol, cutoff_ms),
        )
    ]


def _increment(tick: Decimal, closes: Sequence[tuple[int, float]]) -> Decimal:
    """The tick, or the finest grid every archived close sits on (R46)."""
    prices = [Decimal(repr(close)) for _, close in closes if close > 0.0 and close == close]
    if all(price % tick == 0 for price in prices):
        return tick
    places = max([_places(tick), *(_places(price) for price in prices)])
    return Decimal(1).scaleb(-places)


def _symbol_entry(
    symbol: str, entry: dict[str, Any], closes: Sequence[tuple[int, float]]
) -> dict[str, Any]:
    filters = _filters(entry)
    tick = _positive(symbol, "tickSize", filters.get("PRICE_FILTER", {}).get("tickSize"))
    lot = filters.get("LOT_SIZE", {})
    step = _positive(symbol, "stepSize", lot.get("stepSize"))
    min_qty = _positive(symbol, "minQty", lot.get("minQty"))
    if min_qty % step:
        raise ValueError(
            f"{symbol}: minQty {lot.get('minQty')} is not a multiple of stepSize {lot.get('stepSize')}"
        )
    notional = _positive(symbol, "notional", filters.get("MIN_NOTIONAL", {}).get("notional"))
    initial = _positive(symbol, "requiredMarginPercent", entry.get("requiredMarginPercent"))
    maint = _positive(symbol, "maintMarginPercent", entry.get("maintMarginPercent"))
    increment = _increment(tick, closes)
    fees = FeeSchedule()
    digest = hashlib.sha256(
        "".join(f"{open_ms}:{close!r}\n" for open_ms, close in closes).encode()
    ).hexdigest()
    return {
        "base": entry["baseAsset"],
        "quote": entry["quoteAsset"],
        "onboard_ms": entry.get("onboardDate"),
        "tick_size": str(filters["PRICE_FILTER"]["tickSize"]),
        "price_increment": _plain(increment),
        "price_precision": _places(increment),
        "size_increment": _plain(step),
        "size_precision": _places(step),
        "min_quantity": _plain(min_qty),
        "min_notional": _plain(notional),
        "margin_init": _plain(initial / 100),
        "margin_maint": _plain(maint / 100),
        "maker_fee": _plain(fees.maker_rate),
        "taker_fee": _plain(fees.taker_rate),
        "archive_1h_closes": len(closes),
        "archive_1h_closes_sha256": digest,
    }


def _iso_ms(ms: int) -> str:
    stamp = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


def build_snapshot(
    payload_path: Path,
    archive_path: Path,
    symbols: Sequence[str],
    cutoff_ns: int,
    *,
    fetched_at: str | None = None,
) -> bytes:
    """The snapshot JSON bytes for ``symbols`` (deterministic, offline, R46 / C9-14).

    ``fetched_at`` (the local fetch time) is stored verbatim beside the payload's
    ``captured_at``; without it the key is left out.

    Raises:
        ValueError: a symbol is missing from the payload, or one of its filters is
            absent, zero, not finite, or ``minQty`` is off the ``stepSize`` grid.
    """
    payload_raw = Path(payload_path).read_bytes()
    payload = json.loads(payload_raw)
    by_symbol = {entry["symbol"]: entry for entry in payload["symbols"]}
    cutoff_ms = cutoff_ns // _NS_PER_MS
    conn = sqlite3.connect(f"{Path(archive_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        entries: dict[str, dict[str, Any]] = {}
        for symbol in sorted(symbols):
            entry = by_symbol.get(symbol)
            if entry is None:
                raise ValueError(f"{symbol}: not in exchangeInfo payload {Path(payload_path).name}")
            entries[symbol] = _symbol_entry(symbol, entry, _closes(conn, symbol, cutoff_ms))
    finally:
        conn.close()
    server_ms = int(payload["serverTime"])
    exchange_info: dict[str, Any] = {
        "file": Path(payload_path).name,
        "sha256": hashlib.sha256(payload_raw).hexdigest(),
        "server_time_ms": server_ms,
        "captured_at": _iso_ms(server_ms),
    }
    if fetched_at is not None:
        exchange_info["fetched_at"] = fetched_at
    document = {
        "schema": 1,
        "exchange_info": exchange_info,
        "archive": {
            "file": Path(archive_path).name,
            "table": "raw_klines",
            "interval": "1h",
            "cutoff_ns": cutoff_ns,
            "cutoff": _iso_ms(cutoff_ms).removesuffix(".000Z") + "Z",
        },
        "derivation": DERIVATION,
        "assumptions": ASSUMPTIONS,
        "fee_source": FEE_SOURCE,
        "symbols": entries,
    }
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def _universe(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: rebuild ``research_instruments.json`` from a saved payload (no network)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD, 00:00 UTC, exclusive")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--fetched-at", default=None, help="Local fetch time, stored verbatim")
    args = parser.parse_args(argv)
    raw = build_snapshot(
        args.payload,
        args.archive,
        _universe(args.universe),
        _date_ns(args.cutoff),
        fetched_at=args.fetched_at,
    )
    args.out.write_bytes(raw)
    print(f"{args.out}: sha256 {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
