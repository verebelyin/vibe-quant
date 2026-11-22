"""Research universe: the top-N Binance USDT-M perpetuals by 30-day quote volume.

The list is frozen in ``universe_top20.txt`` (committed: a universe is not a
private result) until the user asks for a refresh. ``python -m vibe_quant.data
universe --top 20 [--as-of YYYY-MM-DD]`` re-ranks and rewrites it; the research
tier download (``download-research``) reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from vibe_quant.data.downloader import BINANCE_FUTURES_API, _call_with_retry

UNIVERSE_FILE = Path(__file__).with_name("universe_top20.txt")

WINDOW_DAYS = 30
_DAY_MS = 86_400_000

# The user's cap on the research universe (ruling 2026-11-19: disk is limited).
USER_CAP = 20

_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_EPOCH = date(1970, 1, 1)

RULE = (
    "Binance USDT-M contracts with contractType PERPETUAL, status TRADING and "
    "quoteAsset USDT (exchangeInfo at run time), ranked by the sum of quote "
    "volume over the 30 complete UTC days ending on the as-of day (1d klines, "
    "public REST). Excluded: underlyingType other than COIN (index/basket, "
    "pre-market), stablecoin-vs-stablecoin pairs, tokenised stock/commodity/FX "
    "perps (TRADIFI_PERPETUAL, PAXG, XAUT). Multiplier-prefixed contracts "
    "(1000X, 1MX, 1000000X) and X are one coin: volumes summed, listed as the "
    "bigger contract."
)

# Contract types whose volume is fetched so an excluded one can be shown in
# the ranking (tokenised stocks/commodities/FX trade as TRADIFI_PERPETUAL).
_POOL_CONTRACT_TYPES = frozenset({"PERPETUAL", "TRADIFI_PERPETUAL"})

STABLECOIN_BASES = frozenset(
    {
        "USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "PYUSD", "USD1",
        "RLUSD", "USDS", "EUR", "EURI", "AEUR", "U",
    }
)  # fmt: skip

# Tokenised real-world assets listed as underlyingType COIN (gold tokens).
TOKENISED_BASES = frozenset({"PAXG", "XAUT"})

_MULTIPLIER_PREFIX = re.compile(r"^(?:1000000|10000|1000|1M)(?=[A-Z])")


def honesty_note(as_of: date) -> str:
    """The selection-bias warning every result on this universe must repeat."""
    return (
        f"HONESTY NOTE: this list is chosen on volume in the {WINDOW_DAYS} days to "
        f"{as_of.isoformat()}, so backtests before {as_of.year} on it are "
        "selection-biased (coins that died or faded are absent; today's winners are "
        "present). Report results on the subset listed before 2021 separately and "
        "treat pre-selection history as optimistic."
    )


@dataclass(frozen=True)
class RankedCoin:
    """One coin in the ranking: its listed contract, summed volume, days seen."""

    symbol: str
    coin: str
    quote_volume: float
    days: int
    excluded: str | None = None


def coin_of(base_asset: str) -> str:
    """The coin behind a base asset: ``1000SHIB`` / ``1MBABYDOGE`` -> ``SHIB`` / ``BABYDOGE``."""
    return _MULTIPLIER_PREFIX.sub("", base_asset)


def exclusion_reason(contract: dict[str, Any]) -> str | None:
    """Why a pool contract is not in the universe, or None when it is eligible."""
    if contract.get("contractType") != "PERPETUAL":
        return f"tokenised stock/commodity/FX ({contract.get('underlyingType')})"
    underlying = contract.get("underlyingType")
    if underlying != "COIN":
        return f"index/basket or non-coin ({underlying})"
    base = str(contract.get("baseAsset", ""))
    if base in STABLECOIN_BASES:
        return "stablecoin vs stablecoin"
    if base in TOKENISED_BASES:
        return "tokenised commodity"
    return None


def candidate_pool(exchange_info: dict[str, Any]) -> list[dict[str, Any]]:
    """TRADING USDT-quoted perpetual-style contracts (eligible or excluded)."""
    return [
        s
        for s in exchange_info.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and s.get("contractType") in _POOL_CONTRACT_TYPES
    ]


def parse_as_of(text: str) -> date:
    """``--as-of``: exactly YYYY-MM-DD, with the 30-day window starting at or after 1970-01-01.

    Raises:
        argparse.ArgumentTypeError: another format, or a window before the epoch.
    """
    if not _ISO_DAY.fullmatch(text):
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}")
    try:
        day = date.fromisoformat(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r}: {e}") from e
    if day < _EPOCH + timedelta(days=WINDOW_DAYS - 1):
        raise argparse.ArgumentTypeError(
            f"{text}: its {WINDOW_DAYS}-day window would start before {_EPOCH.isoformat()}"
        )
    return day


def window_bounds(as_of: date) -> tuple[int, int]:
    """``[start, end]`` ms of the 30 complete UTC days ending on *as_of*."""
    end_excl = datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC) + timedelta(days=1)
    start = end_excl - timedelta(days=WINDOW_DAYS)
    return int(start.timestamp() * 1000), int(end_excl.timestamp() * 1000) - 1


def rank_universe(
    pool: list[dict[str, Any]],
    volumes: dict[str, tuple[float, int]],
    top: int,
) -> tuple[list[RankedCoin], list[RankedCoin]]:
    """Rank coins by summed window quote volume.

    Args:
        pool: Contracts from :func:`candidate_pool`.
        volumes: ``{symbol: (quote volume sum, days present)}``.
        top: Number of coins to keep.

    Returns:
        ``(selected, excluded)``: the top coins, and the excluded contracts that
        out-traded the last selected coin (shown so the exclusion is visible).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    excluded: list[RankedCoin] = []
    for c in pool:
        sym = c["symbol"]
        vol, days = volumes.get(sym, (0.0, 0))
        reason = exclusion_reason(c)
        if reason is not None:
            excluded.append(RankedCoin(sym, str(c.get("baseAsset")), vol, days, reason))
            continue
        groups.setdefault(coin_of(str(c["baseAsset"])), []).append(c)

    coins: list[RankedCoin] = []
    for coin, contracts in groups.items():
        listed = max(contracts, key=lambda c: (volumes.get(c["symbol"], (0.0, 0))[0], c["symbol"]))
        total = sum(volumes.get(c["symbol"], (0.0, 0))[0] for c in contracts)
        days = max(volumes.get(c["symbol"], (0.0, 0))[1] for c in contracts)
        coins.append(RankedCoin(listed["symbol"], coin, total, days))
    coins.sort(key=lambda r: (-r.quote_volume, r.symbol))
    selected = coins[:top]
    floor = selected[-1].quote_volume if selected else 0.0
    shown = sorted(
        (e for e in excluded if e.quote_volume >= floor), key=lambda r: (-r.quote_volume, r.symbol)
    )
    return selected, shown


def fetch_exchange_info(client: httpx.Client) -> dict[str, Any]:
    """``/fapi/v1/exchangeInfo`` with the shared retry policy."""

    def get() -> dict[str, Any]:
        response = client.get(f"{BINANCE_FUTURES_API}/fapi/v1/exchangeInfo")
        response.raise_for_status()
        return dict(response.json())

    return _call_with_retry(get, "exchangeInfo")


def fetch_window_volume(client: httpx.Client, symbol: str, as_of: date) -> tuple[float, int]:
    """Summed 1d quote volume and the number of days present in the window."""
    start, end = window_bounds(as_of)
    params: dict[str, str | int] = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start,
        "endTime": end,
        "limit": WINDOW_DAYS + 1,  # weight 1 below 100
    }

    def get() -> list[Any]:
        response = client.get(f"{BINANCE_FUTURES_API}/fapi/v1/klines", params=params)
        response.raise_for_status()
        return list(response.json())

    rows = [k for k in _call_with_retry(get, f"{symbol} 1d klines") if start <= int(k[0]) <= end]
    return sum(float(k[7]) for k in rows), len(rows)


def listing_dates(exchange_info: dict[str, Any]) -> dict[str, int]:
    """``{symbol: onboardDate ms}`` for every contract in exchangeInfo."""
    return {
        s["symbol"]: int(s["onboardDate"])
        for s in exchange_info.get("symbols", [])
        if s.get("onboardDate") is not None
    }


def render_universe(selected: list[RankedCoin], as_of: date) -> str:
    """The universe file text: rule/as-of/honesty header, one symbol per line."""
    start, end = window_bounds(as_of)
    first = datetime.fromtimestamp(start / 1000, UTC).date()
    header = [
        f"# vibe-quant research universe: top {len(selected)} Binance USDT-M perpetuals "
        f"by {WINDOW_DAYS}-day quote volume",
        f"# as-of: {as_of.isoformat()} (window {first.isoformat()}..{as_of.isoformat()} UTC)",
        f"# rule: {RULE}",
        f"# {honesty_note(as_of)}",
        "# Frozen until the user asks for a refresh: "
        "python -m vibe_quant.data universe --top N --as-of YYYY-MM-DD",
    ]
    return "\n".join([*header, *(c.symbol for c in selected)]) + "\n"


def write_universe(text: str, path: Path) -> str:
    """Write the universe file; return its SHA-256 hex digest."""
    data = text.encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a universe file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_universe(path: Path) -> tuple[list[str], date | None]:
    """Symbols (file order) and the as-of day of a universe file."""
    symbols: list[str] = []
    as_of: date | None = None
    for line in path.read_text().splitlines():
        if line.startswith("# as-of: "):
            as_of = date.fromisoformat(line.removeprefix("# as-of: ").split()[0])
        elif line.strip() and not line.startswith("#"):
            symbols.append(line.strip())
    return symbols, as_of


def build_universe(top: int, as_of: date) -> tuple[list[RankedCoin], list[RankedCoin]]:
    """Fetch exchangeInfo + the window's 1d klines and rank (see :data:`RULE`).

    Raises:
        DownloadError: a request failed after the shared retries.
    """
    with httpx.Client(timeout=30.0) as client:
        pool = candidate_pool(fetch_exchange_info(client))
        volumes = {c["symbol"]: fetch_window_volume(client, c["symbol"], as_of) for c in pool}
    return rank_universe(pool, volumes, top)


def format_ranking(selected: list[RankedCoin], excluded: list[RankedCoin]) -> list[str]:
    """Report lines: ranked coins with volume (and short windows), then exclusions."""
    lines = []
    for rank, c in enumerate(selected, 1):
        days = "" if c.days == WINDOW_DAYS else f"  ({c.days}/{WINDOW_DAYS} days)"
        lines.append(f"{rank:>3}. {c.symbol:<16} {c.quote_volume:>22,.0f} USDT{days}")
    for e in excluded:
        lines.append(f"     {e.symbol:<16} {e.quote_volume:>22,.0f} USDT  EXCLUDED: {e.excluded}")
    return lines
