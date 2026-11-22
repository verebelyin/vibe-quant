"""Top-N research universe: ranking + exclusion rule, frozen file, CLI input checks.

The rule (Binance USDT-M PERPETUAL, status TRADING, ranked by the summed quote
volume of the last 30 complete UTC days; stablecoin pairs, index/basket and
tokenised stock/commodity/FX perps excluded; 1000-prefixed contracts are one
coin) runs against a fixture exchangeInfo whose field values were copied from
the live ``/fapi/v1/exchangeInfo`` on 2026-11-22: ``contractType``
PERPETUAL / TRADIFI_PERPETUAL, ``underlyingType`` COIN / INDEX / EQUITY /
COMMODITY / PREMARKET, ``status`` TRADING / SETTLING, and the ``baseAsset`` of the
1000-prefixed contracts (``1000SHIB``, ``1MBABYDOGE``, ``1000000MOG``).

All HTTP goes through ``httpx.MockTransport``; the universe file lives in tmp.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.data import downloader, ingest, universe

if TYPE_CHECKING:
    from pathlib import Path

DAY_MS = 86_400_000
AS_OF = "2026-11-21"
WINDOW_START = datetime(2026, 10, 23, tzinfo=UTC)  # 30 days ending 2026-11-21


def _contract(
    symbol: str,
    base: str,
    *,
    contract_type: str = "PERPETUAL",
    status: str = "TRADING",
    quote: str = "USDT",
    underlying: str = "COIN",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "pair": symbol,
        "contractType": contract_type,
        "status": status,
        "baseAsset": base,
        "quoteAsset": quote,
        "marginAsset": quote,
        "underlyingType": underlying,
        "underlyingSubType": ["Crypto"],
        "onboardDate": 1_567_965_300_000,
    }


# 22 plain coins, daily quote volume = (30 - rank) * 1e6 so C01 > C02 > ... > C22.
PLAIN = [f"C{i:02d}" for i in range(1, 23)]
DAILY_VOLUME: dict[str, float] = {f"{c}USDT": (30 - i) * 1e6 for i, c in enumerate(PLAIN, 1)}
# Excluded contracts with raw volume inside the top 20.
DAILY_VOLUME |= {
    "USDCUSDT": 100e6,  # stablecoin vs stablecoin
    "AAPLUSDT": 90e6,  # tokenised stock (TRADIFI_PERPETUAL / EQUITY)
    "BTCDOMUSDT": 80e6,  # index
    "PAXGUSDT": 70e6,  # tokenised gold (underlyingType COIN)
    "XAUUSDT": 65e6,  # commodity (TRADIFI_PERPETUAL / COMMODITY)
    "OLDUSDT": 500e6,  # SETTLING: never requested, never counted
    "BTCUSDT_261225": 400e6,  # quarterly: never requested
    "BTCUSDC": 300e6,  # USDC-margined: never requested
}
# 1000-prefixed + plain contract of one coin: 11.6e6 + 11e6 per day -> 678e6 over
# 30 days, ranks between C07 (690e6) and C05 (27 days * 25e6 = 675e6) as ONE coin.
DAILY_VOLUME |= {"1000SHIBUSDT": 11.6e6, "SHIBUSDT": 11e6}

EXCHANGE_INFO: dict[str, Any] = {
    "timezone": "UTC",
    "symbols": [
        *[_contract(f"{c}USDT", c) for c in PLAIN],
        _contract("USDCUSDT", "USDC"),
        _contract("AAPLUSDT", "AAPL", contract_type="TRADIFI_PERPETUAL", underlying="EQUITY"),
        _contract("BTCDOMUSDT", "BTCDOM", underlying="INDEX"),
        _contract("PAXGUSDT", "PAXG"),
        _contract("XAUUSDT", "XAU", contract_type="TRADIFI_PERPETUAL", underlying="COMMODITY"),
        _contract("OLDUSDT", "OLD", status="SETTLING"),
        _contract("BTCUSDT_261225", "BTC", contract_type="CURRENT_QUARTER"),
        _contract("BTCUSDC", "BTC", quote="USDC"),
        _contract("1000SHIBUSDT", "1000SHIB"),
        _contract("SHIBUSDT", "SHIB"),
    ],
}

# Days missing from the 1d klines window, per symbol.
MISSING_DAYS = {"C05USDT": 3}


class FakeBinance:
    """exchangeInfo + 1d klines, with optional forced error statuses per symbol."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail: dict[str, list[int]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/fapi/v1/exchangeInfo":
            return httpx.Response(200, content=json.dumps(EXCHANGE_INFO).encode())
        if path == "/fapi/v1/klines":
            params = request.url.params
            symbol = params["symbol"]
            queue = self.fail.get(symbol)
            if queue:
                return httpx.Response(queue.pop(0) if len(queue) > 1 else queue[0])
            assert params["interval"] == "1d"
            start, end = int(params["startTime"]), int(params["endTime"])
            skip = MISSING_DAYS.get(symbol, 0)
            rows = []
            t = start
            n = 0
            while t <= end:
                n += 1
                if n > skip:
                    vol = DAILY_VOLUME[symbol]
                    rows.append(
                        [t, "1", "1", "1", "1", "1", t + DAY_MS - 1, str(vol), 1, "0", "0", "0"]
                    )
                t += DAY_MS
            return httpx.Response(200, content=json.dumps(rows).encode())
        return httpx.Response(404)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBinance:
    server = FakeBinance()
    real_client = httpx.Client

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(server.handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(downloader.time, "sleep", lambda _s: None)
    monkeypatch.setattr(universe, "UNIVERSE_FILE", tmp_path / "universe_top20.txt")
    return server


EXPECTED_TOP20 = [
    "C01USDT",
    "C02USDT",
    "C03USDT",
    "C04USDT",
    # C05 lost 3 of 30 days: 27 * 25e6 = 675e6 < C06's 30 * 24e6 = 720e6
    "C06USDT",
    "C07USDT",
    "1000SHIBUSDT",  # one coin: 1000SHIB + SHIB summed, listed as its bigger contract
    "C05USDT",  # 675e6 > C08's 660e6
    "C08USDT",
    "C09USDT",
    "C10USDT",
    "C11USDT",
    "C12USDT",
    "C13USDT",
    "C14USDT",
    "C15USDT",
    "C16USDT",
    "C17USDT",
    "C18USDT",
    "C19USDT",
]


def _symbols(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]


def test_rank_excludes_non_coin_contracts_and_next_coins_take_their_place(
    fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
    out = capsys.readouterr().out
    path = universe.UNIVERSE_FILE

    assert _symbols(path) == EXPECTED_TOP20
    # 22 plain coins + the SHIB coin compete for 20 slots (C20-C22 drop out);
    # the five excluded contracts out-trade them all yet take no slot.
    assert "C20USDT" not in _symbols(path)
    assert "SHIBUSDT" not in _symbols(path)
    for sym, reason in [
        ("USDCUSDT", "stablecoin"),
        ("AAPLUSDT", "tokenised"),
        ("BTCDOMUSDT", "index"),
        ("PAXGUSDT", "tokenised"),
        ("XAUUSDT", "tokenised"),
    ]:
        line = next(ln for ln in out.splitlines() if sym in ln)
        assert "EXCLUDED" in line and reason in line, line
    assert "C05USDT" in out and "27/30 days" in out
    # Contracts outside the candidate pool are never even requested.
    asked = {r.url.params.get("symbol") for r in fake.requests}
    assert {"OLDUSDT", "BTCUSDT_261225", "BTCUSDC"}.isdisjoint(asked)

    # The 30-day window ends at the as-of day's close.
    kl = next(r for r in fake.requests if r.url.params.get("symbol") == "C01USDT")
    assert int(kl.url.params["startTime"]) == int(WINDOW_START.timestamp() * 1000)
    assert (
        int(kl.url.params["endTime"])
        == int(datetime(2026, 11, 22, tzinfo=UTC).timestamp() * 1000) - 1
    )

    text = path.read_text()
    assert "as-of: 2026-11-21" in text
    assert "PERPETUAL" in text and "selection-biased" in text
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha in out
    assert "selection-biased" in out


def test_same_as_of_writes_byte_identical_file(fake: FakeBinance) -> None:
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
    first = universe.UNIVERSE_FILE.read_bytes()
    universe.UNIVERSE_FILE.unlink()
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
    assert universe.UNIVERSE_FILE.read_bytes() == first


@pytest.mark.parametrize("top", ["0", "-3", "abc", "2.5"])
def test_bad_top_is_argparse_error(fake: FakeBinance, top: str) -> None:
    with pytest.raises(SystemExit) as exc:
        ingest.main(["universe", "--top", top, "--as-of", AS_OF])
    assert exc.value.code == 2
    assert not universe.UNIVERSE_FILE.exists()
    assert fake.requests == []


def test_as_of_must_be_a_complete_utc_day(
    fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    today = datetime.now(UTC).date()
    for bad in [today, today + timedelta(days=3)]:
        assert ingest.main(["universe", "--top", "20", "--as-of", bad.isoformat()]) != 0
        assert "complete UTC day" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        ingest.main(["universe", "--top", "20", "--as-of", "2026-13-01"])
    assert exc.value.code == 2
    assert not universe.UNIVERSE_FILE.exists()
    assert fake.requests == []


def test_yesterday_is_the_default_as_of(fake: FakeBinance) -> None:
    assert ingest.main(["universe", "--top", "5"]) == 0
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    assert f"as-of: {yesterday.isoformat()}" in universe.UNIVERSE_FILE.read_text()
    assert len(_symbols(universe.UNIVERSE_FILE)) == 5


@pytest.mark.parametrize("status", [429, 503])
def test_rest_errors_retry_then_fail_without_writing(
    fake: FakeBinance, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    fake.fail["C03USDT"] = [status]  # every attempt fails
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 1
    assert not universe.UNIVERSE_FILE.exists()
    tries = [r for r in fake.requests if r.url.params.get("symbol") == "C03USDT"]
    assert len(tries) == len(downloader.MONTHLY_RETRY_DELAYS) + 1
    assert "C03USDT" in capsys.readouterr().err


def test_transient_error_recovers_within_retries(fake: FakeBinance) -> None:
    calls = {"n": 0}
    orig = fake.handler

    def once(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("symbol") == "C03USDT":
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503)
        return orig(request)

    fake.handler = once  # type: ignore[method-assign]
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
    assert _symbols(universe.UNIVERSE_FILE) == EXPECTED_TOP20


def test_read_universe_returns_symbols_and_as_of(fake: FakeBinance) -> None:
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
    symbols, as_of = universe.read_universe(universe.UNIVERSE_FILE)
    assert symbols == EXPECTED_TOP20
    assert as_of == date(2026, 11, 21)


@pytest.mark.parametrize(
    ("base", "coin"),
    [
        ("1000SHIB", "SHIB"),
        ("1MBABYDOGE", "BABYDOGE"),
        ("1000000MOG", "MOG"),
        ("1INCH", "1INCH"),  # a real coin name starting with 1, not a multiplier
        ("BTC", "BTC"),
    ],
)
def test_multiplier_prefix_maps_to_one_coin(base: str, coin: str) -> None:
    assert universe.coin_of(base) == coin


def test_top_above_the_users_cap_of_20_warns_but_writes(
    fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
    assert "cap" not in capsys.readouterr().err
    universe.UNIVERSE_FILE.unlink()
    assert ingest.main(["universe", "--top", "21", "--as-of", AS_OF]) == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "--top 21" in err and "cap of 20" in err
    assert len(_symbols(universe.UNIVERSE_FILE)) == 21


@pytest.mark.parametrize(
    ("as_of", "ok"),
    [
        ("20250101", False),  # compact ISO: not YYYY-MM-DD
        ("2025-W01-1", False),  # ISO week date
        ("2025-1-01", False),
        ("0001-01-01", False),  # the 30-day window underflows
        ("1970-01-29", False),  # window would start 1969-12-31 (negative epoch ms)
        ("1970-01-30", True),  # window starts at epoch 0
    ],
)
def test_as_of_must_be_yyyy_mm_dd_and_bound_a_window(
    fake: FakeBinance, as_of: str, ok: bool
) -> None:
    if ok:
        assert ingest.main(["universe", "--top", "3", "--as-of", as_of]) == 0
        assert f"as-of: {as_of}" in universe.UNIVERSE_FILE.read_text()
        return
    with pytest.raises(SystemExit) as exc:
        ingest.main(["universe", "--top", "20", "--as-of", as_of])
    assert exc.value.code == 2
    assert not universe.UNIVERSE_FILE.exists()
    assert fake.requests == []


def test_same_as_of_and_top_keeps_the_frozen_file_without_http(
    fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
    first = universe.UNIVERSE_FILE.read_bytes()
    capsys.readouterr()
    # exchangeInfo membership changes later (the winner is delisted): a rerun for
    # the same as-of must not re-rank on today's membership.
    EXCHANGE_INFO["symbols"][0]["status"] = "SETTLING"
    try:
        fake.requests.clear()
        assert ingest.main(["universe", "--top", "20", "--as-of", AS_OF]) == 0
        assert fake.requests == []
        assert universe.UNIVERSE_FILE.read_bytes() == first
        out = capsys.readouterr().out
        assert "kept" in out and hashlib.sha256(first).hexdigest() in out
        assert "selection-biased" in out

        # a different N or as-of is a new universe: ranked again
        assert ingest.main(["universe", "--top", "19", "--as-of", AS_OF]) == 0
        assert fake.requests != []
        assert "C01USDT" not in _symbols(universe.UNIVERSE_FILE)
    finally:
        EXCHANGE_INFO["symbols"][0]["status"] = "TRADING"
