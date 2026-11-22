"""Frozen research instrument snapshot (arch-c9-2 R45, R46, R50; dreview C9-14).

The 1h tier takes every instrument from the committed ``research_instruments.json``,
built OFFLINE by :func:`build_snapshot` from a saved exchangeInfo payload and the
archive's 1h closes before a cutoff. The snapshot records its own provenance (payload
SHA-256 and capture time, archive cutoff, derivation rule, effective filters), so the
bytes -- and the fingerprint a result carries -- depend only on those inputs.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from tests.unit.test_carry_bars import kline, ms, ns, write_archive, write_payload
from vibe_quant.fees import FeeSchedule
from vibe_quant.portfolio import instruments as instruments_module
from vibe_quant.portfolio.instruments import (
    SNAPSHOT_PATH,
    build_snapshot,
    load_snapshot,
    research_instrument,
)
from vibe_quant.portfolio.sizing import InstrumentSpec
from vibe_quant.portfolio.spec import CarryDataError

if TYPE_CHECKING:
    from pathlib import Path

CUTOFF = ns("2026-11-22T00:00:00")
_BEFORE = ms("2026-11-20T00:00:00")
_AFTER = ms("2026-11-22T00:00:00")
_PAYLOAD_SHA = "cbfdbeedaf0d7b3de75bcf2f8d3d0a8d2cf2ed68547ce379b49c396c2307c27a"


def _build(
    tmp_path: Path, filters: dict[str, dict[str, str]], closes: dict[str, list[tuple[int, float]]]
) -> bytes:
    payload = write_payload(tmp_path / "payload.json", filters)
    archive = write_archive(
        tmp_path / "a.db",
        {
            symbol: [kline(open_ms, close) for open_ms, close in rows]
            for symbol, rows in closes.items()
        },
    )
    return build_snapshot(payload, archive, sorted(filters), CUTOFF)


def test_builder_derives_the_finest_grid_from_archived_closes(tmp_path: Path) -> None:
    """R46: tickSize 0.001 with an archived close 0.6175 before cutoff -> increment 0.0001."""
    raw = _build(
        tmp_path,
        {
            "ADAUSDT": {"tickSize": "0.001", "stepSize": "1", "minQty": "1"},
            "XRPUSDT": {"tickSize": "0.0001"},
        },
        {
            "ADAUSDT": [(_BEFORE, 0.6175), (_BEFORE + 3_600_000, 0.617)],
            "XRPUSDT": [(_BEFORE, 2.1234)],
        },
    )
    data = json.loads(raw)
    ada = data["symbols"]["ADAUSDT"]
    assert (ada["tick_size"], ada["price_increment"], ada["price_precision"]) == (
        "0.001",
        "0.0001",
        4,
    )
    assert (ada["size_increment"], ada["size_precision"], ada["min_quantity"]) == ("1", 0, "1")
    assert ada["archive_1h_closes"] == 2
    # every close on the tick grid keeps the tick
    xrp = data["symbols"]["XRPUSDT"]
    assert (xrp["price_increment"], xrp["price_precision"]) == ("0.0001", 4)
    assert (xrp["size_increment"], xrp["size_precision"]) == ("0.001", 3)


def test_snapshot_records_its_provenance(tmp_path: Path) -> None:
    """C9-14: payload SHA + capture time, archive cutoff, derivation rule, effective filters."""
    raw = _build(
        tmp_path,
        {"BTCUSDT": {"tickSize": "0.10", "notional": "50"}},
        {"BTCUSDT": [(_BEFORE, 100.1)]},
    )
    data = json.loads(raw)
    payload_sha = hashlib.sha256((tmp_path / "payload.json").read_bytes()).hexdigest()
    assert data["exchange_info"] == {
        "file": "payload.json",
        "sha256": payload_sha,
        "server_time_ms": 1_795_245_209_074,
        "captured_at": "2026-11-21T07:13:29.074Z",
    }
    assert data["archive"] == {
        "file": "a.db",
        "table": "raw_klines",
        "interval": "1h",
        "cutoff_ns": CUTOFF,
        "cutoff": "2026-11-22T00:00:00Z",
    }
    assert "tickSize" in data["derivation"] and "cutoff" in data["derivation"]
    assert "today's filters" in data["assumptions"]
    btc = data["symbols"]["BTCUSDT"]
    assert btc["min_notional"] == "50"
    assert (btc["maker_fee"], btc["taker_fee"]) == ("0.0002", "0.0005")
    assert (btc["margin_init"], btc["margin_maint"]) == ("0.05", "0.025")
    assert (btc["tick_size"], btc["price_increment"], btc["price_precision"]) == ("0.10", "0.1", 1)


def test_snapshot_bytes_ignore_rows_after_cutoff(tmp_path: Path) -> None:
    """R46 + C9-14: same inputs twice -> byte-identical; a finer close appended at/after
    the cutoff leaves the bytes (and so the fingerprint) unchanged."""
    filters = {"BTCUSDT": {"tickSize": "0.10"}}
    payload = write_payload(tmp_path / "payload.json", filters)
    archive = write_archive(tmp_path / "a.db", {"BTCUSDT": [kline(_BEFORE, 100.1)]})
    first = build_snapshot(payload, archive, ["BTCUSDT"], CUTOFF)
    assert build_snapshot(payload, archive, ["BTCUSDT"], CUTOFF) == first
    write_archive(archive, {"BTCUSDT": [kline(_AFTER, 100.123)]})
    assert build_snapshot(payload, archive, ["BTCUSDT"], CUTOFF) == first
    # control: the same finer close one hour BEFORE the cutoff moves the grid
    write_archive(archive, {"BTCUSDT": [kline(_AFTER - 3_600_000, 100.123)]})
    moved = json.loads(build_snapshot(payload, archive, ["BTCUSDT"], CUTOFF))
    assert moved["symbols"]["BTCUSDT"]["price_increment"] == "0.001"


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"tickSize": "0"}, "tickSize"),
        ({"tickSize": "nan"}, "tickSize"),
        ({}, "tickSize"),
        ({"tickSize": "0.1", "stepSize": "-1"}, "stepSize"),
        ({"tickSize": "0.1", "stepSize": "0.01", "minQty": "0.015"}, "minQty"),
        ({"tickSize": "0.1", "notional": "inf"}, "notional"),
    ],
)
def test_builder_rejects_bad_filters(tmp_path: Path, values: dict[str, str], field: str) -> None:
    """R46: a zero/nan/absent filter or minQty off the step grid names symbol and field."""
    with pytest.raises(ValueError, match=rf"^XUSDT: .*{field}"):
        _build(tmp_path, {"XUSDT": values}, {})


def test_builder_rejects_a_symbol_missing_from_the_payload(tmp_path: Path) -> None:
    payload = write_payload(tmp_path / "payload.json", {"BTCUSDT": {"tickSize": "0.1"}})
    archive = write_archive(tmp_path / "a.db", {})
    with pytest.raises(ValueError, match=r"^ETHUSDT: not in exchangeInfo payload"):
        build_snapshot(payload, archive, ["BTCUSDT", "ETHUSDT"], CUTOFF)


def test_builder_and_loader_import_no_network() -> None:
    """R46: the module has no fetch path (no HTTP client, socket or universe fetcher import)."""
    tree = ast.parse(inspect.getsource(instruments_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = {
        "requests",
        "httpx",
        "aiohttp",
        "socket",
        "urllib.request",
        "vibe_quant.data.universe",
    }
    assert imported & banned == set()
    assert not hasattr(instruments_module, "fetch_exchange_info")


def test_missing_symbol_names_the_snapshot_sha(tmp_path: Path) -> None:
    """R45: a 1h symbol absent from the snapshot fails closed with the snapshot's SHA."""
    out = tmp_path / "research_instruments.json"
    out.write_bytes(_build(tmp_path, {"BTCUSDT": {"tickSize": "0.10"}}, {}))
    snapshot = load_snapshot(out)
    assert snapshot.sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(CarryDataError) as error:
        research_instrument(snapshot, "NOPEUSDT")
    assert str(error.value) == f"NOPEUSDT: not in research_instruments.json (sha {snapshot.sha256})"


def test_snapshot_instrument_matches_its_entry(tmp_path: Path) -> None:
    """R50: the instrument carries the snapshot filters and the VIP0 taker fee."""
    out = tmp_path / "research_instruments.json"
    out.write_bytes(
        _build(
            tmp_path,
            {"ADAUSDT": {"tickSize": "0.001", "stepSize": "1", "minQty": "1", "notional": "5"}},
            {"ADAUSDT": [(_BEFORE, 0.6175)]},
        )
    )
    instrument = research_instrument(load_snapshot(out), "ADAUSDT")
    spec = InstrumentSpec.from_instrument(instrument)
    assert spec == InstrumentSpec("ADAUSDT", Decimal("1"), Decimal("1"), 5.0, Decimal("0.0005"))
    assert spec.taker_fee == FeeSchedule().taker_rate
    assert instrument.maker_fee == FeeSchedule().maker_rate
    assert instrument.price_increment.as_decimal() == Decimal("0.0001")
    assert float(instrument.make_price(0.6175)) == 0.6175
    assert str(instrument.id) == "ADAUSDT-PERP.BINANCE"


def test_committed_snapshot_covers_the_universe() -> None:
    """The committed file was built from the chief's saved payload for the frozen universe."""
    snapshot = load_snapshot()
    assert snapshot.path == SNAPSHOT_PATH
    data = json.loads(SNAPSHOT_PATH.read_bytes())
    assert data["exchange_info"]["sha256"] == _PAYLOAD_SHA
    universe = SNAPSHOT_PATH.parents[1] / "data" / "universe_top20.txt"
    symbols = [
        line.strip()
        for line in universe.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert sorted(data["symbols"]) == sorted(symbols)
    for symbol in symbols:
        instrument = research_instrument(snapshot, symbol)
        assert instrument.taker_fee == FeeSchedule().taker_rate
