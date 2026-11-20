"""Venue leverage/margin config: classify, build the read, publish to the NT account.

Contract rows V1-V10, V8b, V9b, V-MAP and V-PUB of design 68m53 (task T1). The paper
node reads Binance ``GET /fapi/v1/symbolConfig`` and compares each session symbol's
leverage and margin type with the resolved run config. A confirmed difference is a
mismatch; anything the module cannot read with certainty is "unverified" (fail closed,
never "match", never a mismatch). A pass then publishes the venue leverage onto the
exact NT ``MarginAccount`` the trading guard reads.

Every test drives a fake ``sign_request``; no network is touched.
"""

from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from typing import Any

import pytest
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
from nautilus_trader.core.nautilus_pyo3 import HttpMethod
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.test_kit.stubs.execution import TestExecStubs

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.paper.venue_config import (
    POSITION_RISK_PATH,
    SYMBOL_CONFIG_PATH,
    ConfigVerdict,
    VenuePosition,
    bounded_alert_detail,
    build_position_risk_read,
    build_publish_map,
    build_symbol_config_read,
    classify_symbol_configs,
    parse_positions,
    publish_leverage,
)

_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def _row(symbol: str, leverage: object = 10, margin: object = "ISOLATED") -> dict[str, object]:
    """A symbolConfig row shaped like the Binance USD-M answer."""
    return {
        "symbol": symbol,
        "marginType": margin,
        "isAutoAddMargin": "false",
        "leverage": leverage,
        "maxNotionalValue": "1000000",
    }


def _classify(body: object, symbols: list[str] | None = None) -> ConfigVerdict:
    return classify_symbol_configs(body, symbols or _SYMBOLS, 10, "ISOLATED")


# ----------------------------------------------------------- V1-V6 matching/mismatch


def test_matching_config_has_no_mismatch() -> None:
    """V1: the session's rows match among 300 unrelated rows."""
    others = [_row(f"X{i}USDT", leverage=i % 125, margin="CROSSED") for i in range(300)]
    verdict = _classify([*others[:150], _row("BTCUSDT"), _row("ETHUSDT"), *others[150:]])

    assert verdict.observed == {"BTCUSDT": (10, "ISOLATED"), "ETHUSDT": (10, "ISOLATED")}
    assert verdict.mismatches == {}
    assert verdict.unverified == {}


def test_mismatch_lists_only_bad_symbol() -> None:
    """V2: only ETH differs, so only ETH is listed."""
    verdict = _classify([_row("BTCUSDT"), _row("ETHUSDT", leverage=5)])

    assert verdict.mismatches == {"ETHUSDT": "ETHUSDT: venue leverage 5x != resolved 10x"}
    assert verdict.unverified == {}
    assert verdict.observed["ETHUSDT"] == (5, "ISOLATED")


def test_margin_mismatch() -> None:
    """V3."""
    verdict = _classify([_row("BTCUSDT"), _row("ETHUSDT", margin="CROSSED")])

    assert verdict.mismatches == {"ETHUSDT": "ETHUSDT: venue margin CROSSED != resolved ISOLATED"}
    assert verdict.unverified == {}


def test_both_fields_one_string() -> None:
    """V4: both fields of one symbol differ -> ONE string joined by "; "."""
    verdict = _classify([_row("BTCUSDT"), _row("ETHUSDT", leverage=5, margin="CROSSED")])

    assert verdict.mismatches == {
        "ETHUSDT": (
            "ETHUSDT: venue leverage 5x != resolved 10x; venue margin CROSSED != resolved ISOLATED"
        )
    }


def test_missing_symbol_is_unverified() -> None:
    """V5: no ETH row is unverified, never a mismatch."""
    verdict = _classify([_row("BTCUSDT")])

    assert verdict.unverified == {"ETHUSDT": "missing from symbolConfig"}
    assert verdict.mismatches == {}
    assert "ETHUSDT" not in verdict.observed


def test_zero_leverage_is_mismatch() -> None:
    """V5b: leverage 0 is a valid reading that differs (edge of 0..125)."""
    verdict = _classify([_row("BTCUSDT"), _row("ETHUSDT", leverage=0)])

    assert verdict.mismatches == {"ETHUSDT": "ETHUSDT: venue leverage 0x != resolved 10x"}
    assert verdict.unverified == {}


@pytest.mark.parametrize(
    ("leverage", "margin", "observed"),
    [
        ("10", "isolated", (10, "ISOLATED")),
        (10, "Isolated", (10, "ISOLATED")),
        ("125", "ISOLATED", (125, "ISOLATED")),
        (125, "ISOLATED", (125, "ISOLATED")),
        ("0", "cross", (0, "CROSSED")),
        (1, "crossed", (1, "CROSSED")),
    ],
)
def test_digit_string_and_margin_case(
    leverage: object, margin: object, observed: tuple[int, str]
) -> None:
    """V6: ASCII digit strings parse; margin case-folds, CROSS -> CROSSED."""
    verdict = classify_symbol_configs(
        [_row("BTCUSDT", leverage=leverage, margin=margin)], ["BTCUSDT"], observed[0], observed[1]
    )

    assert verdict.observed == {"BTCUSDT": observed}
    assert verdict.mismatches == {}
    assert verdict.unverified == {}


# --------------------------------------------------------------- V7/V8 bad fields

_MISSING = object()


def _param_id(value: object) -> str:
    if value is _MISSING:
        return "missing"
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 10**12:
        return f"int:{len(str(abs(value))) if abs(value) < 10**4000 else 'huge'}digits"
    return f"{type(value).__name__}:{value!r}"[:24]


@pytest.mark.parametrize(
    "leverage",
    [
        10.0,
        "10.0",
        "10.5",
        True,
        False,
        None,
        "x",
        "",
        " 10",
        "10\n",
        "+10",
        "0010",
        -1,
        "-1",
        126,
        "126",
        10**30,
        10**400,
        10**5000,
        "１０",
        float("inf"),
        float("nan"),
        "NaN",
        "Infinity",
        [10],
        {"v": 10},
        _MISSING,
    ],
    ids=lambda v: _param_id(v),
)
def test_bad_leverage_is_unverified(leverage: object) -> None:
    """V7: anything but an int or a 1-3 ASCII-digit string in 0..125 is unverified
    (no mismatch, no observed leverage); the other symbol is still classified."""
    eth = _row("ETHUSDT")
    if leverage is _MISSING:
        del eth["leverage"]
        shown = "None"
    else:
        eth["leverage"] = leverage
        shown = None
    verdict = _classify([_row("BTCUSDT", leverage=5), eth])

    reason = verdict.unverified["ETHUSDT"]
    assert reason.startswith("leverage is not a whole number in 0..125: ")
    if shown is not None:
        assert reason == f"leverage is not a whole number in 0..125: {shown}"
    elif not (isinstance(leverage, int) and abs(leverage) > 10**100):
        assert reason == f"leverage is not a whole number in 0..125: {leverage!r}"
    assert "ETHUSDT" not in verdict.mismatches
    assert verdict.observed["ETHUSDT"] == (None, "ISOLATED")
    assert verdict.mismatches == {"BTCUSDT": "BTCUSDT: venue leverage 5x != resolved 10x"}


@pytest.mark.parametrize(
    "margin",
    [None, 1, True, ["ISOLATED"], "", "UNKNOWN", "PORTFOLIO", "ISOLATED ", "ıSOLATED", _MISSING],
    ids=lambda v: _param_id(v),
)
def test_bad_margin_type_is_unverified(margin: object) -> None:
    """V8 (R12): only ISOLATED / CROSSED (any ASCII case, CROSS alias) is a reading;
    anything else is unverified, never a mismatch, even with a matching leverage."""
    eth = _row("ETHUSDT")
    if margin is _MISSING:
        del eth["marginType"]
        shown = "None"
    else:
        eth["marginType"] = margin
        shown = repr(margin)
    verdict = _classify([_row("BTCUSDT"), eth])

    assert verdict.unverified == {"ETHUSDT": f"marginType is not ISOLATED/CROSSED: {shown}"}
    assert verdict.mismatches == {}
    assert verdict.observed["ETHUSDT"] == (10, None)


@pytest.mark.parametrize("case", ["lev", "margin"])
def test_valid_field_mismatches_beside_bad_field(case: str) -> None:
    """V8b (R12): each field is classified on its own; a valid differing field is a
    mismatch even when its sibling field is unverified; both reasons are reported."""
    if case == "lev":
        btc = _row("BTCUSDT", leverage=5, margin=None)
        mismatch = "BTCUSDT: venue leverage 5x != resolved 10x"
        unverified = "marginType is not ISOLATED/CROSSED: None"
        observed: tuple[int | None, str | None] = (5, None)
    else:
        btc = _row("BTCUSDT", leverage="x", margin="CROSSED")
        mismatch = "BTCUSDT: venue margin CROSSED != resolved ISOLATED"
        unverified = "leverage is not a whole number in 0..125: 'x'"
        observed = (None, "CROSSED")
    verdict = _classify([btc, _row("ETHUSDT")])

    assert verdict.mismatches == {"BTCUSDT": mismatch}
    assert verdict.unverified == {"BTCUSDT": unverified}
    assert verdict.observed["BTCUSDT"] == observed
    assert verdict.observed["ETHUSDT"] == (10, "ISOLATED")


def test_both_fields_bad_reports_both_reasons() -> None:
    """Both fields unreadable -> one unverified string carrying both reasons."""
    verdict = _classify([_row("BTCUSDT"), _row("ETHUSDT", leverage=None, margin="")])

    assert verdict.unverified == {
        "ETHUSDT": (
            "leverage is not a whole number in 0..125: None; marginType is not ISOLATED/CROSSED: ''"
        )
    }
    assert verdict.mismatches == {}


# ---------------------------------------------------------------- V9/V9b bodies


def test_malformed_bodies() -> None:
    """V9: a non-list body, a non-object row, duplicate rows and junk rows."""
    as_dict = _classify({"code": -1121, "msg": "x"})
    assert as_dict.unverified == {
        "BTCUSDT": "answer is not a list: dict",
        "ETHUSDT": "answer is not a list: dict",
    }
    assert as_dict.mismatches == {}
    assert as_dict.observed == {}

    for body in (None, "[]", 5):
        verdict = _classify(body)
        assert set(verdict.unverified) == set(_SYMBOLS)
        assert verdict.mismatches == {}

    junk = [
        "BTCUSDT",
        7,
        None,
        ["ETHUSDT", 5],
        {"symbol": 5, "leverage": 5, "marginType": "CROSSED"},
        {"leverage": 5},
        {"symbol": "SOLUSDT", "leverage": "garbage"},
    ]
    ignored = _classify([*junk, _row("BTCUSDT"), _row("ETHUSDT")])
    assert ignored.unverified == {}
    assert ignored.mismatches == {}
    assert ignored.observed == {"BTCUSDT": (10, "ISOLATED"), "ETHUSDT": (10, "ISOLATED")}

    dup = _classify([_row("BTCUSDT"), _row("ETHUSDT"), _row("ETHUSDT", leverage=5)])
    assert dup.unverified == {"ETHUSDT": "2 rows"}
    assert dup.mismatches == {}, "a duplicate is unverified on both fields, never a mismatch"
    assert "ETHUSDT" not in dup.observed

    three = _classify([_row("BTCUSDT")] * 3 + [_row("ETHUSDT")])
    assert three.unverified == {"BTCUSDT": "3 rows"}


@pytest.mark.parametrize("order", ["btc_first", "eth_first"])
def test_mismatch_survives_bad_sibling_row(order: str) -> None:
    """V9b: a confirmed BTC mismatch is reported next to an unreadable ETH row, in
    either row order."""
    rows = [_row("BTCUSDT", leverage=5), _row("ETHUSDT", leverage=None)]
    if order == "eth_first":
        rows.reverse()
    verdict = _classify(rows)

    assert verdict.mismatches == {"BTCUSDT": "BTCUSDT: venue leverage 5x != resolved 10x"}
    assert verdict.unverified == {"ETHUSDT": "leverage is not a whole number in 0..125: None"}


def test_resolved_values_are_forwarded() -> None:
    """The resolved leverage/margin are the comparison's other side (two unrelated sets)."""
    rows = [_row("BTCUSDT", leverage=20, margin="CROSSED")]
    same = classify_symbol_configs(rows, ["BTCUSDT"], 20, "CROSSED")
    other = classify_symbol_configs(rows, ["BTCUSDT"], 3, "ISOLATED")

    assert same.mismatches == {}
    assert other.mismatches == {
        "BTCUSDT": (
            "BTCUSDT: venue leverage 20x != resolved 3x; venue margin CROSSED != resolved ISOLATED"
        )
    }


@pytest.mark.parametrize(("venue", "resolved"), [(9, 10), (11, 10), (10, 9), (10, 11)])
def test_leverage_comparison_edges(venue: int, resolved: int) -> None:
    """One either side of equality is a mismatch; equality is not (V1)."""
    rows = [_row("BTCUSDT", leverage=venue)]
    verdict = classify_symbol_configs(rows, ["BTCUSDT"], resolved, "ISOLATED")

    assert verdict.mismatches == {
        "BTCUSDT": f"BTCUSDT: venue leverage {venue}x != resolved {resolved}x"
    }


# ---------------------------------------------------------------- V10 builder


class _FakeClient:
    """NT's ``BinanceHttpClient`` stand-in: records the signed call, replays bytes."""

    def __init__(self, body: bytes, record: dict[str, Any]) -> None:
        self._body = body
        self._record = record

    async def sign_request(
        self,
        http_method: object,
        url_path: str,
        payload: dict[str, str] | None = None,
        ratelimiter_keys: list[str] | None = None,
    ) -> bytes:
        self._record.setdefault("calls", []).append(
            (http_method, url_path, dict(payload or {}), ratelimiter_keys)
        )
        return self._body


def _patch_factory(monkeypatch: pytest.MonkeyPatch, body: bytes) -> dict[str, Any]:
    record: dict[str, Any] = {}

    def fake_factory(
        *, clock: Any, account_type: Any, api_key: Any, api_secret: Any, environment: Any
    ) -> Any:
        record.update(
            clock=clock,
            account_type=account_type,
            api_key=api_key,
            api_secret=api_secret,
            environment=environment,
        )
        return _FakeClient(body, record)

    monkeypatch.setattr(
        "nautilus_trader.adapters.binance.factories.get_cached_binance_http_client",
        fake_factory,
    )
    return record


@pytest.mark.parametrize(
    ("testnet", "env", "key", "secret"),
    [
        (True, BinanceEnvironment.TESTNET, "key-a", "secret-a"),
        (False, BinanceEnvironment.LIVE, "key-b", "secret-b"),
    ],
)
async def test_builder_decodes_bytes_and_envs(
    monkeypatch: pytest.MonkeyPatch,
    testnet: bool,
    env: BinanceEnvironment,
    key: str,
    secret: str,
) -> None:
    """V10: one signed GET symbolConfig {recvWindow, timestamp} with the session's
    key/secret per call; bytes decoded as JSON; testnet picks the environment."""
    rows = [_row("BTCUSDT"), _row("ETHUSDT", leverage=5)]
    record = _patch_factory(monkeypatch, json.dumps(rows).encode())
    read = build_symbol_config_read(key, secret, "USDT_FUTURES", testnet)

    before = int(time.time() * 1000)
    first = await read()
    second = await read()
    after = int(time.time() * 1000)

    assert first == rows
    assert second == rows
    assert record["account_type"] is BinanceAccountType.USDT_FUTURES
    assert record["environment"] is env
    assert (record["api_key"], record["api_secret"]) == (key, secret)
    assert len(record["calls"]) == 2, "one signed request per read"
    for method, path, payload, _keys in record["calls"]:
        assert method is HttpMethod.GET
        assert path == SYMBOL_CONFIG_PATH == "/fapi/v1/symbolConfig"
        assert set(payload) == {"recvWindow", "timestamp"}
        assert payload["recvWindow"] == "5000"
        assert payload["timestamp"].isdigit()
        assert before <= int(payload["timestamp"]) <= after


async def test_builder_raises_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """V10: a body that is not JSON raises from the read (a failed read upstream)."""
    _patch_factory(monkeypatch, b"{oops")
    read = build_symbol_config_read("k", "s", "USDT_FUTURES", True)

    with pytest.raises(json.JSONDecodeError):
        await read()


async def test_builder_returns_non_list_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """The read decodes only; the classifier judges the shape (V9)."""
    _patch_factory(monkeypatch, b'{"code": -2015, "msg": "Invalid API-key"}')
    read = build_symbol_config_read("k", "s", "USDT_FUTURES", False)

    assert await read() == {"code": -2015, "msg": "Invalid API-key"}


@pytest.mark.parametrize("account_type", ["COIN_FUTURES", "SPOT", "MARGIN", "usdt_futures", ""])
def test_builder_refuses_other_account_types(
    monkeypatch: pytest.MonkeyPatch, account_type: str
) -> None:
    """V10/F10: USD-M only; anything else raises ValueError at build, before a client."""
    record = _patch_factory(monkeypatch, b"[]")

    with pytest.raises(ValueError, match="USDT_FUTURES"):
        build_symbol_config_read("k", "s", account_type, True)
    assert record == {}, "no HTTP client is built for a refused account type"


# ----------------------------------------------------------------- V-MAP


def _compiled_instrument_id(symbol: str) -> object:
    """The ``instrument_id`` value the real DSL compiler puts into a strategy config."""
    from tests.unit.test_audit_paper_node import DSL
    from vibe_quant.paper.node import PaperTradingNode

    dsl = validate_strategy_dict(DSL)
    module = StrategyCompiler().compile_to_module(dsl)
    _strategy_cls, config_cls = PaperTradingNode._class_names(dsl)
    # Decoded exactly like PaperTradingNode._instantiate_strategies does.
    config = getattr(module, config_cls).parse(
        json.dumps({"instrument_id": f"{symbol}-PERP.BINANCE"})
    )
    instrument_id: object = config.instrument_id
    return instrument_id


def test_publish_map_from_compiled_string_configs() -> None:
    """V-MAP (R1): the compiler's string ids become typed ``InstrumentId``s."""
    configured = {sym: _compiled_instrument_id(sym) for sym in _SYMBOLS}
    assert all(isinstance(v, str) for v in configured.values()), "production config is a str"

    mapped = build_publish_map(_SYMBOLS, configured, "BINANCE")

    assert mapped == {
        "BTCUSDT": InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        "ETHUSDT": InstrumentId.from_str("ETHUSDT-PERP.BINANCE"),
    }
    assert isinstance(mapped, dict)
    assert all(type(v) is InstrumentId for v in mapped.values())


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (
            {"BTCUSDT": "BTCUSDT-PERP.BINANCE"},
            {"BTCUSDT": InstrumentId.from_str("BTCUSDT-PERP.BINANCE")},
        ),
        (
            {"BTCUSDT": InstrumentId.from_str("BTCUSDT-PERP.BINANCE")},
            {"BTCUSDT": InstrumentId.from_str("BTCUSDT-PERP.BINANCE")},
        ),
        (
            {"BTCUSDT": "ETHUSDT-PERP.BINANCE"},
            "BTCUSDT: no valid instrument id (ETHUSDT-PERP.BINANCE is not a BTCUSDT contract)",
        ),
        (
            {"BTCUSDT": "BTCUSDT-PERP.BYBIT"},
            "BTCUSDT: no valid instrument id (BTCUSDT-PERP.BYBIT is not on BINANCE)",
        ),
        (
            {"BTCUSDT": "garbage"},
            "BTCUSDT: no valid instrument id (unparsable 'garbage')",
        ),
        (
            {"ETHUSDT": "ETHUSDT-PERP.BINANCE"},
            "BTCUSDT: no valid instrument id (not configured)",
        ),
        (
            {"BTCUSDT": None},
            "BTCUSDT: no valid instrument id (not configured)",
        ),
        (
            {"BTCUSDT": "BTCUSDTX-PERP.BINANCE"},
            "BTCUSDT: no valid instrument id (BTCUSDTX-PERP.BINANCE is not a BTCUSDT contract)",
        ),
    ],
    ids=["str", "typed", "other_symbol", "other_venue", "garbage", "missing", "none", "prefix"],
)
def test_publish_map(configured: dict[str, object], expected: object) -> None:
    """V-MAP (R1): string or typed input; symbol and venue must agree; one entry each."""
    assert build_publish_map(["BTCUSDT"], configured, "BINANCE") == expected


def test_publish_map_venue_is_forwarded() -> None:
    """The venue argument is the comparison's other side (two unrelated venues)."""
    configured = {"BTCUSDT": "BTCUSDT-PERP.BYBIT"}
    assert build_publish_map(["BTCUSDT"], configured, "BYBIT") == {
        "BTCUSDT": InstrumentId.from_str("BTCUSDT-PERP.BYBIT")
    }


def test_publish_map_reports_every_bad_symbol() -> None:
    """Two bad symbols: both named; a good one never hides a bad one."""
    out = build_publish_map(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        {"BTCUSDT": "BTCUSDT-PERP.BINANCE", "SOLUSDT": "garbage"},
        "BINANCE",
    )
    assert out == (
        "ETHUSDT: no valid instrument id (not configured); "
        "SOLUSDT: no valid instrument id (unparsable 'garbage')"
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("x" * 78, f"(unparsable '{'x' * 78}')"),
        ("x" * 79, f"(unparsable '{'x' * 79}...(81 chars))"),
        ("x" * 10_000, f"(unparsable '{'x' * 79}...(10002 chars))"),
        (
            "BTCUSDT-" + "P" * 10_000 + ".BYBIT",
            f"({'BTCUSDT-' + 'P' * 72}...(10014 chars) is not on BINANCE)",
        ),
        (
            "ETHUSDT-" + "P" * 10_000 + ".BINANCE",
            f"({'ETHUSDT-' + 'P' * 72}...(10016 chars) is not a BTCUSDT contract)",
        ),
    ],
    ids=["at_limit", "over_limit", "huge_unparsable", "huge_other_venue", "huge_other_symbol"],
)
def test_publish_map_echo_is_bounded(configured: str, expected: str) -> None:
    """A configured id is echoed up to 80 characters, then cut with its full length."""
    out = build_publish_map(["BTCUSDT"], {"BTCUSDT": configured}, "BINANCE")
    assert out == f"BTCUSDT: no valid instrument id {expected}"


# ----------------------------------------------------------------- V-PUB


_BTC = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
_ETH = InstrumentId.from_str("ETHUSDT-PERP.BINANCE")


def test_publish_leverage_on_real_margin_account() -> None:
    """V-PUB: a real NT ``MarginAccount`` cached at 10x reads the published 1x."""
    account = TestExecStubs.margin_account()
    account.set_leverage(_BTC, Decimal(10))
    account.set_leverage(_ETH, Decimal(10))

    assert publish_leverage(account, {"BTCUSDT": _BTC, "ETHUSDT": _ETH}, 1) is None
    assert account.leverage(_BTC) == Decimal(1)
    assert account.leverage(_ETH) == Decimal(1)


def test_publish_leverage_sets_unset_instrument() -> None:
    """V-PUB: ``leverage()`` is None until set; publishing sets and verifies it."""
    account = TestExecStubs.margin_account()
    assert account.leverage(_BTC) is None

    assert publish_leverage(account, {"BTCUSDT": _BTC}, 7) is None
    assert account.leverage(_BTC) == Decimal(7)


def test_publish_leverage_without_account() -> None:
    """V-PUB: no venue account in the NT cache."""
    assert publish_leverage(None, {"BTCUSDT": _BTC}, 1) == "no venue account in the NT cache"


class _RaisingAccount:
    def set_leverage(self, iid: object, leverage: object) -> None:
        raise ValueError("leverage was not >= 1 MARKER")

    def leverage(self, iid: object) -> object:
        return Decimal(10)


class _IgnoringAccount:
    """A setter that does nothing: the guard would still read the old 10x."""

    def __init__(self) -> None:
        self.sets: list[tuple[object, object]] = []

    def set_leverage(self, iid: object, leverage: object) -> None:
        self.sets.append((iid, leverage))

    def leverage(self, iid: object) -> object:
        return Decimal(10)


def test_publish_leverage_setter_raises(caplog: pytest.LogCaptureFixture) -> None:
    """V-PUB: a raising setter is reported and logged by class only."""
    with caplog.at_level(logging.WARNING):
        out = publish_leverage(_RaisingAccount(), {"BTCUSDT": _BTC}, 1)

    assert out == "BTCUSDT: leverage publish failed (ValueError)"
    assert "venue leverage publish failed for BTCUSDT (ValueError)" in caplog.text
    assert "MARKER" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


def test_publish_leverage_zero_on_real_account_fails() -> None:
    """NT refuses leverage < 1: reported, never silently accepted."""
    account = TestExecStubs.margin_account()
    account.set_leverage(_BTC, Decimal(10))
    out = publish_leverage(account, {"BTCUSDT": _BTC}, 0)
    assert out is not None
    assert out.startswith("BTCUSDT: leverage publish failed (")
    assert account.leverage(_BTC) == Decimal(10)


def test_publish_leverage_readback_differs() -> None:
    """V-PUB: a setter that did not take is caught by the read-back."""
    account = _IgnoringAccount()
    out = publish_leverage(account, {"BTCUSDT": _BTC}, 1)

    assert out == "BTCUSDT: guard reads leverage 10 after publishing 1"
    assert account.sets == [(_BTC, Decimal(1))]
    assert type(account.sets[0][1]) is Decimal


def test_publish_leverage_readback_equal_edge() -> None:
    """Read-back equal to the published value passes; one off fails (both sides)."""
    account = _IgnoringAccount()  # always reads 10
    assert publish_leverage(account, {"BTCUSDT": _BTC}, 10) is None
    assert publish_leverage(account, {"BTCUSDT": _BTC}, 9) == (
        "BTCUSDT: guard reads leverage 10 after publishing 9"
    )
    assert publish_leverage(account, {"BTCUSDT": _BTC}, 11) == (
        "BTCUSDT: guard reads leverage 10 after publishing 11"
    )


# ------------------------------------------------------- U3/U4 positionRisk (T4)


def _pos(symbol: object, amount: object, side: object = "BOTH") -> dict[str, object]:
    """A positionRisk v3 row (only the fields the check reads, plus noise)."""
    return {
        "symbol": symbol,
        "positionSide": side,
        "positionAmt": amount,
        "entryPrice": "100.0",
        "markPrice": "101.0",
    }


_FLAT = {"BTCUSDT": Decimal(0), "ETHUSDT": Decimal(0)}


@pytest.mark.parametrize(
    ("body", "amounts", "unknown", "unverified"),
    [
        pytest.param([], _FLAT, [], {}, id="omitted_is_flat"),
        pytest.param(
            [_pos("BTCUSDT", "0.010"), _pos("ETHUSDT", "-1.5")],
            {"BTCUSDT": Decimal("0.010"), "ETHUSDT": Decimal("-1.5")},
            [],
            {},
            id="session_amounts",
        ),
        pytest.param([_pos("BTCUSDT", "-0.000")], _FLAT, [], {}, id="negative_zero_is_flat"),
        pytest.param(
            [_pos("SOLUSDT", "3.5"), _pos("XRPUSDT", "0.0")],
            _FLAT,
            [VenuePosition("SOLUSDT", "BOTH", Decimal("3.5"))],
            {},
            id="foreign_nonzero_only",
        ),
        pytest.param(
            [_pos("BTCUSDT", "0.010", "LONG"), _pos("BTCUSDT", "0", "SHORT")],
            _FLAT,
            [VenuePosition("BTCUSDT", "LONG", Decimal("0.010"))],
            {},
            id="hedge_row_kept_with_side",
        ),
        pytest.param(
            [_pos("BTCUSDT", "NaN")],
            {"ETHUSDT": Decimal(0)},
            [],
            {"BTCUSDT": "BOTH positionAmt is not a finite decimal: 'NaN'"},
            id="nan",
        ),
        pytest.param(
            [_pos("ETHUSDT", "-Infinity")],
            {"BTCUSDT": Decimal(0)},
            [],
            {"ETHUSDT": "BOTH positionAmt is not a finite decimal: '-Infinity'"},
            id="inf",
        ),
        pytest.param(
            [_pos("BTCUSDT", 0.01)],
            {"ETHUSDT": Decimal(0)},
            [],
            {"BTCUSDT": "BOTH positionAmt is not a finite decimal: 0.01"},
            id="number_not_string",
        ),
        pytest.param(
            [_pos("BTCUSDT", "1E+1"), _pos("ETHUSDT", "\uff11")],
            {},
            [],
            {
                "BTCUSDT": "BOTH positionAmt is not a finite decimal: '1E+1'",
                "ETHUSDT": "BOTH positionAmt is not a finite decimal: '\uff11'",
            },
            id="exponent_and_fullwidth_digit",
        ),
        pytest.param(
            [_pos("BTCUSDT", "0.010"), _pos("BTCUSDT", "0.010")],
            {"ETHUSDT": Decimal(0)},
            [],
            {"BTCUSDT": "2 BOTH rows"},
            id="two_both_rows",
        ),
        pytest.param(
            [{"symbol": "ETHUSDT", "positionAmt": "1"}],
            {"BTCUSDT": Decimal(0)},
            [],
            {"ETHUSDT": "positionSide is not BOTH/LONG/SHORT: None"},
            id="missing_side",
        ),
        pytest.param(
            [{"symbol": "BTCUSDT", "positionSide": "BOTH"}],
            {"ETHUSDT": Decimal(0)},
            [],
            {"BTCUSDT": "BOTH positionAmt is not a finite decimal: None"},
            id="missing_amount",
        ),
        pytest.param(
            [{"symbol": "SOLUSDT", "positionSide": "LONG", "positionAmt": "x"}],
            _FLAT,
            [],
            {"SOLUSDT": "LONG positionAmt is not a finite decimal: 'x'"},
            id="foreign_bad_amount_keeps_side",
        ),
    ],
)
def test_parse_positions(
    body: object,
    amounts: dict[str, Decimal],
    unknown: list[VenuePosition],
    unverified: dict[str, str],
) -> None:
    """U4: flat/omitted/-0.000 read as 0; NaN/Inf, non-string amounts, duplicate BOTH
    rows and rows missing side or amount leave that symbol unverified (side kept in
    the reason); nonzero foreign BOTH rows and hedge rows are unknown positions."""
    reading = parse_positions(body, ["BTCUSDT", "ETHUSDT"])
    assert reading.amounts == amounts
    assert reading.unknown == unknown
    assert reading.unverified == unverified


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"code": -2015}, id="not_a_list"),
        pytest.param([_pos("BTCUSDT", "0"), "junk"], id="non_object_row"),
        pytest.param([{"positionSide": "BOTH", "positionAmt": "1"}], id="missing_symbol"),
    ],
)
def test_parse_positions_unattributable_answer(body: object) -> None:
    """U4: an answer that is not a list or holds a row naming no symbol could hide any
    session symbol's row: every session symbol is unverified, none is read as flat."""
    reading = parse_positions(body, ["BTCUSDT", "ETHUSDT"])
    assert reading.amounts == {}
    assert set(reading.unverified) == {"answer", "BTCUSDT", "ETHUSDT"}
    assert reading.unverified["BTCUSDT"] == reading.unverified["answer"]


def test_unknown_position_texts() -> None:
    """U1/U3 wording: a foreign one-way position and a hedge-mode row."""
    assert VenuePosition("SOLUSDT", "BOTH", Decimal("3.5")).describe() == (
        "SOLUSDT 3.5 (not traded by this session)"
    )
    assert VenuePosition("BTCUSDT", "SHORT", Decimal("-0.020")).describe() == (
        "BTCUSDT SHORT -0.020 (hedge-mode row)"
    )


async def test_position_builder_signs_position_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positionRisk read signs GET /fapi/v3/positionRisk {recvWindow, timestamp} with
    the session key and refuses non-USD-M account types like the symbolConfig read."""
    body = [_pos("BTCUSDT", "0.010")]
    record = _patch_factory(monkeypatch, json.dumps(body).encode())
    read = build_position_risk_read("key-p", "secret-p", "USDT_FUTURES", False)

    assert await read() == body
    assert (record["api_key"], record["api_secret"]) == ("key-p", "secret-p")
    assert record["environment"] is BinanceEnvironment.LIVE
    [(method, path, payload, _keys)] = record["calls"]
    assert method is HttpMethod.GET
    assert path == POSITION_RISK_PATH == "/fapi/v3/positionRisk"
    assert set(payload) == {"recvWindow", "timestamp"}
    with pytest.raises(ValueError, match="USDT_FUTURES"):
        build_position_risk_read("k", "s", "COIN_FUTURES", False)


# ------------------------------------------------ alert detail bound (Telegram 4,096)


def test_alert_detail_at_bound_is_unchanged() -> None:
    """A detail of exactly the bound is sent whole; one character more is cut."""
    text = "x" * 3_000
    assert bounded_alert_detail(text) == text
    assert bounded_alert_detail(text + "x") != text + "x"


@pytest.mark.parametrize(
    ("items", "sep", "limit", "expected"),
    [
        pytest.param(
            [f"ALT{i:03}USDT 1.000 (not traded by this session)" for i in range(100)],
            "; ",
            200,
            "ALT000USDT 1.000 (not traded by this session); "
            "ALT001USDT 1.000 (not traded by this session); "
            "ALT002USDT 1.000 (not traded by this session) … (+97 more; full text in status)",
            id="positions",
        ),
        pytest.param(
            [f"S{i}USDT (bad positionAmt)" for i in range(40)],
            ", ",
            120,
            "S0USDT (bad positionAmt), S1USDT (bad positionAmt) … (+38 more; full text in status)",
            id="unverified",
        ),
    ],
)
def test_long_alert_detail_keeps_a_sample_and_the_count(
    items: list[str], sep: str, limit: int, expected: str
) -> None:
    """Cut at the last whole item that leaves room for the suffix; the suffix counts
    the items left out. The result never exceeds the bound."""
    out = bounded_alert_detail(sep.join(items), limit)
    assert out == expected
    assert len(out) <= limit


def test_long_alert_detail_without_items_is_cut_hard() -> None:
    out = bounded_alert_detail("y" * 500, 100)
    assert out == "y" * 52 + " … (cut; full text in status)"
    assert len(out) <= 100
