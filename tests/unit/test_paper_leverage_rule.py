"""Paper validates leverage through the one shared ``LiquidationSettings`` rule.

Bead vibe-quant-74gyn (task P1). Before this change paper read a validation
run's stored leverage with its own loose rule (any non-positive or non-numeric
value silently fell back to 10, ``float(10**400)`` raised ``OverflowError``),
then applied its own integer check in ``_resolve_run_config`` and forced the
config-file override through ``int()`` (so ``2.5`` became ``2`` and ``True``
became ``1``). Both now go through
:class:`vibe_quant.liquidation_settings.LiquidationSettings`; the
``max_leverage`` cap stays.

Contract rows (design r2, job 20261015-honest-safe):
* R35  validated leverage absent -> 10
* R35b stored 5 / 10.0 / 125 pass the rule; 125 then trips ``max_leverage``
       (default 20) with "leverage 125 exceeds max_leverage 20"
* R36  stored 0.5, 2.5, 0, -5, 126, 1e308, 10**400, True, "5", None, nan, inf
       -> ``ConfigurationError("Validation run {id} has invalid liquidation
       settings: " + the shared rule's text)`` before any node is built
* R37  stored ``liquidation_mode`` "ISOLATED" or None -> the same prefix
* R38  config-file ``leverage`` 2.5 -> ``ConfigurationError``
* R38b config-file ``leverage`` 3 -> 3
* R38c config-file ``leverage`` absent -> None (the validated value is used)
* R38d config-file ``leverage`` True, "5", 0, 126 -> ``ConfigurationError``
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_audit_paper_node import (
    fake_tradingnode as _audit_fake_tradingnode,  # noqa: F401  (via getfixturevalue)
)
from tests.unit.test_audit_paper_node import seeded as _audit_seeded  # noqa: F401
from vibe_quant.db.state_manager import StateManager
from vibe_quant.liquidation_settings import LIQUIDATION_MODES
from vibe_quant.paper.cli import load_config_from_json
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.node import (
    PaperTradingNode,
    validated_run_leverage,
    validated_run_liquidation,
)

if TYPE_CHECKING:
    from pathlib import Path

# The audit suite's `seeded` (strategy + completed validation run) and
# `fake_tradingnode` (no real NT node) fixtures are imported under aliases and
# pulled in by name: importing them under their own names shadows the test
# parameters (ruff F811).
_LEV_INVALID_PREFIX = "leverage must be a whole number in [1, 125], got "
_INVALID_LEVERAGES: list[object] = [
    0.5,
    2.5,
    0,
    -5,
    126,
    1e308,
    10**400,
    True,
    "5",
    None,
    math.nan,
    math.inf,
]
_INVALID_LEVERAGE_IDS = [
    "0.5",
    "2.5",
    "0",
    "-5",
    "126",
    "1e308",
    "10**400",
    "true",
    "str",
    "null",
    "nan",
    "inf",
]


def _validation_run(seeded: dict[str, Any], parameters: dict[str, Any]) -> int:
    sm = StateManager(seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seeded["sid"],
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters=parameters,
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid




def test_validated_run_liquidation_absent_uses_the_shared_default() -> None:
    """No stored leverage -> the shared default 10 (no second NULL rule)."""
    absent = {"id": 1, "parameters": {}}
    settings = validated_run_liquidation(absent)
    assert settings.mode == "off"
    assert settings.leverage == 10.0
    assert validated_run_leverage(absent) == 10.0
    # Non-dict parameters are "absent", exactly as the old rule read them.
    assert validated_run_leverage({"id": 2, "parameters": None}) == 10.0
    # A valid stored mode + leverage is read, not rejected.
    stored = validated_run_liquidation(
        {"id": 3, "parameters": {"liquidation_mode": "isolated", "leverage": 5}}
    )
    assert stored.mode == "isolated"
    assert stored.leverage == 5.0


@pytest.mark.parametrize(("stored", "expected"), [(5, 5), (10.0, 10)], ids=["int", "float"])
async def test_validated_leverage_in_domain_passes_unchanged(
    request: pytest.FixtureRequest, stored: object, expected: int
) -> None:
    """A whole-number stored leverage in [1, 125] reaches the node unchanged."""
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    request.getfixturevalue("_audit_fake_tradingnode")
    rid = _validation_run(seeded, {"leverage": stored})
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.leverage == expected
    finally:
        await node._shutdown()


async def test_validated_leverage_125_trips_max_leverage_cap(
    request: pytest.FixtureRequest,
) -> None:
    """The rule admits 125; the max_leverage cap (default 20) then refuses it."""
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    fake: type[Any] = request.getfixturevalue("_audit_fake_tradingnode")
    rid = _validation_run(seeded, {"leverage": 125})
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    with pytest.raises(ConfigurationError, match=r"leverage 125 exceeds max_leverage 20"):
        await node._initialize()
    assert fake.instances == []




@pytest.mark.parametrize("bad", _INVALID_LEVERAGES, ids=_INVALID_LEVERAGE_IDS)
async def test_stored_fractional_or_garbage_leverage_fails_paper_start(
    request: pytest.FixtureRequest, bad: object
) -> None:
    """A stored non-whole/out-of-domain leverage fails before any node is built."""
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    fake: type[Any] = request.getfixturevalue("_audit_fake_tradingnode")
    rid = _validation_run(seeded, {"leverage": bad})
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    expected = f"Validation run {rid} has invalid liquidation settings: {_LEV_INVALID_PREFIX}{bad!r}"
    with pytest.raises(ConfigurationError) as excinfo:
        await node._initialize()
    assert str(excinfo.value) == expected
    assert fake.instances == [], "an invalid leverage must not build a node"




@pytest.mark.parametrize("mode", ["ISOLATED", None], ids=["uppercase", "null"])
async def test_stored_invalid_liquidation_mode_fails_paper_start(
    request: pytest.FixtureRequest, mode: object
) -> None:
    """The same rule also rejects a stored mode (case-sensitive; null is not off)."""
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    fake: type[Any] = request.getfixturevalue("_audit_fake_tradingnode")
    rid = _validation_run(seeded, {"liquidation_mode": mode})
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    expected = (
        f"Validation run {rid} has invalid liquidation settings: "
        f"liquidation_mode must be one of {LIQUIDATION_MODES}, got {mode!r}"
    )
    with pytest.raises(ConfigurationError) as excinfo:
        await node._initialize()
    assert str(excinfo.value) == expected
    assert fake.instances == []


async def test_cfg_level_leverage_override_is_validated_by_the_shared_rule(
    request: pytest.FixtureRequest,
) -> None:
    """A programmatic cfg override is validated too (``True`` is not 1)."""
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    fake: type[Any] = request.getfixturevalue("_audit_fake_tradingnode")
    node = PaperTradingNode(_config(seeded, leverage=True))
    expected = f"{_LEV_INVALID_PREFIX}True"
    with pytest.raises(ConfigurationError) as excinfo:
        await node._initialize()
    assert str(excinfo.value) == expected
    assert fake.instances == []




@pytest.fixture(autouse=True)
def _binance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "test_secret")


def _config_file(tmp_path: Path, **extra: object) -> Path:
    data = {"trader_id": "PAPER-001", "symbols": ["BTCUSDT"], "strategy_id": 1, **extra}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("bad", [2.5, True, "5", 0, 126], ids=["2.5", "true", "str", "0", "126"])
def test_config_file_leverage_override_invalid(tmp_path: Path, bad: object) -> None:
    """The loader validates the raw override with the shared rule (no int())."""
    path = _config_file(tmp_path, leverage=bad)
    with pytest.raises(ConfigurationError, match=re.escape(_LEV_INVALID_PREFIX.strip())):
        load_config_from_json(path)


def test_config_file_leverage_override_whole_number_is_kept(tmp_path: Path) -> None:
    """A whole-number override is stored as an int, not truncated."""
    config = load_config_from_json(_config_file(tmp_path, leverage=3))
    assert config.leverage == 3
    assert isinstance(config.leverage, int)


def test_config_file_leverage_absent_is_none(tmp_path: Path) -> None:
    """Absent -> None, so the validated run's leverage is used."""
    config = load_config_from_json(_config_file(tmp_path))
    assert config.leverage is None
