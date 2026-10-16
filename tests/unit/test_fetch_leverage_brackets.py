"""scripts/fetch_leverage_brackets.py — atomic, validated snapshot replacement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.liquidation.tiers import load_tier_tables

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fetch_leverage_brackets.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_leverage_brackets", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_leverage_brackets"] = module
    spec.loader.exec_module(module)
    return module


script = _load()


def _valid_symbol() -> dict[str, object]:
    return {
        "brackets": [
            {"floor": 0.0, "cap": 50_000.0, "mmr": 0.004, "cum": 0.0, "max_leverage": 125},
            {"floor": 50_000.0, "cap": 250_000.0, "mmr": 0.005, "cum": 50.0, "max_leverage": 100},
        ],
        "clearance_fee_rate": None,
        "clearance_fee_source": script.CLEARANCE_FEE_SOURCE,
    }


def _invalid_symbol() -> dict[str, object]:
    # A gap between cap 50_000 and floor 60_000: load_tier_tables rejects it.
    return {
        "brackets": [
            {"floor": 0.0, "cap": 50_000.0, "mmr": 0.004, "cum": 0.0, "max_leverage": 125},
            {"floor": 60_000.0, "cap": 250_000.0, "mmr": 0.005, "cum": 60.0, "max_leverage": 100},
        ],
        "clearance_fee_rate": None,
        "clearance_fee_source": script.CLEARANCE_FEE_SOURCE,
    }


def test_valid_fetch_replaces_target_and_the_result_loads(tmp_path, monkeypatch):
    out = tmp_path / "brackets.json"
    out.write_bytes(b'{"version": "old"}\n')
    monkeypatch.setattr(script, "fetch_keyless", lambda symbols: {"BTCUSDT": _valid_symbol()})

    assert script.main(["--symbols", "BTCUSDT", "--out", str(out)]) == 0

    tables = load_tier_tables(out)
    assert [bracket.mmr for bracket in tables["BTCUSDT"].brackets] == [0.004, 0.005]
    assert out.read_bytes().endswith(b"\n")


def test_invalid_fetch_keeps_old_bytes_and_leaves_no_temp_file(tmp_path, monkeypatch, capsys):
    out = tmp_path / "brackets.json"
    original = b'{"version": "old", "symbols": {}}\n'
    out.write_bytes(original)
    monkeypatch.setattr(script, "fetch_keyless", lambda symbols: {"BTCUSDT": _invalid_symbol()})

    assert script.main(["--symbols", "BTCUSDT", "--out", str(out)]) == 1

    assert out.read_bytes() == original
    assert "BTCUSDT" in capsys.readouterr().err
    assert [path.name for path in tmp_path.iterdir()] == [out.name]


def test_invalid_fetch_without_existing_target_creates_no_target(tmp_path, monkeypatch):
    out = tmp_path / "nested" / "brackets.json"
    monkeypatch.setattr(script, "fetch_keyless", lambda symbols: {"BTCUSDT": _invalid_symbol()})

    assert script.main(["--symbols", "BTCUSDT", "--out", str(out)]) == 1

    assert not out.exists()
    assert list(out.parent.iterdir()) == []
