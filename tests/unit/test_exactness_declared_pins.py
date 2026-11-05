"""exactness_239.py --expect: a declared comparability break's pins, loaded fail-closed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "exactness_239.py"
_spec = importlib.util.spec_from_file_location("exactness_239", _SCRIPT)
assert _spec is not None and _spec.loader is not None
ex = importlib.util.module_from_spec(_spec)
sys.modules["exactness_239"] = ex
_spec.loader.exec_module(ex)

_FIVE = {"sharpe": 0.81, "trades": 66, "return": 0.15, "profit_factor": 1.6, "max_drawdown": 0.06}


def _pins(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(data))
    return path


def test_declared_pins_load_per_section(tmp_path: Path) -> None:
    multi = {**_FIVE, "trades": 160}
    pins = ex.load_declared_pins(_pins(tmp_path, {"screening": _FIVE, "validation_multi": multi}))
    assert pins == {"screening": _FIVE, "validation_multi": multi}


@pytest.mark.parametrize(
    "data",
    [
        {},  # nothing declared
        {"screnning": _FIVE},  # typo'd section
        {"screening": {k: v for k, v in _FIVE.items() if k != "max_drawdown"}},  # partial override
        {"screening": {**_FIVE, "win_rate": 0.5}},  # extra metric
        {"screening": {**_FIVE, "trades": "66"}},  # string
        {"screening": {**_FIVE, "trades": True}},  # bool is an int in Python
        [],
    ],
)
def test_malformed_pins_are_refused(tmp_path: Path, data: object) -> None:
    with pytest.raises(ValueError):
        ex.load_declared_pins(_pins(tmp_path, data))
