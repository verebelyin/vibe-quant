"""Tests for scripts/agents/models.py — the model routing CLI.

Every test drives the CLI with `--file` against a throwaway JSON file in
tmp_path (never docs/orchestration/models.json), so the committed routing file
is untouched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "models.py"
_spec = importlib.util.spec_from_file_location("models", _SCRIPT)
assert _spec is not None and _spec.loader is not None
models = importlib.util.module_from_spec(_spec)
sys.modules["models"] = models
_spec.loader.exec_module(models)


def _model(
    runtime: str = "claude",
    target: dict[str, object] | None = None,
    enabled: bool = True,
    until: str | None = None,
    note: str = "",
) -> dict[str, object]:
    return {
        "runtime": runtime,
        "target": target if target is not None else {"model": "sonnet"},
        "enabled": enabled,
        "until": until,
        "note": note,
    }


def _write(path: Path, models_map: dict[str, object], routes: dict[str, object]) -> Path:
    path.write_text(json.dumps({"models": models_map, "routes": routes}, indent=2) + "\n")
    return path


def test_route_picks_fallback_when_first_is_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path / "models.json",
        {
            "off-one": _model(enabled=False, until="2999-01-01T00:00:00+00:00", note="cooling"),
            "fallback": _model(
                runtime="t3",
                target={
                    "providerInstanceId": "commandcode_command",
                    "model": "deepseek/deepseek-v4.1-flash",
                    "options": {"effort": "high"},
                },
            ),
        },
        {"maker": ["off-one", "fallback"]},
    )

    rc = models.main(["--file", str(path), "route", "maker"])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    head, _, payload = lines[0].partition(" -> ")
    key, _, rest = payload.partition(" ")
    runtime, _, target = rest.partition(" ")
    assert head == "maker"
    assert (key, runtime) == ("fallback", "t3")
    assert json.loads(target) == {
        "providerInstanceId": "commandcode_command",
        "model": "deepseek/deepseek-v4.1-flash",
        "options": {"effort": "high"},
    }
    assert lines[1].startswith("  skipped off-one: ")
    assert "2999-01-01" in lines[1]
