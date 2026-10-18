"""Tests for scripts/agents/models.py — the model routing CLI.

Every test drives the CLI with `--file` against a throwaway JSON file in
tmp_path (never docs/orchestration/models.json), so the committed routing file
is untouched.
"""

from __future__ import annotations

import importlib.util
import json
import re
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

_REAL_MODELS = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "models.json"
_SKILL = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "orchestrate" / "SKILL.md"


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


def test_until_in_the_past_reenables_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path / "models.json",
        {
            "off-one": _model(enabled=False, until="2000-01-01T00:00:00+00:00"),
            "fallback": _model(),
        },
        {"maker": ["off-one", "fallback"]},
    )

    rc = models.main(["--file", str(path), "route", "maker"])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("maker -> off-one ")
    assert not any(ln.startswith("  skipped") for ln in lines)


def test_set_off_then_on_round_trips_and_stamps_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path / "models.json", {"sonnet": _model()}, {"reviewer": ["sonnet"]})

    assert models.main(["--file", str(path), "set", "sonnet", "--off", "--note", "paused"]) == 0
    data = json.loads(path.read_text())
    assert data["models"]["sonnet"]["enabled"] is False
    assert data["models"]["sonnet"]["note"] == "paused"
    assert isinstance(data["changed"], str) and data["changed"]

    assert models.main(["--file", str(path), "set", "sonnet", "--on"]) == 0
    data = json.loads(path.read_text())
    assert data["models"]["sonnet"]["enabled"] is True
    assert data["models"]["sonnet"]["until"] is None
    assert data["models"]["sonnet"]["note"] == "paused"
    capsys.readouterr()


def test_set_off_with_unparseable_until_exits_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path / "models.json", {"sonnet": _model()}, {"reviewer": ["sonnet"]})
    before = path.read_text()

    rc = models.main(["--file", str(path), "set", "sonnet", "--off", "--until", "tomorrow maybe"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "until" in err
    assert "tomorrow maybe" in err
    assert path.read_text() == before


def test_unknown_role_exits_2_listing_valid_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path / "models.json", {"sonnet": _model()}, {"maker": ["sonnet"]})

    rc = models.main(["--file", str(path), "route", "nope"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "nope" in err
    assert "maker" in err


def test_unknown_key_exits_2_listing_valid_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path / "models.json", {"sonnet": _model()}, {"maker": ["sonnet"]})

    rc = models.main(["--file", str(path), "set", "nope", "--off"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "nope" in err
    assert "sonnet" in err


def test_show_renders_on_off_and_note(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path / "models.json",
        {
            "sonnet": _model(note="reviewer"),
            "haiku": _model(enabled=False, until="2026-11-01T00:00:00+00:00", note="cheap"),
            "opus": _model(enabled=False, note="off indefinitely"),
        },
        {"maker": ["sonnet"]},
    )

    rc = models.main(["--file", str(path), "show"])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "sonnet ON reviewer"
    assert lines[1] == "haiku OFF(until 2026-11-01T00:00:00+00:00) cheap"
    assert lines[2] == "opus OFF off indefinitely"


def test_operator_routes_to_deepseek_flash_on_the_real_models_file() -> None:
    data = json.loads(_REAL_MODELS.read_text())

    assert data["routes"]["operator"] == ["deepseek-flash", "sonnet"]
    assert data["models"]["deepseek-flash"]["runtime"] == "t3"
    assert data["models"]["deepseek-flash"]["target"] == {
        "providerInstanceId": "commandcode_command",
        "model": "deepseek/deepseek-v4.1-flash",
        "options": {"effort": "high"},
    }


def test_operator_falls_back_to_sonnet_when_deepseek_flash_is_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(_REAL_MODELS.read_text())
    data["models"]["deepseek-flash"]["enabled"] = False
    data["models"]["deepseek-flash"]["until"] = None
    path = _write(tmp_path / "models.json", data["models"], data["routes"])

    rc = models.main(["--file", str(path), "route", "operator"])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    role, _, payload = lines[0].partition(" -> ")
    key, _, rest = payload.partition(" ")
    runtime, _, target = rest.partition(" ")
    assert role == "operator"
    assert (key, runtime) == ("sonnet", "claude")
    assert json.loads(target) == {"model": "sonnet"}
    assert lines[1] == "  skipped deepseek-flash: off"


def test_skill_role_list_and_models_json_routes_cannot_drift() -> None:
    match = re.search(r"\(roles: ([^)]+)\)", _SKILL.read_text())
    assert match is not None, "SKILL.md no longer has a '(roles: ...)' list"
    skill_roles = {role.strip() for role in match.group(1).split(",")}
    routes = set(json.loads(_REAL_MODELS.read_text())["routes"])

    assert skill_roles == routes
