"""Tests for scripts/agents/route_guard.py — the PreToolUse route guard.

The hook is driven as Claude Code drives it: one JSON payload on stdin, the exit
code decides (2 = block). A swarm job is a throwaway directory with the brief and
the route record mkbrief writes next to it, so no real checkout is touched.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "agents" / "route_guard.py"
_SETTINGS = _ROOT / ".claude" / "settings.json"
_T3 = "mcp__t3-code__delegate_task"
_ROUTED = {"providerInstanceId": "commandcode_command", "model": "deepseek/deepseek-v4.1-flash"}
_USED = {"providerInstanceId": "commandcode_command", "model": "deepseek/deepseek-v4-pro"}


def _job(tmp_path: Path, target: dict[str, object] | None = None, ident: str = "cc-demo-1") -> Path:
    """A job dir holding `<id>.md` and (unless target is None) `<id>.delegate.json`."""
    job = tmp_path / "job"
    (job / "prompts").mkdir(parents=True)
    brief = job / "prompts" / f"{ident}.md"
    brief.write_text("# brief\n")
    if target is not None:
        (job / "prompts" / f"{ident}.delegate.json").write_text(
            json.dumps({
                "task": f"Read your brief at {brief} and follow it exactly.",
                "target": target, "mode": "async", "clientRequestId": ident,
            })  # fmt: skip
        )
    return job


def _task(job: Path, ident: str = "cc-demo-1") -> str:
    return (
        f"Read your brief at {job / 'prompts' / f'{ident}.md'} and follow it exactly.\n"
        "TELEMETRY: task=demo round=1"
    )


def _t3_payload(job: Path, target: dict[str, object], ident: str = "cc-demo-1") -> dict[str, object]:
    return {
        "tool_name": _T3,
        "tool_input": {"task": _task(job, ident), "target": target,
                       "mode": "async", "clientRequestId": ident},
    }  # fmt: skip


def _agent_payload(job: Path, model: str | None, ident: str = "cc-demo-1") -> dict[str, object]:
    tool_input: dict[str, object] = {
        "description": "demo", "subagent_type": "general", "prompt": _task(job, ident),
    }
    if model is not None:
        tool_input["model"] = model
    return {"tool_name": "Agent", "tool_input": tool_input}


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "SWARM_ROUTE_OVERRIDE"}
    env.update(extra)
    return env


def _run(payload: object, **env: str) -> subprocess.CompletedProcess[str]:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(_SCRIPT)], input=raw, capture_output=True, text=True,
        env=_env(**env), timeout=60, check=False,
    )  # fmt: skip


def _refusal(ident: str, routed: str, used: str) -> str:
    return (
        f"route_guard: {ident} routed to {routed} by models.py; this call uses {used}. "
        "Re-render with mkbrief --route, or run models.py set (user decision) first."
    )


# ------------------------------------------------------------------ delegate_task


def test_the_recorded_route_is_allowed(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    done = _run(_t3_payload(job, dict(_ROUTED)))
    assert done.returncode == 0
    assert done.stderr == ""


def test_a_different_model_is_refused(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    done = _run(_t3_payload(job, dict(_USED)))
    assert done.returncode == 2
    assert done.stderr.strip() == _refusal(
        "cc-demo-1", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-pro"
    )


def test_a_different_provider_is_refused(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    same_model = {"providerInstanceId": "codex", "model": "deepseek/deepseek-v4.1-flash"}
    done = _run(_t3_payload(job, same_model))
    assert done.returncode == 2
    assert "route_guard: cc-demo-1 routed to" in done.stderr


def test_a_task_without_a_brief_path_is_allowed(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    payload = _t3_payload(job, dict(_USED))
    payload["tool_input"]["task"] = "Summarize the repository."  # type: ignore[index]
    done = _run(payload)
    assert done.returncode == 0
    assert done.stderr == ""


def test_a_brief_without_a_record_is_allowed(tmp_path: Path) -> None:
    job = _job(tmp_path, None)
    done = _run(_t3_payload(job, dict(_USED)))
    assert done.returncode == 0


def test_garbage_stdin_is_allowed(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    done = _run("{not json")
    assert done.returncode == 0
    assert done.stderr == ""
    assert not (job / "route-overrides.jsonl").exists()


def test_an_unrelated_tool_is_allowed(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    payload = _t3_payload(job, dict(_USED))
    payload["tool_name"] = "Bash"
    done = _run(payload)
    assert done.returncode == 0


# ------------------------------------------------------------------------ override


def test_the_override_env_allows_and_logs_it(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    done = _run(_t3_payload(job, dict(_USED)), SWARM_ROUTE_OVERRIDE="user chose pro for this task")
    assert done.returncode == 0
    assert done.stderr == ""
    rows = (job / "route-overrides.jsonl").read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["id"] == "cc-demo-1"
    assert row["routed"] == "deepseek/deepseek-v4.1-flash"
    assert row["used"] == "deepseek/deepseek-v4-pro"
    assert row["reason"] == "user chose pro for this task"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", row["ts"])


def test_the_override_log_keeps_two_unrelated_calls(tmp_path: Path) -> None:
    first = _job(tmp_path / "one", _ROUTED, ident="cc-alpha-1")
    second = _job(tmp_path / "two", _ROUTED, ident="cc-beta-2")
    other = {"providerInstanceId": "codex", "model": "gpt-6.1-sol"}
    assert _run(
        _t3_payload(first, dict(_USED), "cc-alpha-1"),
        SWARM_ROUTE_OVERRIDE="alpha needs the pro model",
    ).returncode == 0
    assert _run(
        _t3_payload(second, dict(other), "cc-beta-2"),
        SWARM_ROUTE_OVERRIDE="beta is a codex task",
    ).returncode == 0
    rows = [json.loads(line) for line in (first / "route-overrides.jsonl").read_text().splitlines()]
    rows += [json.loads(line) for line in (second / "route-overrides.jsonl").read_text().splitlines()]
    assert [(row["id"], row["routed"], row["used"], row["reason"]) for row in rows] == [
        ("cc-alpha-1", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-pro",
         "alpha needs the pro model"),
        ("cc-beta-2", "deepseek/deepseek-v4.1-flash", "gpt-6.1-sol", "beta is a codex task"),
    ]  # fmt: skip


# ----------------------------------------------------------------------- Agent tool


def _claude_job(tmp_path: Path, ident: str = "cc-demo-1") -> Path:
    job = _job(tmp_path, None, ident)
    (job / "prompts" / f"{ident}.delegate.json").write_text(
        json.dumps({"role": "fixer", "model": "sonnet"})
    )
    return job


def test_agent_with_the_routed_model_is_allowed(tmp_path: Path) -> None:
    job = _claude_job(tmp_path)
    done = _run(_agent_payload(job, "sonnet"))
    assert done.returncode == 0
    assert done.stderr == ""


def test_agent_with_another_model_is_refused(tmp_path: Path) -> None:
    job = _claude_job(tmp_path)
    done = _run(_agent_payload(job, "opus"))
    assert done.returncode == 2
    assert done.stderr.strip() == _refusal("cc-demo-1", "sonnet", "opus")


def test_agent_without_a_model_is_refused_as_an_inherited_override(tmp_path: Path) -> None:
    job = _claude_job(tmp_path)
    done = _run(_agent_payload(job, None))
    assert done.returncode == 2
    assert done.stderr.strip() == _refusal("cc-demo-1", "sonnet", "(inherited)")


# ------------------------------------------------------------------------ settings


def test_settings_register_the_guard_for_both_dispatch_tools() -> None:
    settings = json.loads(_SETTINGS.read_text())
    entries = [
        entry
        for entry in settings["hooks"]["PreToolUse"]
        if "route_guard.py" in json.dumps(entry)
    ]
    assert len(entries) == 1
    matcher = re.compile(entries[0]["matcher"])
    assert matcher.search(_T3) and matcher.search("Agent")
    assert not matcher.search("Bash")
    command = entries[0]["hooks"][0]["command"]
    assert 'python3 "$CLAUDE_PROJECT_DIR"/scripts/agents/route_guard.py' in command


def test_a_prompt_that_only_quotes_a_brief_line_is_not_its_dispatch(tmp_path: Path) -> None:
    """rev-route-1 F1: an Agent call explaining an old dispatch is not that dispatch."""
    job = _claude_job(tmp_path)
    payload = _agent_payload(job, "haiku")
    payload["tool_input"]["prompt"] = (  # type: ignore[index]
        "Explain this historical dispatch example:\n```\n" + _task(job) + "\n```\n"
    )
    done = _run(payload)
    assert done.returncode == 0
    assert done.stderr == ""


def test_a_leading_brief_line_without_its_suffix_is_still_checked(tmp_path: Path) -> None:
    job = _claude_job(tmp_path)
    payload = _agent_payload(job, "haiku")
    payload["tool_input"]["prompt"] = (  # type: ignore[index]
        f"  Read your brief at {job / 'prompts' / 'cc-demo-1.md'}\n"
    )
    done = _run(payload)
    assert done.returncode == 2
    assert done.stderr.strip() == _refusal("cc-demo-1", "sonnet", "haiku")


def test_an_override_that_cannot_be_logged_is_refused(tmp_path: Path) -> None:
    job = _job(tmp_path, _ROUTED)
    (job / "route-overrides.jsonl").mkdir()
    done = _run(_t3_payload(job, dict(_USED)), SWARM_ROUTE_OVERRIDE="user chose pro")
    assert done.returncode == 2
    assert "cannot log the override" in done.stderr
