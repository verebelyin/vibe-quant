"""Steering files stay small: word budgets, no changelogs, no incident narrative.

Agent definitions, the orchestrate skill and docs/orchestration/ are loaded into
agents' context on every task. They grew from 0 to ~48k words in two weeks of
one-rule-per-incident edits; on 2026-10-17 they were distilled to ~18k. Adding a line
now means merging or deleting another; raising a budget is the user's call
(orchestrate skill § Self-improvement). History lives in commit messages and
docs/orchestration/history.md (loaded by nothing).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Word budgets (`len(text.split())`), ~10% above the distilled size.
BUDGETS: dict[str, int] = {
    "CLAUDE.md": 2860,
    ".claude/skills/orchestrate/SKILL.md": 1650,
    ".claude/skills/grind/SKILL.md": 700,
    ".claude/skills/discovery-run/SKILL.md": 1900,
    ".claude/skills/perf-profiling/SKILL.md": 300,
    ".claude/agents/architect.md": 1650,
    ".claude/agents/reviewer.md": 1850,
    ".claude/agents/reviewer-lite.md": 830,
    ".claude/agents/implementer.md": 820,
    ".claude/agents/verifier.md": 920,
    ".claude/agents/risk-officer.md": 980,
    ".claude/agents/overfit-auditor.md": 580,
    ".claude/agents/alpha-scout.md": 560,
    ".claude/agents/strategy-author.md": 500,
    ".claude/agents/backtest-operator.md": 540,
    "docs/orchestration/README.md": 1780,
    "docs/orchestration/cheap-agents.md": 610,
    "docs/orchestration/brief-checklist.md": 860,
    "docs/orchestration/research-swarm.md": 700,
    "docs/orchestration/prompts/cmd-task.md": 500,
    "docs/orchestration/prompts/fix-task.md": 200,
    "docs/orchestration/prompts/review-task.md": 220,
    "docs/orchestration/prompts/bus-protocol.md": 270,
    "docs/orchestration/prompts/commit-message.md": 350,
    "docs/claude/discovery.md": 1560,
    "docs/claude/engine-modes.md": 480,
    "docs/claude/nautilus-gotchas.md": 800,
    "docs/claude/beads.md": 910,
    "docs/claude/agent-speed.md": 240,
    "docs/claude/ui-testing.md": 300,
}

MAX_LINE_CHARS = 1300

# The always-loaded user profile: Claude Code's auto-memory index for this checkout
# (outside the repo, so only checked where it exists). Same cap as the tiered-memory model.
PROFILE = (
    Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(ROOT)) / "memory" / "MEMORY.md"
)
PROFILE_MAX_CHARS = 4000
PROFILE_MAX_FACTS = 100
NARRATIVE = re.compile(r"\bjob 2026\d{4}-|\b2026\d{4}-[a-z][a-z0-9-]+\b")


def _steering_files() -> list[Path]:
    files = [ROOT / "CLAUDE.md"]
    files += sorted((ROOT / ".claude" / "agents").glob("*.md"))
    files += sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    files += sorted((ROOT / "docs" / "orchestration").glob("*.md"))
    files += sorted((ROOT / "docs" / "orchestration" / "prompts").rglob("*.md"))
    return [f for f in files if f.name != "history.md"]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@pytest.mark.parametrize("rel", sorted(BUDGETS))
def test_steering_file_within_word_budget(rel: str) -> None:
    words = len((ROOT / rel).read_text().split())
    assert words <= BUDGETS[rel], (
        f"{rel} has {words} words, budget {BUDGETS[rel]}: merge or delete a line "
        "(raising a budget needs the user)"
    )


def test_every_agent_and_skill_has_a_budget() -> None:
    governed = [
        _rel(f)
        for f in _steering_files()
        if f.parent.name == "agents" or f.name == "SKILL.md"
    ]
    assert [r for r in governed if r not in BUDGETS] == []


@pytest.mark.parametrize("path", _steering_files(), ids=_rel)
def test_no_changelog_section(path: Path) -> None:
    assert not re.search(r"^#+\s*Changelog\b", path.read_text(), re.M), (
        f"{_rel(path)}: history goes in the commit message, not the steering file"
    )


@pytest.mark.parametrize("path", _steering_files(), ids=_rel)
def test_no_overlong_lines(path: Path) -> None:
    long = [i for i, line in enumerate(path.read_text().splitlines(), 1) if len(line) > MAX_LINE_CHARS]
    assert long == [], f"{_rel(path)}: lines over {MAX_LINE_CHARS} chars: {long}"


@pytest.mark.parametrize(
    "path",
    [f for f in _steering_files() if f.parent.name == "agents" or f.name == "SKILL.md"],
    ids=_rel,
)
def test_no_incident_narrative_in_agent_prompts(path: Path) -> None:
    hits = [m.group(0) for m in NARRATIVE.finditer(path.read_text())]
    assert hits == [], f"{_rel(path)}: job ids belong in commit messages: {hits[:5]}"


def profile_problems(text: str) -> list[str]:
    """Why a user profile breaks the cap (a fact = one `- ` bullet line); [] when it fits."""
    facts = sum(1 for line in text.splitlines() if line.lstrip().startswith("- "))
    out = []
    if len(text) > PROFILE_MAX_CHARS:
        out.append(f"{len(text)} chars > {PROFILE_MAX_CHARS}")
    if facts > PROFILE_MAX_FACTS:
        out.append(f"{facts} facts > {PROFILE_MAX_FACTS}")
    return out


def test_profile_cap_counts_chars_and_facts() -> None:
    assert profile_problems("# Profile\n" + "- fact\n" * 100) == []
    assert profile_problems("- fact\n" * 101) == ["101 facts > 100"]
    assert profile_problems("x" * 4001) == ["4001 chars > 4000"]


def test_user_profile_within_cap() -> None:
    if not PROFILE.is_file():
        pytest.skip(f"no auto-memory profile at {PROFILE}")
    assert profile_problems(PROFILE.read_text()) == [], (
        f"{PROFILE}: merge facts or move detail into the linked memory files "
        "(one line per fact, detail lives in the file it points to)"
    )
