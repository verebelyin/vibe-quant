"""Tests for scripts/agents/mkbrief.py — the brief generator.

The CLI is driven end-to-end against throwaway git repos in tmp_path (glob
validation walks the --worktree filesystem; review mode runs real `git
merge-base`/`git rev-parse`), so no real project checkout is ever touched.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "mkbrief.py"
_spec = importlib.util.spec_from_file_location("mkbrief", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mkbrief = importlib.util.module_from_spec(_spec)
sys.modules["mkbrief"] = mkbrief
_spec.loader.exec_module(mkbrief)

# Hermetic, non-interactive git so a throwaway repo needs no config and no hooks.
_GIT_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "mkbrief",
    "GIT_AUTHOR_EMAIL": "mkbrief@test",
    "GIT_COMMITTER_NAME": "mkbrief",
    "GIT_COMMITTER_EMAIL": "mkbrief@test",
}

_TEMPLATE = """# Task: {TASK}
{TELEMETRY}

ID: {ID}
JOB: {JOB}
BUS: {BUS}
WT: {WT}
RUN_AS: {RUN_AS}
SCOPE: {SCOPE}
GATE: {GATE}
"""

_REVIEW_TEMPLATE = """# Review: {TASK}
{TELEMETRY}
BASE_SHA: {BASE_SHA}
HEAD_SHA: {HEAD_SHA}
RANGE: {RANGE}
REVIEW_PACK: {REVIEW_PACK}
"""


def _git(repo: Path, *args: str) -> str:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV)
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    )
    return out.stdout.strip()


def _repo(tmp_path: Path, name: str = "wt") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    (repo / "tests" / "unit" / "sub").mkdir(parents=True)
    (repo / "tests" / "unit" / "test_x.py").write_text("def test_x() -> None:\n    assert True\n")
    (repo / "tests" / "unit" / "sub" / "test_y.py").write_text(
        "def test_y() -> None:\n    assert True\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    (job / "prompts").mkdir(parents=True)
    return job


def _template(tmp_path: Path, text: str = _TEMPLATE, name: str = "t.md") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _argv(
    job: Path,
    template: Path,
    *,
    request_id: str = "cc-rd-demo-1",
    scope: str = "a.py",
    worktree: Path | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    args = ["--job", str(job), "--id", request_id, "--template", str(template)]
    if worktree is not None:
        args += ["--worktree", str(worktree)]
    if scope is not None:
        args += ["--scope", scope]
    return args + (extra or [])


@pytest.mark.parametrize(
    ("client_id", "task", "round_no"),
    [
        ("cc-rd-mkbrief-1", "rd-mkbrief", 1),
        ("rev-foo-2", "foo", 2),
        ("impl-bar", "bar", 1),
        ("ver-baz-3", "baz", 3),
        ("plain", "plain", 1),
    ],
)
def test_parse_id_splits_prefix_and_round(client_id: str, task: str, round_no: int) -> None:
    assert mkbrief.parse_id(client_id) == (task, round_no)


def test_dispatch_text_carries_telemetry_and_file_exists_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(_argv(job, template, request_id="cc-rd-demo-1", scope="a.py", worktree=repo))

    out_path = job / "prompts" / "cc-rd-demo-1.md"
    assert rc == 0
    assert out_path.exists()
    assert capsys.readouterr().out == (
        f"Read your brief at {out_path.resolve()} and follow it exactly.\n"
        "TELEMETRY: task=rd-demo round=1\n"
    )
    assert "TELEMETRY: task=rd-demo round=1" in out_path.read_text()


def test_unfilled_placeholder_exits_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path, "# Task {WHO}\n")

    rc = mkbrief.main(_argv(job, template, worktree=repo))

    assert rc == 2
    assert "WHO" in capsys.readouterr().err
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_scope_glob_matching_nothing_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(_argv(job, template, scope="nope/*.py", worktree=repo))

    assert rc == 2
    assert "nope/*.py" in capsys.readouterr().err
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_new_file_glob_is_allowed_and_plus_stripped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(
        _argv(job, template, scope="a.py,+scripts/agents/new.py", worktree=repo)
    )

    assert rc == 0
    capsys.readouterr()
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    assert "SCOPE: a.py,scripts/agents/new.py" in text
    assert "+scripts/agents/new.py" not in text


def test_broad_unit_glob_flags_missing_top_level_glob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(_argv(job, template, scope="tests/unit/**/*.py", worktree=repo))

    assert rc == 2
    err = capsys.readouterr().err
    assert "glob tests/unit/**/*.py does not match top-level tests/unit files" in err
    assert "add tests/unit/*.py" in err


def test_review_mode_pins_merge_base_after_parent_moves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    fork = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "parent")

    _git(repo, "checkout", "-qb", "feat")
    (repo / "feat.py").write_text("FEAT = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat work")
    head = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-q", "parent")
    (repo / "parent.py").write_text("PARENT = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "parent moves on")
    parent_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "feat")

    job = _job(tmp_path)
    template = _template(tmp_path, _REVIEW_TEMPLATE)
    rc = mkbrief.main(
        _argv(
            job,
            template,
            scope="a.py",
            worktree=repo,
            extra=["--review", "feat", "--parent", "parent"],
        )
    )

    assert rc == 0
    capsys.readouterr()
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    assert f"BASE_SHA: {fork}" in text
    assert f"HEAD_SHA: {head}" in text
    assert f"RANGE: {fork}..{head}" in text
    assert f"REVIEW_PACK: python3 scripts/agents/review_pack.py {repo} --range {fork}..{head}" in text
    assert parent_tip not in text


def test_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)
    argv = _argv(job, template, worktree=repo)

    assert mkbrief.main(argv) == 0
    original = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    capsys.readouterr()

    rc = mkbrief.main(argv)
    assert rc == 2
    assert "exists" in capsys.readouterr().err
    assert (job / "prompts" / "cc-rd-demo-1.md").read_text() == original

    assert mkbrief.main(argv + ["--force"]) == 0
    capsys.readouterr()


def test_gate_command_includes_base_and_bus_when_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    (job / "bus").mkdir()
    template = _template(tmp_path)

    rc = mkbrief.main(
        _argv(
            job,
            template,
            scope="a.py,b.py",
            worktree=repo,
            extra=["--base", "main"],
        )
    )

    assert rc == 2  # b.py does not exist -> glob problem, but the gate is still rendered
    capsys.readouterr()
    # A valid scope with a bus present: only then is the rendered GATE checked.
    rc = mkbrief.main(_argv(job, template, scope="a.py", worktree=repo, extra=["--base", "main"]))
    assert rc == 0
    capsys.readouterr()
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    gate = next(ln for ln in text.splitlines() if ln.startswith("GATE: "))
    assert gate == (
        f"GATE: scripts/agents/swarm-check.sh {repo} --scope 'a.py'"
        f" --base main --bus {job / 'bus'} --bus-agent cc-rd-demo-1"
    )


def test_set_value_containing_placeholder_is_kept_literally(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path, "# Task: {TASK}\nFACTS:\n{FACTS}\n")
    facts = tmp_path / "facts.md"
    facts.write_text('Use `f"{RUN_ID}"` in the log line.\n')

    rc = mkbrief.main(_argv(job, template, worktree=repo, extra=["--set-file", f"FACTS={facts}"]))

    assert rc == 0
    capsys.readouterr()
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    assert 'f"{RUN_ID}"' in text


def test_unfilled_placeholder_in_appended_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)
    extra = tmp_path / "rules.md"
    extra.write_text("RULE {MISSING}\n")

    rc = mkbrief.main(_argv(job, template, worktree=repo, extra=["--append", str(extra)]))

    assert rc == 2
    assert "MISSING" in capsys.readouterr().err
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_leading_html_comment_stripped_from_template_and_appends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path, "<!--\nusage notes\n-->\n# Task: {TASK}\n")
    rules = tmp_path / "rules.md"
    rules.write_text("<!-- rules comment -->\nBODY {TASK}\n")

    rc = mkbrief.main(_argv(job, template, worktree=repo, extra=["--append", str(rules)]))

    assert rc == 0
    capsys.readouterr()
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    assert "usage notes" not in text
    assert "rules comment" not in text
    assert text.startswith("# Task: rd-demo")
    assert "BODY rd-demo" in text


def test_set_file_and_append_are_rendered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path, "# Task: {TASK}\nBLOCK:\n{CONTRACT}\n")
    contract = tmp_path / "contract.md"
    contract.write_text("- row one\n")
    rules = tmp_path / "rules.md"
    rules.write_text("WT={WT}\n")

    rc = mkbrief.main(
        _argv(
            job,
            template,
            scope="a.py",
            worktree=repo,
            extra=["--set-file", f"CONTRACT={contract}", "--append", str(rules)],
        )
    )

    assert rc == 0
    capsys.readouterr()
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    assert "- row one" in text
    assert text.rstrip().endswith(f"WT={repo}")


def _commit(repo: Path, fname: str) -> str:
    (repo / fname).write_text("X = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", fname)
    return _git(repo, "rev-parse", "HEAD")


def test_stacked_worktree_without_base_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    a_tip = _commit(repo, "a1.py")
    _git(repo, "checkout", "-qb", "swarm/b")
    _commit(repo, "b1.py")
    rc = mkbrief.main(_argv(_job(tmp_path), _template(tmp_path), worktree=repo))
    err = capsys.readouterr().err
    assert rc == 2
    assert "stacked worktree: pass --base <fork point>" in err
    assert a_tip in err
    assert not (tmp_path / "job" / "prompts" / "cc-rd-demo-1.md").exists()


def test_stacked_worktree_with_base_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    a_tip = _commit(repo, "a1.py")
    _git(repo, "checkout", "-qb", "swarm/b")
    _commit(repo, "b1.py")
    argv = _argv(_job(tmp_path), _template(tmp_path), worktree=repo, extra=["--base", a_tip])
    assert mkbrief.main(argv) == 0


def test_merge_commit_branch_without_base_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "side")
    _commit(repo, "s1.py")
    _git(repo, "checkout", "-qb", "feat", "main")
    _commit(repo, "f1.py")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    _git(repo, "branch", "-q", "-D", "side")
    rc = mkbrief.main(_argv(_job(tmp_path), _template(tmp_path), worktree=repo))
    assert rc == 2
    assert "stacked worktree" in capsys.readouterr().err


def test_plain_worktree_cut_from_older_main_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "feat")
    _commit(repo, "f1.py")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "m2.py")  # main moved on; feat is still a plain branch from main
    _git(repo, "checkout", "-q", "feat")
    assert mkbrief.main(_argv(_job(tmp_path), _template(tmp_path), worktree=repo)) == 0


def test_fix_task_template_renders_with_findings_and_bus_block(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "prompts"
    job = _job(tmp_path)
    findings = tmp_path / "findings.md"
    findings.write_text("F1: off-by-one at {EDGE} in a.py\n")
    argv = _argv(job, root / "fix-task.md", request_id="impl-demo-2", worktree=_repo(tmp_path))
    argv += ["--set-file", f"PRIOR_FINDINGS={findings}", "--set", "RULINGS=none", "--set", "PEERS=none"]
    argv += ["--append", str(root / "bus-protocol.md")]
    assert mkbrief.main(argv) == 0
    text = (job / "prompts" / "impl-demo-2.md").read_text()
    assert "round 2" in text
    assert "F1: off-by-one at {EDGE} in a.py" in text
    assert "SWARM BUS" in text
    assert "swarm-check.sh" in text


def test_child_branch_at_parent_tip_without_commits_is_stacked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    a_tip = _commit(repo, "a1.py")
    _git(repo, "checkout", "-qb", "swarm/b")  # no commits of its own
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "m2.py")  # main advanced too
    _git(repo, "checkout", "-q", "swarm/b")
    rc = mkbrief.main(_argv(_job(tmp_path), _template(tmp_path), worktree=repo))
    err = capsys.readouterr().err
    assert rc == 2
    assert f"swarm/a {a_tip}" in err
    assert "swarm/b " not in err  # the worktree's own branch is never a candidate
