"""Tests for scripts/agents/mkbrief.py — the brief generator.

The CLI is driven end-to-end against throwaway git repos in tmp_path (glob
validation walks the --worktree filesystem; review mode runs real `git
merge-base`/`git rev-parse`), so no real project checkout is ever touched.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
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


def _build_repo(repo: Path) -> Path:
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


_REPO_TEMPLATE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _repo_template_fixture(tmp_path_factory: pytest.TempPathFactory) -> None:
    """One committed base repo; each test gets a private copy (8ms) instead of a fresh
    `git init`+`add`+`commit` (131ms). Tests then branch/commit on their own copy."""
    global _REPO_TEMPLATE
    _REPO_TEMPLATE = _build_repo(tmp_path_factory.mktemp("mkbrief-template") / "wt")


def _repo(tmp_path: Path, name: str = "wt") -> Path:
    assert _REPO_TEMPLATE is not None, "the module template fixture did not run"
    repo = tmp_path / name
    shutil.copytree(_REPO_TEMPLATE, repo)
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


def test_scope_path_the_task_deletes_exists_at_base(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    (repo / "old.py").write_text("OLD = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add old")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "rm", "-q", "old.py")
    _git(repo, "commit", "-qm", "delete old")
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(
        _argv(job, template, scope="a.py,old.py", worktree=repo, extra=["--base", base])
    )

    assert rc == 0
    capsys.readouterr()
    assert "SCOPE: a.py,old.py" in (job / "prompts" / "cc-rd-demo-1.md").read_text()


def test_scope_path_named_in_deletes_is_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(
        _argv(
            job, template, scope="a.py,nope/deleted_test.py", worktree=repo,
            extra=["--base", base, "--deletes", "nope/deleted_test.py"],
        )
    )

    assert rc == 0
    capsys.readouterr()
    assert "SCOPE: a.py,nope/deleted_test.py" in (job / "prompts" / "cc-rd-demo-1.md").read_text()


def test_deleted_scope_path_does_not_mask_a_missing_glob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    (repo / "old.py").write_text("OLD = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add old")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "rm", "-q", "old.py")
    _git(repo, "commit", "-qm", "delete old")
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(
        _argv(job, template, scope="a.py,old.py,nope/*.py", worktree=repo, extra=["--base", base])
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "glob nope/*.py matches no file" in err
    assert "old.py" not in err


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


def test_review_defaults_parent_to_the_worktree_vq_base(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    fork = _git(repo, "rev-parse", "HEAD")
    (repo / ".vq-base").write_text(fork + "\n")
    _git(repo, "checkout", "-qb", "feat")
    (repo / "feat.py").write_text("FEAT = 1\n")
    _git(repo, "add", "feat.py")
    _git(repo, "commit", "-qm", "feat work")
    head = _git(repo, "rev-parse", "HEAD")

    rc = mkbrief.main(
        _argv(_job(tmp_path), _template(tmp_path, _REVIEW_TEMPLATE), worktree=repo,
              extra=["--review", "feat"])
    )

    assert rc == 0
    capsys.readouterr()
    text = (tmp_path / "job" / "prompts" / "cc-rd-demo-1.md").read_text()
    assert f"RANGE: {fork}..{head}" in text


def _merged_parent_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """feat forks at `fork`, main moves on, feat merges main: returns (repo, fork, main_tip)."""
    repo = _repo(tmp_path)
    fork = _git(repo, "rev-parse", "HEAD")
    (repo / ".vq-base").write_text(fork + "\n")
    _git(repo, "checkout", "-qb", "feat")
    (repo / "feat.py").write_text("FEAT = 1\n")
    _git(repo, "add", "feat.py")
    _git(repo, "commit", "-qm", "feat work")
    _git(repo, "checkout", "-q", "main")
    (repo / "main.py").write_text("MAIN = 1\n")
    _git(repo, "add", "main.py")
    _git(repo, "commit", "-qm", "main moves")
    main_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "feat")
    _git(repo, "merge", "-q", "--no-edit", "main")
    return repo, fork, main_tip


def test_review_default_parent_refuses_a_branch_that_merged_its_parent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _fork, _tip = _merged_parent_repo(tmp_path)
    job = _job(tmp_path)
    rc = mkbrief.main(
        _argv(job, _template(tmp_path, _REVIEW_TEMPLATE), worktree=repo, extra=["--review", "feat"])
    )
    assert rc == 2
    assert "merged its parent" in capsys.readouterr().err
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_explicit_parent_wins_over_the_recorded_vq_base(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, fork, main_tip = _merged_parent_repo(tmp_path)
    head = _git(repo, "rev-parse", "feat")
    rc = mkbrief.main(
        _argv(_job(tmp_path), _template(tmp_path, _REVIEW_TEMPLATE), worktree=repo,
              extra=["--review", "feat", "--parent", "main"])
    )
    assert rc == 0
    capsys.readouterr()
    text = (tmp_path / "job" / "prompts" / "cc-rd-demo-1.md").read_text()
    assert f"RANGE: {main_tip}..{head}" in text
    assert fork not in text


def test_review_default_parent_names_a_vq_base_that_is_not_an_ancestor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "sibling")
    (repo / "sib.py").write_text("S = 1\n")
    _git(repo, "add", "sib.py")
    _git(repo, "commit", "-qm", "sibling")
    (repo / ".vq-base").write_text(_git(repo, "rev-parse", "HEAD") + "\n")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "checkout", "-qb", "feat")
    (repo / "feat.py").write_text("FEAT = 1\n")
    _git(repo, "add", "feat.py")
    _git(repo, "commit", "-qm", "feat")
    rc = mkbrief.main(
        _argv(_job(tmp_path), _template(tmp_path, _REVIEW_TEMPLATE), worktree=repo, extra=["--review", "feat"])
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert ".vq-base is not an ancestor of feat; pass --parent" in err
    assert "given together" not in err


def test_review_without_parent_and_without_vq_base_still_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    rc = mkbrief.main(
        _argv(_job(tmp_path), _template(tmp_path, _REVIEW_TEMPLATE), worktree=repo,
              extra=["--review", "main"])
    )
    assert rc == 2
    assert "--review and --parent must be given together" in capsys.readouterr().err


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


def _gate(brief: Path) -> str:
    return next(ln for ln in brief.read_text().splitlines() if ln.startswith("GATE: "))


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
    main_sha = _git(repo, "rev-parse", "main")
    assert _gate(job / "prompts" / "cc-rd-demo-1.md") == (
        f"GATE: scripts/agents/swarm-check.sh {repo} --scope 'a.py'"
        f" --base {main_sha} --bus {job / 'bus'} --bus-agent cc-rd-demo-1"
        f"  # base main @ render"
    )


def test_gate_base_branch_resolves_to_sha_at_render_time(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    first = _commit(repo, "a1.py")
    job = _job(tmp_path)
    template = _template(tmp_path)
    brief = job / "prompts" / "cc-rd-demo-1.md"

    rc = mkbrief.main(_argv(job, template, worktree=repo, extra=["--base", "swarm/a"]))

    assert rc == 0
    assert _gate(brief) == (
        f"GATE: scripts/agents/swarm-check.sh {repo} --scope 'a.py'"
        f" --base {first}  # base swarm/a @ render"
    )

    moved = _commit(repo, "a2.py")  # the branch moves on after the brief was written
    assert moved != first
    assert f"--base {first}" in brief.read_text()
    assert moved not in brief.read_text()

    # A brief rendered now pins the branch's new tip, not the frozen one.
    rc = mkbrief.main(
        _argv(job, template, request_id="cc-rd-demo-2", worktree=repo, extra=["--base", "swarm/a"])
    )
    assert rc == 0
    assert _gate(job / "prompts" / "cc-rd-demo-2.md") == (
        f"GATE: scripts/agents/swarm-check.sh {repo} --scope 'a.py'"
        f" --base {moved}  # base swarm/a @ render"
    )


def test_review_mode_skips_the_branch_scope_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "branch", "parent")
    _git(repo, "checkout", "-qb", "feat")
    base = _git(repo, "rev-parse", "parent")
    _commit(repo, "out_of_scope.py")
    assert "out_of_scope.py" in _git(repo, "diff", "--name-only", f"{base}...HEAD")
    job = _job(tmp_path)

    rc = mkbrief.main(
        _argv(
            job,
            _template(tmp_path, _REVIEW_TEMPLATE),
            scope="a.py",
            worktree=repo,
            extra=["--review", "feat", "--parent", "parent", "--base", base],
        )
    )

    assert rc == 0
    capsys.readouterr()
    assert (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_gate_base_sha_input_omits_redundant_comment(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    job = _job(tmp_path)

    rc = mkbrief.main(_argv(job, _template(tmp_path), worktree=repo, extra=["--base", sha]))

    assert rc == 0
    # The given ref already IS the resolved sha: no "# base ... @ render" comment.
    assert _gate(job / "prompts" / "cc-rd-demo-1.md") == (
        f"GATE: scripts/agents/swarm-check.sh {repo} --scope 'a.py' --base {sha}"
    )
    assert "@ render" not in (job / "prompts" / "cc-rd-demo-1.md").read_text()


def test_branch_change_outside_scope_exits_2_and_lists_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "swarm/fix")
    _commit(repo, "in_scope.py")
    _commit(repo, "out_of_scope.py")
    job = _job(tmp_path)

    rc = mkbrief.main(
        _argv(
            job,
            _template(tmp_path),
            scope="in_scope.py,a.py",
            worktree=repo,
            extra=["--base", base],
        )
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "out_of_scope.py" in err
    assert "matches no --scope glob" in err
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_branch_changes_all_in_scope_render(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "swarm/fix")
    _commit(repo, "in_scope.py")
    job = _job(tmp_path)

    rc = mkbrief.main(
        _argv(job, _template(tmp_path), scope="in_scope.py", worktree=repo, extra=["--base", base])
    )

    assert rc == 0
    capsys.readouterr()
    assert (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_branch_scope_check_is_skipped_without_scope_globs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "swarm/fix")
    _commit(repo, "b.py")
    job = _job(tmp_path)

    # No --scope: the gate SKIPs its own scope check, so the branch check must too.
    rc = mkbrief.main(
        _argv(job, _template(tmp_path), scope=None, worktree=repo, extra=["--base", base])
    )

    assert rc == 0
    capsys.readouterr()
    assert (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_board_mirror_change_is_never_a_scope_offender(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "swarm/fix")
    _commit(repo, "in_scope.py")
    mirror = repo / "docs" / "orchestration" / "board"
    mirror.mkdir(parents=True)
    (mirror / "messages.jsonl").write_text("{}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "bus mirror")
    job = _job(tmp_path)

    rc = mkbrief.main(
        _argv(job, _template(tmp_path), scope="in_scope.py", worktree=repo, extra=["--base", base])
    )

    assert rc == 0
    capsys.readouterr()
    assert (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_scope_check_is_skipped_without_a_base(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    _commit(repo, "a1.py")
    job = _job(tmp_path)

    # No --base: the branch-vs-scope check does not run, so an unchecked branch renders.
    rc = mkbrief.main(_argv(job, _template(tmp_path), scope="a.py", worktree=repo))

    assert rc == 0
    capsys.readouterr()
    assert (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_unresolvable_base_exits_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(_argv(job, template, worktree=repo, extra=["--base", "no-such-ref"]))

    assert rc == 2
    err = capsys.readouterr().err
    assert "no-such-ref" in err
    assert "does not resolve" in err
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_base_is_left_verbatim_without_a_worktree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = _job(tmp_path)

    rc = mkbrief.main(_argv(job, _template(tmp_path), extra=["--base", "main"]))

    assert rc == 0
    capsys.readouterr()
    assert _gate(job / "prompts" / "cc-rd-demo-1.md") == (
        "GATE: scripts/agents/swarm-check.sh  --scope 'a.py' --base main"
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
    assert "stacked worktree without .vq-base: pass --base <fork point>" in err
    assert a_tip in err
    assert not (tmp_path / "job" / "prompts" / "cc-rd-demo-1.md").exists()


def test_stacked_worktree_with_base_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    a_tip = _commit(repo, "a1.py")
    _git(repo, "checkout", "-qb", "swarm/b")
    _commit(repo, "b1.py")
    argv = _argv(
        _job(tmp_path), _template(tmp_path), scope="a.py,b1.py", worktree=repo, extra=["--base", a_tip]
    )
    assert mkbrief.main(argv) == 0


def test_stacked_worktree_takes_base_from_vq_base(tmp_path: Path) -> None:
    """worktree.sh records the fork point in .vq-base; a stacked render needs no --base then."""
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    a_tip = _commit(repo, "a1.py")
    _git(repo, "checkout", "-qb", "swarm/b")
    _commit(repo, "b1.py")
    (repo / ".vq-base").write_text(a_tip + "\n")
    job = _job(tmp_path)
    assert mkbrief.main(_argv(job, _template(tmp_path), scope="a.py,b1.py", worktree=repo)) == 0
    assert f"--base {a_tip}" in (job / "prompts" / "cc-rd-demo-1.md").read_text()


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


def test_bus_block_peers_defaults_to_none(tmp_path: Path) -> None:
    """Clerks had to pass --set PEERS=none on every render; an explicit value still wins."""
    root = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "prompts"
    job = _job(tmp_path)
    repo = _repo(tmp_path)
    argv = _argv(job, _template(tmp_path), worktree=repo) + ["--append", str(root / "bus-protocol.md")]
    assert mkbrief.main(argv) == 0
    assert "Peers now: none" in (job / "prompts" / "cc-rd-demo-1.md").read_text()
    argv = _argv(job, _template(tmp_path), request_id="cc-demo-2", worktree=repo)
    argv += ["--append", str(root / "bus-protocol.md"), "--set", "PEERS=cc-x-1"]
    assert mkbrief.main(argv) == 0
    assert "Peers now: cc-x-1" in (job / "prompts" / "cc-demo-2.md").read_text()


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


def test_scope_path_differing_only_by_case_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """T6 listed sizingChip.test.ts while the file is SizingChip.test.ts (gate FAIL + an ask)."""
    repo = _repo(tmp_path)
    (repo / "SizingChip.test.ts").write_text("export const x = 1\n")
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(_argv(job, template, scope="a.py,+sizingChip.test.ts", worktree=repo))

    assert rc == 2
    assert "sizingChip.test.ts differs only by case from SizingChip.test.ts" in capsys.readouterr().err


def _deleting_repo(tmp_path: Path) -> Path:
    repo = _repo(tmp_path)
    (repo / "a.py").write_text(
        "def _startup_halt(x):\n    return x\n\n\ndef keep_me(x):\n    return x\n"
    )
    (repo / "tests" / "unit" / "test_pins.py").write_text("from a import _startup_halt\n")
    (repo / "tests" / "unit" / "test_other.py").write_text("from a import keep_me\n")
    return repo


def test_deleted_name_used_by_a_test_outside_scope_fails_rendering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _deleting_repo(tmp_path)
    job = _job(tmp_path)

    rc = mkbrief.main(
        _argv(job, _template(tmp_path), worktree=repo, extra=["--deletes", "_startup_halt"])
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "tests/unit/test_pins.py uses deleted _startup_halt" in err
    assert (
        "test_other.py" not in err
    )  # only exact-name users are listed, not every test on a scoped symbol
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def test_deleted_name_test_in_scope_or_accepted_renders_without_the_old_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _deleting_repo(tmp_path)
    job = _job(tmp_path)
    extra = ["--deletes", "_startup_halt"]

    rc = mkbrief.main(
        _argv(
            job,
            _template(tmp_path),
            scope="a.py,tests/unit/test_pins.py",
            worktree=repo,
            extra=extra,
        )
    )
    assert rc == 0
    rc = mkbrief.main(
        _argv(job, _template(tmp_path), request_id="cc-rd-demo-2", worktree=repo,
              extra=[*extra, "--accept-outside", "tests/unit/test_pins.py"])
    )  # fmt: skip
    assert rc == 0
    assert "outside --scope" not in capsys.readouterr().err


def test_facts_delete_line_names_the_deleted_symbol(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _deleting_repo(tmp_path)
    job = _job(tmp_path)
    facts = tmp_path / "facts.md"
    facts.write_text("Plan: delete `_startup_halt` and rewire callers.\n")

    rc = mkbrief.main(
        _argv(job, _template(tmp_path), worktree=repo, extra=["--set-file", f"FACTS={facts}"])
    )

    assert rc == 2
    assert "tests/unit/test_pins.py uses deleted _startup_halt" in capsys.readouterr().err


def test_no_deletes_means_no_outside_scope_noise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _deleting_repo(tmp_path)

    rc = mkbrief.main(_argv(_job(tmp_path), _template(tmp_path), worktree=repo))

    assert rc == 0
    assert "outside --scope" not in capsys.readouterr().err


def _hygiene_repo(tmp_path: Path, body: str) -> Path:
    repo = _repo(tmp_path)
    (repo / "a.py").write_text(body)
    src = (
        Path(mkbrief.__file__).resolve().parents[2] / "tests" / "unit" / "test_exception_hygiene.py"
    )
    (repo / "tests" / "unit" / "test_exception_hygiene.py").write_text(src.read_text())
    return repo


def test_moved_silent_handler_adds_exception_hygiene_to_scope(tmp_path: Path) -> None:
    repo = _hygiene_repo(
        tmp_path,
        "def _poll(x):\n    try:\n        return int(x)\n    except ValueError:\n        pass\n",
    )
    job = _job(tmp_path)

    rc = mkbrief.main(_argv(job, _template(tmp_path), worktree=repo, extra=["--deletes", "_poll"]))

    assert rc == 0
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    assert "SCOPE: a.py,tests/unit/test_exception_hygiene.py" in text


def test_moved_clean_symbol_leaves_scope_alone(tmp_path: Path) -> None:
    repo = _hygiene_repo(tmp_path, "def _poll(x):\n    return x\n")
    job = _job(tmp_path)

    rc = mkbrief.main(_argv(job, _template(tmp_path), worktree=repo, extra=["--deletes", "_poll"]))

    assert rc == 0
    assert "SCOPE: a.py\n" in (job / "prompts" / "cc-rd-demo-1.md").read_text()


def test_review_with_an_empty_range_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)

    rc = mkbrief.main(
        _argv(job, _template(tmp_path, _REVIEW_TEMPLATE), worktree=repo,
              extra=["--review", "main", "--parent", "main"])
    )  # fmt: skip

    assert rc == 2
    assert "empty review range" in capsys.readouterr().err
    assert not (job / "prompts" / "cc-rd-demo-1.md").exists()


def _models_json(tmp_path: Path, maker_enabled: bool = True) -> Path:
    import json

    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "models": {
            "flash": {"enabled": maker_enabled, "runtime": "t3",
                      "target": {"providerInstanceId": "cc", "model": "ds-flash"}},
            "sonnet": {"enabled": True, "runtime": "claude", "target": {"subagent_type": "implementer"}},
        },
        "routes": {"maker": ["flash"], "fixer": ["sonnet"]},
    }))  # fmt: skip
    return path


def test_route_prints_the_delegate_call_and_appends_a_ledger_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.setenv("VQ_MODELS_JSON", str(_models_json(tmp_path)))
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main([*_argv(job, template, scope="a.py", worktree=repo), "--route", "maker", "--bead", "vq-1"])

    assert rc == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("DELEGATE_TASK "))
    call = json.loads(line.removeprefix("DELEGATE_TASK "))
    assert call["clientRequestId"] == "cc-rd-demo-1" and call["mode"] == "async"
    assert call["target"] == {"providerInstanceId": "cc", "model": "ds-flash"}
    assert call["task"].startswith("Read your brief at ") and "TELEMETRY: task=rd-demo round=1" in call["task"]
    assert json.loads((job / "prompts" / "cc-rd-demo-1.delegate.json").read_text()) == call
    rows = (job / "ledger.md").read_text().splitlines()
    assert rows[1] == "| agent | runtime | model | task | bead | worktree | status |"
    assert rows[-1].startswith(f"| cc-rd-demo-1 | t3 | flash | rd-demo | vq-1 | {repo} | ")
    assert rows[-1].endswith(" rendered |")


def test_route_with_no_available_model_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VQ_MODELS_JSON", str(_models_json(tmp_path, maker_enabled=False)))
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    rc = mkbrief.main([*_argv(job, _template(tmp_path), scope="a.py", worktree=repo), "--route", "maker"])
    assert rc == 2
    assert "route maker: none available (flash: off)" in capsys.readouterr().err
    assert not (job / "ledger.md").exists()


def test_route_with_an_unknown_role_lists_the_valid_roles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VQ_MODELS_JSON", str(_models_json(tmp_path)))
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    rc = mkbrief.main([*_argv(job, _template(tmp_path), scope="a.py", worktree=repo), "--route", "reviewer"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "route reviewer: unknown role; valid roles: maker, fixer" in err
    assert "none available" not in err
