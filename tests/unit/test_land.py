"""scripts/agents/land.py: per-task commits from each fork point, tree == verified branch."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/land.py"
_spec = importlib.util.spec_from_file_location("land", _PATH)
assert _spec and _spec.loader
land = importlib.util.module_from_spec(_spec)
sys.modules["land"] = land  # @dataclass resolves its module through sys.modules
_spec.loader.exec_module(land)

_ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=_ENV
    ).stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)


def _job(tmp_path: Path, monkeypatch: object) -> tuple[Path, list[str]]:
    """main: a.py; t1 edits a.py; t2 (stacked on t1) adds b.py; main then gains c.py;
    the verified job branch merges t1 and t2 onto that newer main."""
    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)  # type: ignore[attr-defined]
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "a.py", "A = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-qb", "t1")
    _write(repo, "a.py", "A = 2\n")
    _git(repo, "commit", "-qam", "wip t1")
    _git(repo, "checkout", "-qb", "t2")
    _write(repo, "b.py", "B = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "wip t2")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "c.py", "C = 1\n")  # main moved on after the fork
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "main moves")
    _git(repo, "checkout", "-qb", "job")
    _git(repo, "merge", "-q", "--no-edit", "t2")
    _git(repo, "checkout", "-q", "main")
    msgs = []
    for name in ("t1", "t2"):
        msg = tmp_path / f"{name}.txt"
        msg.write_text(f"feat: {name}\n\nWhy:\nx\n")
        msgs.append(str(msg))
    return repo, msgs


def _adjacent_job(tmp_path: Path, monkeypatch: object) -> tuple[Path, list[str]]:
    """main: SPEC.md l1..l5; ta edits l2 and tb edits l3 (same fork, adjacent lines);
    main then moves on; the verified job branch merges ta and tb, resolving the
    SPEC.md conflict by hand."""
    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)  # type: ignore[attr-defined]
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "SPEC.md", "l1\nl2\nl3\nl4\nl5\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-qb", "ta")
    _write(repo, "SPEC.md", "l1\nl2a\nl3\nl4\nl5\n")
    _git(repo, "commit", "-qam", "wip ta")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "checkout", "-qb", "tb")
    _write(repo, "SPEC.md", "l1\nl2\nl3b\nl4\nl5\n")
    _git(repo, "commit", "-qam", "wip tb")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "moved.py", "M = 1\n")  # main moved on after the fork
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "main moves")
    _git(repo, "checkout", "-qb", "job")
    _git(repo, "merge", "-q", "--no-edit", "ta")
    merge = subprocess.run(
        ["git", "-C", str(repo), "merge", "--no-edit", "tb"],
        capture_output=True,
        text=True,
        env=_ENV,
    )
    assert merge.returncode != 0  # adjacent edits conflict
    _write(repo, "SPEC.md", "l1\nl2a\nl3b\nl4\nl5\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "merge tb")
    _git(repo, "checkout", "-q", "main")
    msgs = []
    for name in ("ta", "tb"):
        msg = tmp_path / f"{name}.txt"
        msg.write_text(f"feat: {name}\n\nWhy:\nx\n")
        msgs.append(str(msg))
    return repo, msgs


def test_conflict_on_a_path_no_later_task_touches_takes_the_verified_version(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    repo, (ma, mb) = _adjacent_job(tmp_path, monkeypatch)
    rc = land.main(["--repo", str(repo), "--job", "j", "--verified", "job", f"ta={ma}", f"tb={mb}"])
    assert rc == 0
    assert (repo / "SPEC.md").read_text() == "l1\nl2a\nl3b\nl4\nl5\n"
    assert _git(repo, "log", "--format=%s", "-3").splitlines() == [
        "feat: tb",
        "feat: ta",
        "main moves",
    ]
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "SPEC.md conflicted" in out and "tree equals job" in out


def test_conflict_on_a_path_a_later_task_touches_still_fails(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    repo, (ma, mb) = _adjacent_job(tmp_path, monkeypatch)
    _git(repo, "checkout", "-qb", "tc")  # a third sibling, landed after tb
    _write(repo, "SPEC.md", "l1\nl2\nl3\nl4\nl5c\n")
    _git(repo, "commit", "-qam", "wip tc")
    _git(repo, "checkout", "-q", "main")
    msg = tmp_path / "tc.txt"
    msg.write_text("feat: tc\n\nWhy:\nx\n")
    rc = land.main(
        [
            "--repo",
            str(repo),
            "--job",
            "j",
            "--verified",
            "job",
            f"ta={ma}",
            f"tb={mb}",
            f"tc={msg}",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "git apply --index --3way" in err and "main started at" in err


def test_a_non_conflict_apply_failure_still_fails(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    repo, (m1, _m2) = _job(tmp_path, monkeypatch)
    real_run = land.subprocess.run

    def fake_run(cmd: list[str], *args: object, **kwargs: object) -> object:
        if "apply" in cmd:  # a failure with nothing left unmerged
            return subprocess.CompletedProcess(cmd, 1, "", "patch does not apply")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(land.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    rc = land.main(["--repo", str(repo), "--job", "j", "--verified", "job", f"t1={m1}"])
    assert rc == 1
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "git apply --index --3way - failed: patch does not apply" in err


def test_a_resolved_conflict_is_still_checked_against_verified(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    repo, (ma, mb) = _adjacent_job(tmp_path, monkeypatch)
    _git(repo, "checkout", "-q", "job")
    _git(repo, "checkout", "-qb", "other")
    _write(repo, "extra.py", "E = 1\n")  # verified tree holds a file the job lacks
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "other")
    _git(repo, "checkout", "-q", "main")
    rc = land.main(
        ["--repo", str(repo), "--job", "j", "--verified", "other", f"ta={ma}", f"tb={mb}"]
    )
    assert rc == 1
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "tree differs from other" in err and "extra.py" in err


def test_lands_one_commit_per_task_and_keeps_mains_own_changes(
    tmp_path: Path, monkeypatch: object
) -> None:
    repo, (m1, m2) = _job(tmp_path, monkeypatch)
    rc = land.main(
        ["--repo", str(repo), "--job", "j", "--verified", "job", f"t1={m1}", f"t2@t1={m2}"]
    )
    assert rc == 0
    assert _git(repo, "log", "--format=%s", "-3").splitlines() == [
        "feat: t2",
        "feat: t1",
        "main moves",
    ]
    assert (repo / "c.py").read_text() == "C = 1\n"  # a two-dot diff would have deleted it
    assert (repo / "a.py").read_text() == "A = 2\n" and (repo / "b.py").read_text() == "B = 1\n"
    assert _git(repo, "show", "--stat", "--format=", "HEAD").count("|") == 1  # t2 = b.py only


def test_tree_mismatch_exits_1_and_names_the_start(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    repo, (m1, _m2) = _job(tmp_path, monkeypatch)
    start = _git(repo, "rev-parse", "HEAD")
    rc = land.main(
        ["--repo", str(repo), "--job", "j", "--verified", "job", f"t1={m1}"]
    )  # t2 missing
    assert rc == 1
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "b.py" in err and f"reset main to {start}" in err


def test_dirty_board_mirror_is_committed_first_but_other_changes_refuse(
    tmp_path: Path, monkeypatch: object
) -> None:
    repo, (m1, m2) = _job(tmp_path, monkeypatch)
    _write(repo, "docs/orchestration/board/messages.jsonl", "{}\n")
    rc = land.main(
        ["--repo", str(repo), "--job", "j", "--verified", "job", f"t1={m1}", f"t2@t1={m2}"]
    )
    assert rc == 0
    assert "orchestration: board mirror (job j)" in _git(repo, "log", "--format=%s")
    _write(repo, "stray.py", "S = 1\n")
    assert land.main(["--repo", str(repo), "--job", "j", "--verified", "job", f"t1={m1}"]) == 2


def test_refuses_off_main(tmp_path: Path, monkeypatch: object) -> None:
    repo, (m1, _m2) = _job(tmp_path, monkeypatch)
    _git(repo, "checkout", "-q", "t1")
    assert land.main(["--repo", str(repo), "--job", "j", "--verified", "job", f"t1={m1}"]) == 2
