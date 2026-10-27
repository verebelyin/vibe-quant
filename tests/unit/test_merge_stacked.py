import importlib.util
import subprocess
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/merge_stacked.py"
_spec = importlib.util.spec_from_file_location("merge_stacked", _PATH)
assert _spec and _spec.loader
merge_stacked = importlib.util.module_from_spec(_spec)
sys.modules["merge_stacked"] = merge_stacked
_spec.loader.exec_module(merge_stacked)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def test_child_of_a_squash_landed_parent_merges_as_its_own_delta(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("base\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "parent")
    (repo / "a.txt").write_text("base\nparent\n")
    _git(repo, "commit", "-qam", "parent")
    _git(repo, "checkout", "-q", "-b", "child")
    (repo / "b.txt").write_text("child\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "child")
    _git(repo, "checkout", "-q", "main")  # parent lands squashed: same tree, new commit
    (repo / "a.txt").write_text("base\nparent\n")
    _git(repo, "commit", "-qam", "parent (squashed)")

    assert merge_stacked.merge_stacked("child", "parent", cwd=str(repo)) == 0

    assert (repo / "b.txt").read_text() == "child\n"
    assert (repo / "a.txt").read_text() == "base\nparent\n"
    assert _git(repo, "log", "-1", "--format=%s").strip() == "merge child (delta over parent)"
    assert _git(repo, "status", "--porcelain") == ""
