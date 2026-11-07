"""Mutant trees (mutate.py, the gate's parity step) never reach the user's real data.

Job 20261107-indswitch: parity mutant `vibe_quant/data/archive.py:153` (`db_path or DEFAULT`
-> `and`) made every `RawDataArchive(tmp)` open the cwd-relative default archive, which in the
tree was a symlink to the real 8.6 GB archive; tests inserted 92 rows into it. A tree now gets
private copy-on-write copies of data/archive and data/catalog (or no data at all), and the
runners compare the real files' (size, mtime) before and after (fail closed).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"


def _load(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, _AGENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mutate = _load("mutate")
parity = _load("parity_mutants")

_REAL = Path(__file__).resolve().parents[2]
_WRITER_TEST = """\
from pathlib import Path

from vibe_quant.data.archive import RawDataArchive


def test_insert_into_private_archive(tmp_path: Path) -> None:
    archive = RawDataArchive(tmp_path / "a.db")
    archive.insert_funding_rates("BTCUSDT", [(1700000000000, 0.0001, 30000.0)], "t")
    archive.close()
"""
_PATH_DEFAULT_MUTANT = ("self._db_path = db_path or DEFAULT_ARCHIVE_PATH", "self._db_path = db_path and DEFAULT_ARCHIVE_PATH")


def _git(repo: Path, *args: str) -> None:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _sha(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def _make_repo(tmp_path: Path) -> Path:
    """A git repo holding the real archive.py + connection.py, with a fake 'real' data dir."""
    repo = tmp_path / "repo"
    for rel in ("vibe_quant/data/archive.py", "vibe_quant/db/connection.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REAL / rel, repo / rel)
    for rel in ("vibe_quant/__init__.py", "vibe_quant/data/__init__.py", "vibe_quant/db/__init__.py"):
        (repo / rel).write_text("")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_writer.py").write_text(_WRITER_TEST)
    (repo / ".gitignore").write_text("/data/\n")
    archive = repo / "data" / "archive" / "raw_data.db"
    archive.parent.mkdir(parents=True)
    with sqlite3.connect(archive) as conn:
        conn.execute("CREATE TABLE marker (x INTEGER)")
        conn.execute("INSERT INTO marker VALUES (1)")
    conn.close()
    (repo / "data" / "catalog").mkdir()
    (repo / "data" / "catalog" / "bars.parquet").write_bytes(b"bars" * 100)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return repo


@pytest.fixture(autouse=True)
def _plain_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def test_path_default_mutant_cannot_write_the_real_archive(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    archive = repo / "data" / "archive" / "raw_data.db"
    before = _sha(archive)
    old, new = _PATH_DEFAULT_MUTANT
    rc = mutate.main(
        [
            "--repo", str(repo), "--file", "vibe_quant/data/archive.py",
            "--old", old, "--new", new, "--", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        ]
    )  # fmt: skip
    assert _sha(archive) == before
    assert rc in (0, 1)  # the mutant ran (killed or survived), it did not error out


def test_a_symlinked_file_inside_the_data_dir_is_not_written_through(tmp_path: Path) -> None:
    """rev-gvc-1 B1: data/archive/raw_data.db itself a symlink to the real file."""
    repo = _make_repo(tmp_path)
    archive = repo / "data" / "archive" / "raw_data.db"
    outside = tmp_path / "real.db"
    archive.rename(outside)
    archive.symlink_to(outside)
    before = _sha(outside)
    old, new = _PATH_DEFAULT_MUTANT
    mutate.main(
        [
            "--repo", str(repo), "--file", "vibe_quant/data/archive.py",
            "--old", old, "--new", new, "--", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        ]
    )  # fmt: skip
    assert _sha(outside) == before


def test_a_nested_directory_symlink_leaves_the_data_dir_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    real = tmp_path / "real_bars"
    real.mkdir()
    (real / "a.parquet").write_bytes(b"real")
    (repo / "data" / "catalog").mkdir(parents=True)
    (repo / "data" / "catalog" / "bar").symlink_to(real)
    dest = tmp_path / "dest"
    dest.mkdir()
    mutate.clone_data_dirs(repo, dest)
    assert not os.path.lexists(dest / "data" / "catalog")
    assert "INFO data: data/catalog holds a symlink; not linked" in capsys.readouterr().out


def test_clone_tree_drops_a_data_dir_holding_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.db"
    real.write_bytes(b"real")
    src = tmp_path / "src"
    (src / "data" / "archive").mkdir(parents=True)
    (src / "data" / "archive" / "raw_data.db").symlink_to(real)
    dst = tmp_path / "dst"
    parity.clone_tree(src, dst)
    assert not os.path.lexists(dst / "data" / "archive")


def test_the_tree_gets_a_private_copy_not_a_link(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    mutate.build_copy(repo, tree)
    for rel in ("data/archive", "data/catalog"):
        assert not (tree / rel).is_symlink()
    assert (tree / "data/catalog/bars.parquet").read_bytes() == b"bars" * 100
    (tree / "data/archive/raw_data.db").write_bytes(b"scribble")
    (tree / "data/catalog/bars.parquet").write_bytes(b"scribble")
    assert (repo / "data/catalog/bars.parquet").read_bytes() == b"bars" * 100
    assert _sha(repo / "data/archive/raw_data.db")[0] != hashlib.sha256(b"scribble").hexdigest()


def test_extract_rev_gives_a_private_copy_too(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    mutate.extract_rev(repo, tree, "HEAD")
    assert not (tree / "data/archive").is_symlink()
    assert (tree / "data/archive/raw_data.db").is_file()


def test_without_copy_on_write_the_data_dir_is_absent_not_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(mutate, "cow_clone", lambda src, dst: False)
    tree = tmp_path / "tree"
    tree.mkdir()
    mutate.build_copy(repo, tree)
    assert not os.path.lexists(tree / "data/archive")
    assert not os.path.lexists(tree / "data/catalog")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("INFO data:")]
    assert lines == [
        "INFO data: no copy-on-write; data/catalog not linked",
        "INFO data: no copy-on-write; data/archive not linked",
    ]


def test_clone_tree_carries_no_data_symlink(tmp_path: Path) -> None:
    """A tree built by an older mutate.py (data symlinks) is cloned without them."""
    real = tmp_path / "real"
    (real / "archive").mkdir(parents=True)
    (real / "archive" / "raw_data.db").write_bytes(b"real")
    src = tmp_path / "src"
    (src / "data").mkdir(parents=True)
    (src / "data" / "archive").symlink_to(real / "archive")
    (src / "code.py").write_text("x = 1\n")
    dst = tmp_path / "dst"
    parity.clone_tree(src, dst)
    assert (dst / "code.py").read_text() == "x = 1\n"
    assert not os.path.lexists(dst / "data" / "archive")


def test_clone_tree_keeps_the_private_data_copy(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    mutate.build_copy(repo, tree)
    dst = tmp_path / "dst"
    parity.clone_tree(tree, dst)
    assert not (dst / "data/archive").is_symlink()
    assert (dst / "data/archive/raw_data.db").read_bytes() == (repo / "data/archive/raw_data.db").read_bytes()


def test_snapshot_sees_size_mtime_add_and_remove(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    before = mutate.snapshot_real_data(repo)
    assert len(before) == 2  # the archive file and one catalog file
    assert mutate.real_data_changes(before, repo) == []
    catalog = repo / "data/catalog/bars.parquet"
    st = catalog.stat()
    os.utime(catalog, ns=(st.st_atime_ns, st.st_mtime_ns + 1))  # mtime only
    assert mutate.real_data_changes(before, repo) == [str(catalog.resolve())]
    os.utime(catalog, ns=(st.st_atime_ns, st.st_mtime_ns))
    catalog.write_bytes(b"bars" * 100 + b"x")
    os.utime(catalog, ns=(st.st_atime_ns, st.st_mtime_ns))  # size only
    assert mutate.real_data_changes(before, repo) == [str(catalog.resolve())]
    catalog.write_bytes(b"bars" * 100)
    os.utime(catalog, ns=(st.st_atime_ns, st.st_mtime_ns))
    added = repo / "data/catalog/new.parquet"
    added.write_bytes(b"n")
    assert mutate.real_data_changes(before, repo) == [str(added.resolve())]
    added.unlink()
    (repo / "data/archive/raw_data.db").unlink()
    assert mutate.real_data_changes(before, repo) == [str((repo / "data/archive/raw_data.db").resolve())]


def test_mutate_main_fails_closed_when_the_real_data_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    archive = repo / "data/archive/raw_data.db"

    def leaky_run(cmd: object, cwd: object, env: object, timeout: object) -> object:
        with archive.open("ab") as handle:  # what a symlinked tree used to do
            handle.write(b"leak")
        return mutate.RunResult(0, 0.01)

    old, new = _PATH_DEFAULT_MUTANT
    rc = mutate.main(
        ["--repo", str(repo), "--file", "vibe_quant/data/archive.py", "--old", old, "--new", new, "--", "true"],
        run=leaky_run,
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert f"REAL DATA CHANGED: {archive.resolve()}" in out


def test_mutate_main_is_quiet_when_the_real_data_is_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    old, new = _PATH_DEFAULT_MUTANT
    rc = mutate.main(
        ["--repo", str(repo), "--file", "vibe_quant/data/archive.py", "--old", old, "--new", new, "--", "true"],
        run=lambda cmd, cwd, env, timeout: mutate.RunResult(0, 0.01),
    )
    assert "REAL DATA CHANGED" not in capsys.readouterr().out
    assert rc == 1  # survived: `true` kills nothing


@pytest.mark.slow(reason="copies the repo tree and runs a catalog-backed test file in it")
def test_catalog_backed_test_is_green_inside_a_mutant_tree(tmp_path: Path) -> None:
    """The private copy still carries the real catalog: a read-only research test runs there."""
    repo = _REAL
    if not (repo / "data/catalog").exists():
        pytest.skip("no data/catalog in this checkout")
    tree = tmp_path / "tree"
    tree.mkdir()
    mutate.build_copy(repo, tree)
    assert not (tree / "data/catalog").is_symlink()
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "tests/unit/test_screening_pta_precompute.py"],
        cwd=tree, env=env, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-500:]
