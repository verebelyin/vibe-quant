"""swarm_check's caches: the per-process cache of the tree-wide scans (graph, pytest_plugins,
loaders) and the per-test-file PASS cache shared across worktrees."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "swarm_check.py"
if "swarm_check" in sys.modules:
    swarm_check = sys.modules["swarm_check"]
else:
    _spec = importlib.util.spec_from_file_location("swarm_check", _SCRIPT)
    assert _spec is not None and _spec.loader is not None
    swarm_check = importlib.util.module_from_spec(_spec)
    sys.modules["swarm_check"] = swarm_check
    _spec.loader.exec_module(swarm_check)


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "vibe_quant").mkdir()
    (tmp_path / "vibe_quant" / "a.py").write_text("import vibe_quant.b\n")
    (tmp_path / "vibe_quant" / "b.py").write_text("B = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("import vibe_quant.a\n")
    return tmp_path


def _count_calls(monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    calls: list[int] = []
    real: Callable[..., Any] = getattr(swarm_check, name)

    def counted(*args: Any, **kw: Any) -> Any:
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(swarm_check, name, counted)
    return calls


def test_plugins_and_loader_scans_are_cached_and_see_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _tree(tmp_path)
    loader_calls = _count_calls(monkeypatch, "_loader_call_sites")
    plugin_calls = _count_calls(monkeypatch, "_check_pytest_plugins")
    assert swarm_check.loader_call_sites(wt) == []
    assert swarm_check.loader_call_sites(wt) == []
    swarm_check.check_pytest_plugins(wt)
    swarm_check.check_pytest_plugins(wt)
    assert (len(loader_calls), len(plugin_calls)) == (1, 1)
    (wt / "vibe_quant" / "b.py").write_text("import importlib\nimportlib.import_module('x')\n")
    assert swarm_check.loader_call_sites(wt) == ["vibe_quant/b.py"]
    (wt / "tests" / "test_a.py").write_text("pytest_plugins = ['x']\n")
    for _ in range(2):  # a FullSuite is cached and re-raised
        with pytest.raises(swarm_check.FullSuite):
            swarm_check.check_pytest_plugins(wt)
    assert len(plugin_calls) == 2


# --- the per-test-file PASS cache (shared across worktrees; chief ruling, thread 7760) ------


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "HOME": str(repo),
        "GIT_AUTHOR_NAME": "g",
        "GIT_AUTHOR_EMAIL": "g@t",
        "GIT_COMMITTER_NAME": "g",
        "GIT_COMMITTER_EMAIL": "g@t",
        "PATH": "/usr/bin:/bin",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    ).stdout


class _Pytest:
    """RunFn: git runs for real; a pytest call records its test files and writes the junit
    report its PYTEST_ADDOPTS names, one testcase per file (failing for files in `red`)."""

    def __init__(self) -> None:
        self.runs: list[list[str]] = []
        self.red: set[str] = set()
        self.skipped: set[str] = set()  # their testcase carries a <skipped/> row
        self.deselected: set[str] = set()  # a hook deselected one of their items
        self.during: Callable[[], None] | None = None  # runs inside the pytest call

    def __call__(
        self, cmd: Sequence[str], cwd: Path, env: Mapping[str, str] | None = None, timeout: int = 0
    ) -> tuple[int, str]:
        c = [str(x) for x in cmd]
        if c[0] == "git":
            proc = subprocess.run(c, cwd=str(cwd), capture_output=True, text=True)
            return proc.returncode, proc.stdout + proc.stderr
        files = [a for a in c if a.startswith("tests/")]
        self.runs.append(files)
        junit = next(
            o.removeprefix("--junitxml=")
            for o in (env or {})["PYTEST_ADDOPTS"].split()
            if o.startswith("--junitxml=")
        )
        if self.during is not None:
            self.during()
        # a pytest a test spawns inherits the env: the plugin must ride on argv only (it
        # leaked through PYTEST_ADDOPTS and the nested run exited 3)
        assert swarm_check.DESELECT_PLUGIN not in (env or {}).get("PYTEST_ADDOPTS", "")
        if record := (env or {}).get(swarm_check.DESELECT_ENV):  # what the record plugin writes
            assert c[c.index("-p") + 1] == swarm_check.DESELECT_PLUGIN
            lines = "".join(f"{f}::test_y\n" for f in files if f in self.deselected)
            (Path(record) / "gw0.txt").write_text(lines)
        cases = "".join(
            f'<testcase classname="{f[:-3].replace("/", ".")}" name="test_x" time="0.01">'
            + ("<failure message='boom'/>" if f in self.red else "")
            + ("<skipped message='unavailable'/>" if f in self.skipped else "")
            + "</testcase>"
            for f in files
        )
        Path(junit).write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
        bad = [f for f in files if f in self.red]
        out = "".join(f"FAILED {f}::test_x - boom\n" for f in bad)
        return (1, out + "1 failed") if bad else (0, f"{len(files)} passed in 0.01s")


_SELECTED = ["tests/unit/test_foo.py", "tests/unit/test_other.py", "tests/unit/test_tool.py"]


def _cache_repo(tmp_path: Path) -> Path:
    """main: three modules and three tests; feature: one commit changing all three modules.
    test_tool.py reaches scripts/agents/tool.py only BY PATH (the board_io case)."""
    repo = tmp_path / "wt"
    files = {
        "vibe_quant/foo.py": "F = 1\n",
        "vibe_quant/bar.py": "B = 1\n",
        "scripts/agents/tool.py": "T = 1\n",
        "tests/unit/test_foo.py": "from vibe_quant.foo import F\n",
        "tests/unit/test_other.py": "from vibe_quant.bar import B\n",
        "tests/unit/test_tool.py": 'TOOL = ROOT / "scripts" / "agents" / "tool.py"\n',
        "docs/a.md": "a\n",
        "uv.lock": "lock\n",  # main's (_main_env): the tests run from main's venv
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _main_env(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-qb", "feature")
    _commit(
        repo,
        {
            "vibe_quant/foo.py": "F = 2\n",
            "vibe_quant/bar.py": "B = 2\n",
            "scripts/agents/tool.py": "T = 2\n",
        },
    )
    return repo


def _main_env(main: Path) -> None:
    """The main checkout's inputs of every test run (file_cache_env): lockfile, venv config
    and one installed package."""
    (main / "uv.lock").write_text("lock\n")
    site = main / ".venv" / "lib" / "python3.13" / "site-packages"
    (site / "pytest-9.0.0.dist-info").mkdir(parents=True)
    (main / ".venv" / "pyvenv.cfg").write_text("version = 3.13\n")


def _commit(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "round")


def _gate(tmp_path: Path, repo: Path, run: _Pytest, *, full: bool = False) -> tuple[Any, list[Any]]:
    cfg = swarm_check.Config(
        worktree=repo,
        main_root=tmp_path,
        base="main",
        full=full,
        file_cache_dir=tmp_path / "file-cache",
    )
    log = swarm_check.ToolLog(run)
    state = swarm_check.git_state(repo, "main", log)
    result = swarm_check._run_tests(cfg, state, {"pytest": ["pytest"]}, log, {})
    return result, log.entries


def _ran(run: _Pytest) -> set[str]:
    return {f for files in run.runs for f in files}


class TestFileCache:
    def test_a_by_path_edit_reruns_its_reader_and_reuses_the_rest(self, tmp_path: Path) -> None:
        repo = _cache_repo(tmp_path)
        run = _Pytest()
        first, _ = _gate(tmp_path, repo, run)
        assert first.check.status == "PASS" and _ran(run) == set(_SELECTED)
        _commit(repo, {"scripts/agents/tool.py": "T = 3\n"})  # a fix round: one script edit
        run = _Pytest()
        second, log = _gate(tmp_path, repo, run)
        assert second.check.status == "PASS"
        assert _ran(run) == {"tests/unit/test_tool.py"}  # loads tool.py by path: re-run
        assert "cached: 2 files skipped" in second.check.detail
        tree = _git(repo, "rev-parse", "HEAD~1^{tree}").strip()
        assert f"reused tests/unit/test_foo.py (passed at tree {tree[:12]})" in log

    def test_an_import_closure_edit_reruns_the_importer(self, tmp_path: Path) -> None:
        repo = _cache_repo(tmp_path)
        _gate(tmp_path, repo, _Pytest())
        _commit(repo, {"vibe_quant/bar.py": "B = 3\n"})
        run = _Pytest()
        _gate(tmp_path, repo, run)
        assert _ran(run) == {"tests/unit/test_other.py"}

    def test_a_second_hop_edit_reruns_the_test_two_hops_away(self, tmp_path: Path) -> None:
        """Review rev-gatefast-1 B1 (chief ruling R-gf-4): round 1 edits foo.py (which imports
        bar.py) and test_foo PASSes; round 2 edits only bar.py. test_foo is still selected
        (foo.py differs from the base) and must re-run, not reuse its stale PASS: the cache
        invalidation walks the transitive closure, not the one-hop selection."""
        repo = _cache_repo(tmp_path)
        _commit(repo, {"vibe_quant/foo.py": "from vibe_quant.bar import B\nF = 2\n"})
        _gate(tmp_path, repo, _Pytest())
        _commit(repo, {"vibe_quant/bar.py": "B = 3\n"})
        run = _Pytest()
        run.red = {"tests/unit/test_foo.py"}
        result, _ = _gate(tmp_path, repo, run)
        assert _ran(run) == {"tests/unit/test_foo.py", "tests/unit/test_other.py"}
        assert result.check.status == "FAIL"

    def test_a_failed_file_is_never_stored(self, tmp_path: Path) -> None:
        repo = _cache_repo(tmp_path)
        _gate(tmp_path, repo, _Pytest())
        _commit(repo, {"vibe_quant/bar.py": "B = 3\n"})
        run = _Pytest()
        run.red = {"tests/unit/test_other.py"}
        assert _gate(tmp_path, repo, run)[0].check.status == "FAIL"
        _commit(repo, {"docs/a.md": "b\n"})  # nothing reaches test_other: its old PASS is stale
        run = _Pytest()
        _gate(tmp_path, repo, run)
        assert _ran(run) == {"tests/unit/test_other.py"}

    def test_full_mode_neither_reads_nor_writes_the_cache(self, tmp_path: Path) -> None:
        repo = _cache_repo(tmp_path)
        _gate(tmp_path, repo, _Pytest())
        run = _Pytest()
        result, _ = _gate(tmp_path, repo, run, full=True)
        assert run.runs[0] == ["tests/unit"] and "cached" not in result.check.detail

    @pytest.mark.parametrize("damage", ["garbage", "missing-tree", "conftest"])
    def test_any_doubt_is_a_miss(self, tmp_path: Path, damage: str) -> None:
        repo = _cache_repo(tmp_path)
        _gate(tmp_path, repo, _Pytest())
        entries = sorted((tmp_path / "file-cache").rglob("*.json"))
        assert len(entries) == 3
        if damage == "garbage":
            for entry in entries:
                entry.write_text("{not json")
        elif damage == "missing-tree":
            for entry in entries:
                entry.write_text(
                    entry.read_text().replace(
                        _git(repo, "rev-parse", "HEAD^{tree}").strip(), "0" * 40
                    )
                )
        else:  # any conftest change between the PASS and now
            (repo / "tests" / "unit" / "conftest.py").write_text("X = 1\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", "conftest")
        _commit(repo, {"docs/a.md": "c\n"})
        run = _Pytest()
        result, _ = _gate(tmp_path, repo, run)
        assert "cached" not in result.check.detail
        assert set(_SELECTED) <= _ran(run) or run.runs[0] == ["tests/unit"]

    def test_a_test_file_in_the_diff_is_never_reused(self, tmp_path: Path) -> None:
        repo = _cache_repo(tmp_path)
        _commit(repo, {"tests/unit/test_foo.py": "from vibe_quant.foo import F  # edited\n"})
        _gate(tmp_path, repo, _Pytest())
        _commit(repo, {"docs/a.md": "d\n"})
        run = _Pytest()
        _gate(tmp_path, repo, run)
        assert "tests/unit/test_foo.py" in _ran(run)
        assert "tests/unit/test_other.py" not in _ran(run)

    # --- review rev-gate-1: a stored PASS must be a whole, stable, fresh-equivalent run ------

    @pytest.mark.parametrize("partial", ["skipped", "deselected"])
    def test_a_partial_run_is_never_stored(self, tmp_path: Path, partial: str) -> None:
        """B2: a skipped row (NT preflight, skipif) or a call that deselected items (a
        collection hook) did not prove the file green."""
        repo = _cache_repo(tmp_path)
        run = _Pytest()
        if partial == "skipped":
            run.skipped = {"tests/unit/test_other.py"}
        else:
            run.deselected = {"tests/unit/test_other.py"}
        assert _gate(tmp_path, repo, run)[0].check.status == "PASS"
        _commit(repo, {"docs/a.md": "e\n"})
        run = _Pytest()
        _gate(tmp_path, repo, run)
        assert _ran(run) == {"tests/unit/test_other.py"}

    def test_a_head_that_moved_during_the_run_stores_nothing(self, tmp_path: Path) -> None:
        """B3: HEAD switched away and back while pytest ran: same tree after, but the PASS
        may belong to the other commit (the reflog grew)."""
        repo = _cache_repo(tmp_path)
        run = _Pytest()

        def detour() -> None:
            _git(repo, "checkout", "-q", "main")
            _git(repo, "checkout", "-q", "feature")

        run.during = detour
        assert _gate(tmp_path, repo, run)[0].check.status == "PASS"
        assert list((tmp_path / "file-cache").glob("*.json")) == []

    @pytest.mark.parametrize("change", ["new-package", "no-venv"])
    def test_another_environment_reuses_nothing(self, tmp_path: Path, change: str) -> None:
        """B1: an installed package changed (no lockfile change), or the env is unreadable."""
        repo = _cache_repo(tmp_path)
        _gate(tmp_path, repo, _Pytest())
        site = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages"
        if change == "new-package":
            (site / "pytest_randomly-4.0.dist-info").mkdir()
        else:
            (tmp_path / ".venv" / "pyvenv.cfg").unlink()
        _commit(repo, {"docs/a.md": "f\n"})
        run = _Pytest()
        result, _ = _gate(tmp_path, repo, run)
        assert _ran(run) == set(_SELECTED) and "cached" not in result.check.detail
        if change == "no-venv":  # unreadable: no reusable "missing" token is stored either
            run = _Pytest()
            _gate(tmp_path, repo, run)
            assert _ran(run) == set(_SELECTED)

    @pytest.mark.parametrize(
        ("source", "stored"),
        [
            ('P = "data/catalog/x"', False),
            ("from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH", False),
            ("from tests.unit.test_fast_tier import needs_catalog", False),
            ('P = ROOT / "data" / "archive"', False),
            ("C = ParquetDataCatalog(tmp_path / 'catalog')  # a temp catalog", True),
        ],
    )
    def test_only_a_test_reading_real_data_is_never_stored(
        self, tmp_path: Path, source: str, stored: bool
    ) -> None:
        """B1: the ignored real catalog/archive/state can change with no tree change; a test
        that merely names a catalog (a temp one) still caches."""
        repo = _cache_repo(tmp_path)
        _git(repo, "checkout", "-q", "main")  # on the base: the file is not in the diff
        _commit(repo, {"tests/unit/test_other.py": f"from vibe_quant.bar import B\n{source}\n"})
        _git(repo, "checkout", "-q", "feature")
        _git(repo, "rebase", "-q", "main")
        _gate(tmp_path, repo, _Pytest())
        _commit(repo, {"docs/a.md": "g\n"})
        run = _Pytest()
        _gate(tmp_path, repo, run)
        assert ("tests/unit/test_other.py" in _ran(run)) is not stored

    @pytest.mark.slow(
        reason="one real pytest -n 2 run: the deselection must happen in an xdist worker"
    )
    def test_a_hook_deselection_in_an_xdist_worker_is_never_stored(self, tmp_path: Path) -> None:
        """Review rev-gate-1 PARTIAL HOOK: a conftest hook deselected the red test inside the
        workers, the controller summary said only `1 passed`, and the file's PASS was stored.
        The record plugin (ruling R3c) blocks that file; an untouched file is still stored."""
        _main_env(tmp_path)
        repo = tmp_path / "wt"
        files = {
            "pyproject.toml": "[tool.pytest.ini_options]\n",
            "uv.lock": "lock\n",
            ".gitignore": "__pycache__/\n.pytest_cache/\n",
            "vibe_quant/__init__.py": "",
            "vibe_quant/foo.py": "X = 1\n",
            "tests/conftest.py": (
                "def pytest_collection_modifyitems(config, items):\n"
                "    drop = [i for i in items if i.name == 'test_regression']\n"
                "    items[:] = [i for i in items if i not in drop]\n"
                "    config.hook.pytest_deselected(items=drop)\n"
            ),
            "tests/unit/test_sample.py": (
                "from vibe_quant.foo import X\n"
                "def test_smoke(): assert X\n"
                "def test_regression(): assert X == 1\n"
            ),
            "tests/unit/test_plain.py": "from vibe_quant.foo import X\ndef test_plain(): assert X\n",
        }
        for rel, text in files.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(text)
        (tmp_path / "pyproject.toml").write_text(
            files["pyproject.toml"]
        )  # main's deps = the worktree's
        _git(tmp_path, "init", "-q", "-b", "main", str(repo))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "base")
        _git(repo, "checkout", "-qb", "feature")
        _commit(repo, {"vibe_quant/foo.py": "X = 2\n"})  # test_regression is now red
        cfg = swarm_check.Config(
            worktree=repo, main_root=tmp_path, base="main", file_cache_dir=tmp_path / "file-cache"
        )
        log = swarm_check.ToolLog(swarm_check.real_run)
        state = swarm_check.git_state(repo, "main", log)
        env = {"PYTHONPATH": str(repo), "PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}
        result = swarm_check._run_tests(
            cfg, state, {"pytest": [sys.executable, "-m", "pytest"]}, log, env
        )
        assert result.check.status == "PASS", result.check.detail  # the red test never ran
        stored = [
            json.loads(p.read_text())["path"] for p in (tmp_path / "file-cache").glob("*.json")
        ]
        assert stored == ["tests/unit/test_plain.py"]
