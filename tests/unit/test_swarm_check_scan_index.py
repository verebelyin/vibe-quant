"""Equality proofs for swarm_check's per-tree scan caches (vibe-quant-qlx46 tooling).

A gate that selects fewer tests hides failures, so every cache this module adds is pinned
against the UNCACHED scan it replaced, on the real tree, for a hub module, a plugin and a
test-only diff: same files, same commands, same full/not-full verdict. Trees built by hand
where the rule under test is a shape (a needle inside another needle, a reached module named
in a string) so a regression fails here, not in a maker's gate.

The git-facts tests pin the other half of the round: one run asks `merge-base <base> HEAD`
and `rev-parse <base>^{commit}` once per PHASE (pre-run key, post-run freshness check),
not three times each.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "agents" / "swarm_check.py"
if "swarm_check" in sys.modules:
    swarm_check = sys.modules["swarm_check"]
else:
    _spec = importlib.util.spec_from_file_location("swarm_check", _SCRIPT)
    assert _spec is not None and _spec.loader is not None
    swarm_check = importlib.util.module_from_spec(_spec)
    sys.modules["swarm_check"] = swarm_check
    _spec.loader.exec_module(swarm_check)


@pytest.fixture(autouse=True)
def _fresh_tree_cache() -> Iterator[None]:
    """Each test starts and ends with an empty per-process tree cache: a plan built here is
    never served by a cache another test filled, and fills none it does not need."""
    swarm_check._TREE_CACHE.clear()
    yield
    swarm_check._TREE_CACHE.clear()


def _plan(worktree: Path, changed: list[str], *, full: bool = False) -> Any:
    cfg = swarm_check.Config(worktree=worktree, main_root=worktree, full=full)
    return swarm_check.plan_tests(cfg, swarm_check.GitState(changed=changed, untracked=[]))


def _naive_mention(worktree: Path, needles: set[str]) -> list[str]:
    """The pre-cache `tests_mentioning`: a plain substring scan of every tests/unit test file."""
    if not needles:
        return []
    root = worktree / "tests" / "unit"
    if not root.is_dir():
        return []
    return [
        path.relative_to(worktree).as_posix()
        for path in sorted(root.rglob("test_*.py"))
        if any(needle in path.read_text(encoding="utf-8", errors="replace") for needle in needles)
    ]


def _naive_dynamic(
    reached: set[str], worktree: Path, snapshot: Any = None
) -> list[str]:
    """The pre-index `dynamic_reference_test_files`, straight from each file's own strings.

    `snapshot` is accepted and ignored: the plain scan re-reads the tree it is asked about."""
    if not reached:
        return []
    packages = {m.removesuffix(".__main__") for m in reached if m.endswith(".__main__")}
    root = worktree / "tests" / "unit"
    if not root.is_dir():
        return []
    out: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        names = {m.group(1) for m in swarm_check.QUOTED_MODULE_RE.finditer(text)}
        runs = {m.group(1) for m in swarm_check.RUN_MODULE_RE.finditer(text)}
        hit = any(
            {".".join(parts[:i]) for i in range(2, len(parts) + 1)} & reached
            for parts in (name.split(".") for name in names)
        ) or bool(runs & packages)
        if hit:
            out.append(path.relative_to(worktree).as_posix())
    return out


class TestCachedSelectionEqualsUncached:
    """The kill zone: a cached selection must BE the uncached one, file for file."""

    @pytest.mark.parametrize(
        "changed",
        [
            "vibe_quant/screening/replay_drift.py",  # hub module: a wide import closure
            "vibe_quant/dsl/plugins/rams.py",  # a plugin: DIRECTORY_LOADERS, not imported
            "tests/unit/test_mkbrief.py",  # test-only: the -k group, no vibe_quant module
        ],
    )
    def test_the_plan_is_the_same_with_the_caches_and_with_a_plain_scan(
        self, monkeypatch: pytest.MonkeyPatch, changed: str
    ) -> None:
        cached = _plan(_ROOT, [changed])
        assert cached.explicit  # the diff selects something: an empty plan proves nothing

        swarm_check._TREE_CACHE.clear()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(swarm_check, "tests_mentioning", _naive_mention)
            mp.setattr(swarm_check, "dynamic_reference_test_files", _naive_dynamic)
            plain = _plan(_ROOT, [changed])

        assert plain.explicit == cached.explicit
        assert plain.files == cached.files
        assert plain.commands == cached.commands
        assert plain.full == cached.full

    @pytest.mark.parametrize(
        "changed",
        [
            ["vibe_quant/discovery/pipeline.py"],
            ["vibe_quant/discovery/__main__.py", "vibe_quant/dsl/compiler.py"],
            ["vibe_quant/data/downloader.py", "vibe_quant/screening/replay_drift.py"],
            ["tests/unit/pta_stream_harness.py"],
        ],
    )
    def test_the_dynamic_index_matches_the_plain_scan_on_the_real_tree(
        self, changed: list[str]
    ) -> None:
        """`reached` is computed from the real graph: no dotted module name is spelled in this
        file's source (a gate-meta test that quotes one is pulled into every such gate)."""
        assert all((_ROOT / path).exists() for path in changed)
        graph = swarm_check.build_import_graph(_ROOT)
        reached = swarm_check.reached_modules(changed, graph)
        assert len(reached) > 1
        assert swarm_check.dynamic_reference_test_files(reached, _ROOT) == _naive_dynamic(
            reached, _ROOT
        )

    @pytest.mark.parametrize(
        "text",
        [
            'TARGET = "vibe_quant.foo.f"\n',  # a string PREFIX of the reached module
            'TARGET = "vibe_quant.foo.f.bar.baz"\n',  # deeper: a prefix hit at i=3
            'cmd = [sys.executable, "-m", "vibe_quant.foo"]\n',  # the -m package form
            'X = "vibe_quant.foo"\n',  # a package name alone is no reference
            'X = "vibe_quant.foolish"\n',  # a longer name, not a dotted prefix
        ],
    )
    def test_a_string_reference_is_selected_exactly_as_before(
        self, tmp_path: Path, text: str
    ) -> None:
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "test_ref.py").write_text(text)
        reached = {"vibe_quant.foo.f", "vibe_quant.foo.__main__"}
        got = swarm_check.dynamic_reference_test_files(reached, tmp_path)
        assert got == _naive_dynamic(reached, tmp_path)

    def test_a_needle_inside_another_needle_changes_nothing(self, tmp_path: Path) -> None:
        """`file_needles` returns the path AND the file name; a file holding the path holds the
        name, so dropping the wider needle is exact - a shortcut, not a narrowing."""
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "test_pathy.py").write_text('P = "vibe_quant/pkg/mod.py"\n')
        (unit / "test_bare.py").write_text('N = "mod.py"\n')
        (unit / "test_other.py").write_text("X = 1\n")
        wide = {"vibe_quant/pkg/mod.py", "mod.py"}
        both = ["tests/unit/test_bare.py", "tests/unit/test_pathy.py"]
        assert swarm_check.tests_mentioning(tmp_path, wide) == both
        assert swarm_check.tests_mentioning(tmp_path, {"mod.py"}) == both
        assert swarm_check.tests_mentioning(_ROOT, wide) == _naive_mention(_ROOT, wide)


class TestCacheInvalidation:
    """A cache that outlives an edit would gate the pre-edit tree: every one must re-key."""

    def _repo(self, tmp_path: Path) -> Path:
        (tmp_path / "vibe_quant").mkdir()
        (tmp_path / "vibe_quant" / "foo.py").write_text("def f() -> int:\n    return 1\n")
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "test_other.py").write_text("X = 1\n")
        return tmp_path

    def test_editing_a_test_file_between_two_plans_changes_the_selection(
        self, tmp_path: Path
    ) -> None:
        repo = self._repo(tmp_path)
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == set()

        (repo / "tests" / "unit" / "test_other.py").write_text('P = "foo.py"\n')
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == {"tests/unit/test_other.py"}

        (repo / "tests" / "unit" / "test_other.py").write_text("X = 1\n")
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == set()

    def test_adding_a_string_reference_between_two_plans_changes_the_selection(
        self, tmp_path: Path
    ) -> None:
        repo = self._repo(tmp_path)
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == set()

        (repo / "tests" / "unit" / "test_other.py").write_text('P = "vibe_quant.foo.f"\n')
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == {"tests/unit/test_other.py"}

    def test_a_plan_walks_the_tree_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three accessors used to stat the tree three times in one plan; now it is one walk."""
        walked: list[tuple[str, ...]] = []
        real = swarm_check.tree_fingerprint

        def counting(worktree: Path, *subdirs: str) -> tuple[tuple[str, int, int], ...]:
            walked.append(subdirs)
            return real(worktree, *subdirs)

        monkeypatch.setattr(swarm_check, "tree_fingerprint", counting)
        plan = _plan(_ROOT, ["vibe_quant/screening/replay_drift.py"])
        assert plan.explicit
        assert len(walked) == 1, walked

    def test_editing_an_import_between_two_plans_changes_the_selection(
        self, tmp_path: Path
    ) -> None:
        """The import graph keys on the snapshot: a new vibe_quant edge in the SAME process
        must re-plan (a stale graph would miss the test that now reaches the module)."""
        repo = self._repo(tmp_path)
        (repo / "vibe_quant" / "bar.py").write_text("B = 1\n")
        (repo / "tests" / "unit" / "test_other.py").write_text(
            "from vibe_quant.foo import F\n\n\ndef test_x() -> None:\n    assert F == 1\n"
        )
        assert _plan(repo, ["vibe_quant/bar.py"]).explicit == set()

        (repo / "vibe_quant" / "foo.py").write_text("from vibe_quant.bar import B\n\nF = B\n")
        assert _plan(repo, ["vibe_quant/bar.py"]).explicit == {"tests/unit/test_other.py"}

    def test_editing_a_loader_between_two_plans_changes_the_plan(self, tmp_path: Path) -> None:
        """The loader scan keys on the snapshot: a dynamic import added to a vibe_quant file
        must send the next plan to the full suite (a stale scan PASSes it silently)."""
        repo = self._repo(tmp_path)
        assert not _plan(repo, ["vibe_quant/foo.py"]).full

        (repo / "vibe_quant" / "foo.py").write_text(
            "import importlib\n\nMOD = importlib.import_module('x')\n"
        )
        plan = _plan(repo, ["vibe_quant/foo.py"])
        assert plan.full
        assert "unmodelled dynamic loader vibe_quant/foo.py" in plan.notes[0]

    def test_editing_a_conftest_between_two_plans_changes_the_plan(self, tmp_path: Path) -> None:
        """The conftest index (parsed fixtures, `known`/`bare`, tests_in) keys on the snapshot:
        a fixture that stops importing the module must stop sending the plan to the suite."""
        repo = self._repo(tmp_path)
        (repo / "tests" / "unit" / "conftest.py").write_text(
            "import pytest\n\nfrom vibe_quant.foo import f\n\n\n"
            "@pytest.fixture\ndef fx():\n    return f()\n"
        )
        (repo / "tests" / "unit" / "test_other.py").write_text(
            "def test_x(fx) -> None:\n    assert fx == 1\n"
        )
        first = _plan(repo, ["vibe_quant/foo.py"])
        assert first.full  # the tests/unit conftest's fixture depends on the change

        (repo / "tests" / "unit" / "conftest.py").write_text(
            "import pytest\n\n\n@pytest.fixture\ndef fx():\n    return 1\n"
        )
        second = _plan(repo, ["vibe_quant/foo.py"])
        assert not second.full

    def test_a_test_file_that_gains_an_import_between_two_plans_is_selected(
        self, tmp_path: Path
    ) -> None:
        """R6a: the graph analyses the file's CURRENT bytes - a tree cached by path alone would
        keep selecting nothing after the file gains an edge to the changed module."""
        repo = self._repo(tmp_path)
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == set()

        (repo / "tests" / "unit" / "test_other.py").write_text("import vibe_quant.foo\n")
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == {"tests/unit/test_other.py"}

    def test_a_conftest_that_gains_pytest_plugins_between_two_plans_is_full_suite(
        self, tmp_path: Path
    ) -> None:
        """R6b + R4: the verdict comes from the graph walk's fold and is keyed on the plan's
        snapshot, so an edit that adds a registration re-keys (a stale "none" would run a
        partial suite against plugin fixtures)."""
        repo = self._repo(tmp_path)
        conftest = repo / "tests" / "unit" / "conftest.py"
        conftest.write_text("import pytest\n")
        assert not _plan(repo, ["vibe_quant/foo.py"]).full

        conftest.write_text("pytest_plugins = ['vibe_quant.plugin']\n")
        plan = _plan(repo, ["vibe_quant/foo.py"])
        assert plan.full
        assert plan.notes == [
            "full tests/unit: pytest_plugins in tests/unit/conftest.py "
            "(plugin fixtures are not modelled)"
        ]

    def test_a_unicode_spelled_pytest_plugins_still_is_full_suite(self, tmp_path: Path) -> None:
        """Python NFKC-normalises identifiers: fullwidth `ｐｙｔｅｓｔ_plugins` IS the
        `pytest_plugins` attribute, though the raw text lacks that substring (review
        rev-parse-1 F1). The fold's substring shortcut must not skip such a file."""
        repo = self._repo(tmp_path)
        (repo / "tests" / "unit" / "conftest.py").write_text(
            'ｐｙｔｅｓｔ_plugins = ["vibe_quant.plugin"]\n', encoding="utf-8"
        )
        plan = _plan(repo, ["vibe_quant/foo.py"])
        assert plan.full
        assert plan.notes == [
            "full tests/unit: pytest_plugins in tests/unit/conftest.py "
            "(plugin fixtures are not modelled)"
        ]

    def test_pytest_plugins_under_type_checking_still_is_full_suite(self, tmp_path: Path) -> None:
        """R3: a conftest's tree is SHARED by the graph walk and the plugin fold, so the
        TYPE_CHECKING skip must not empty the body in place. pytest reads the attribute after
        the module ran; the plan must stay full (a mutation would SHRINK the selection)."""
        repo = self._repo(tmp_path)
        (repo / "tests" / "unit" / "conftest.py").write_text(
            "from typing import TYPE_CHECKING\n\n"
            "if TYPE_CHECKING:\n    pytest_plugins = ['vibe_quant.plugin']\n"
        )
        plan = _plan(repo, ["vibe_quant/foo.py"])
        assert plan.full
        assert plan.notes == [
            "full tests/unit: pytest_plugins in tests/unit/conftest.py "
            "(plugin fixtures are not modelled)"
        ]


class TestOneTreePerFile:
    """R2: the graph walk, the pytest_plugins fold, `_conftest_parts` and `_requests` all
    analyse the same file bytes, so a cold plan must parse each text ONCE (it parsed the test
    files three times: graph, plugin scan, `_requests`)."""

    def _repo(self, tmp_path: Path) -> Path:
        (tmp_path / "vibe_quant").mkdir()
        (tmp_path / "vibe_quant" / "foo.py").write_text("def f() -> int:\n    return 1\n")
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "conftest.py").write_text(
            "import pytest\n\n\n@pytest.fixture\ndef fx():\n"
            "    from vibe_quant.foo import f\n\n    return f()\n"
        )
        (unit / "test_other.py").write_text("def test_x(fx) -> None:\n    assert fx == 1\n")
        return tmp_path

    def _counted_parses(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Every text `ast.parse` is handed while the patch is on (the cache starts empty)."""
        parsed: list[str] = []
        real_parse = swarm_check.ast.parse

        def counting(source: str, *args: Any, **kwargs: Any) -> Any:
            parsed.append(source)
            return real_parse(source, *args, **kwargs)

        monkeypatch.setattr(swarm_check.ast, "parse", counting)
        swarm_check.ast_cached.cache_clear()
        return parsed

    def test_a_plan_parses_each_text_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = self._repo(tmp_path)
        parsed = self._counted_parses(monkeypatch)
        plan = _plan(repo, ["vibe_quant/foo.py"])
        assert plan.explicit == {"tests/unit/test_other.py"}  # the fixture pulls the test in
        assert parsed, "the plan parsed nothing: the counter is off the parse path"

        counts: dict[str, int] = {}
        for text in parsed:
            counts[text] = counts.get(text, 0) + 1
        assert max(counts.values()) == 1, {
            text[:40]: n for text, n in counts.items() if n > 1
        }

    def test_a_second_plan_on_the_same_tree_parses_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = self._repo(tmp_path)
        parsed = self._counted_parses(monkeypatch)
        first = _plan(repo, ["vibe_quant/foo.py"])
        assert first.explicit
        before = len(parsed)
        again = _plan(repo, ["vibe_quant/foo.py"])
        assert again.explicit == first.explicit
        assert len(parsed) == before

    def test_a_plan_does_not_walk_the_tests_tree_twice_for_pytest_plugins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R4: the graph walk visits every tests/**/*.py, so the plan gets its verdict there;
        the standalone scan (a second read AND parse of every test file) must not run."""
        def boom(_worktree: Path) -> None:
            raise AssertionError("the plan re-walked tests/ for pytest_plugins")

        monkeypatch.setattr(swarm_check, "_check_pytest_plugins", boom)
        plan = _plan(_ROOT, ["vibe_quant/screening/replay_drift.py"])
        assert plan.explicit


# --------------------------------------------------------------------------- git spawns

_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "cache",
    "GIT_AUTHOR_EMAIL": "cache@test",
    "GIT_COMMITTER_NAME": "cache",
    "GIT_COMMITTER_EMAIL": "cache@test",
}


def _git_env() -> dict[str, str]:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV_OVERRIDES)
    return env


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
        env=_git_env(),
    )  # fmt: skip
    return out.stdout


def _template_repo() -> Path:
    """One committed repo per process: a copytree is the independent clone each test dirties."""
    root = Path(tempfile.mkdtemp(prefix="swarm-check-spawns-"))
    import atexit

    atexit.register(shutil.rmtree, root, ignore_errors=True)
    repo = root / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=_git_env())
    (repo / "vibe_quant").mkdir()
    (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 1\n")
    unit = repo / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_foo.py").write_text(
        "from vibe_quant.foo import foo\n\n\ndef test_foo() -> None:\n    assert foo()\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
    _git(repo, "commit", "-qam", "foo")
    return repo


_TEMPLATE: Path | None = None


def _repo(tmp_path: Path) -> Path:
    global _TEMPLATE
    if _TEMPLATE is None or not _TEMPLATE.exists():
        _TEMPLATE = _template_repo()
    repo = tmp_path / "wt"
    shutil.copytree(_TEMPLATE, repo, symlinks=True)
    return repo


_PASS_CHECKS = (
    swarm_check.Check("scope", "SKIP", "no --scope given"),
    swarm_check.Check("clean", "PASS", "no untracked files"),
    swarm_check.Check("tests", "PASS", "1 passed in 0.01s"),
)


class CountingRun:
    """Real git through the binary, tools answered by `responses`; every argv is recorded."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: dict[str, tuple[int, str]] = {}

    def __call__(
        self, cmd: Sequence[str], cwd: Path, env: Any = None, timeout: int = 600
    ) -> tuple[int, str]:
        c = [str(x) for x in cmd]
        self.calls.append(c)
        if c[0] == "git":
            proc = subprocess.run(c, cwd=str(cwd), capture_output=True, text=True, env=_git_env())
            return proc.returncode, proc.stdout + proc.stderr
        for needle, resp in self.responses.items():
            if any(needle in Path(arg).name for arg in c):
                return resp
        return 0, ""

    def git(self, *argv: str) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "git" and all(a in c for a in argv)]

    def base_commit_asks(self) -> int:
        return len([c for c in self.calls if c[0] == "git" and any(a.endswith("^{commit}") for a in c)])


class TestGitFactsOncePerPhase:
    def test_a_phase_asks_merge_base_and_the_base_commit_once(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        runner = CountingRun()
        cfg = swarm_check.Config(worktree=repo, main_root=tmp_path, base="main")

        facts = swarm_check.GitFacts()
        state = swarm_check.git_state(repo, "main", runner, facts)
        assert state.changed == ["vibe_quant/foo.py"]
        key = swarm_check._cache_probe(cfg, runner, state=state, facts=facts)[0]

        assert key
        assert len(runner.git("merge-base")) == 1
        assert runner.base_commit_asks() == 1
        # the probe took the state it was given instead of re-asking diff/status
        assert len(runner.git("diff")) == 1
        assert len(runner.git("status")) == 1

    def test_the_post_run_probe_is_a_new_phase_and_the_stamp_reuses_it(
        self, tmp_path: Path
    ) -> None:
        """Pre-run key, then the post-run freshness check with a NEW memo: `merge-base` is asked
        again (a mid-run commit or moved base must be seen), and `write_stamp` records the very
        answers the key was checked against instead of spawning git a third time."""
        repo = _repo(tmp_path)
        runner = CountingRun()
        cfg = swarm_check.Config(worktree=repo, main_root=tmp_path, base="main")

        pre = swarm_check.GitFacts()
        state = swarm_check.git_state(repo, "main", runner, pre)
        key = swarm_check._cache_probe(cfg, runner, state=state, facts=pre)[0]
        assert key and len(runner.git("merge-base")) == 1

        post = swarm_check.GitFacts()
        post_key = swarm_check._cache_probe(cfg, runner, facts=post)[0]
        assert post_key == key  # nothing moved: the same key, so a stamp is allowed
        assert len(runner.git("merge-base")) == 2  # the new phase re-asks: not a stale memo

        swarm_check.write_stamp(repo, runner, cfg, key, _PASS_CHECKS, post)
        assert swarm_check.load_stamp(repo, runner, key, post) is not None
        assert len(runner.git("merge-base")) == 2
        assert runner.base_commit_asks() == 2

    def test_a_gate_run_asks_each_question_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Through the entry point: git_state, the PASS-stamp probe and run_checks all want
        diff/status/base/merge-base, and the run asks each exactly once. The LOG trailer's
        post-run read (did HEAD/tree/edits move while the tools ran?) is ONE more phase:
        status and HEAD twice, never more."""
        repo = _repo(tmp_path)
        for var in ("SWARM_CHECK_RUFF", "SWARM_CHECK_MYPY", "SWARM_CHECK_PYTEST"):
            monkeypatch.setenv(var, "/bin/true")
        monkeypatch.setenv("SWARM_CHECK_ALLOW_STUBS", "1")
        monkeypatch.delenv("SWARM_BUS", raising=False)
        monkeypatch.delenv("SWARM_AGENT", raising=False)
        runner = CountingRun()

        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never"],
            run=runner,
        )

        assert rc == 0
        assert len(runner.git("merge-base")) == 1
        assert runner.base_commit_asks() == 1
        assert len(runner.git("diff")) == 1
        assert len(runner.git("status")) == 2
        assert len(runner.git("rev-parse", "HEAD")) == 2
