"""Per-round test selection of scripts/agents/swarm_check.py.

The per-round gate runs the reverse import closure, the tests that reach a changed file by
path or name, the policy tests and a `-k` group restricted to files that can match it; the
whole tests/unit runs only under --full or a full_suite_reason. These tests pin what that
selection must still catch (on the REAL tree where a past miss happened) and the --full /
--list-tests modes.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

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


@pytest.fixture(scope="module", autouse=True)
def _memo_conftest_parse() -> Iterator[None]:
    """Perf shims, no assertion change: these plan the REAL tree, and three of its tree-wide
    scans redo the same read-only work for every plan.

    - `_conftest_parts`/`_requests`: every plan re-parses the conftests' fixtures and every test
      file's fixture requests (~2 s per plan on the REAL tree).
    - `import_graph_pair`: `plan_tests` keys the graph cache on the request's non-existent extra
      modules too, so each of the two real-tree probes that names an ADDED file
      (`dsl/plugins/brand_new.py`, `research/sources/zz_review_duplicate.py`) re-parses all ~720
      files for a graph that is the tree's own. Nothing imports a module that is not there, so
      the extra name only resolves imports OF that file and there are none: one graph per tree
      state serves every request.
    Both functions are pure in their arguments, so memoise them here. Production should cache
    them itself (handoff `open:`: neither is in its per-process `_TREE_CACHE`).
    """
    real_parts = swarm_check._conftest_parts
    parts: dict[tuple[object, ...], Any] = {}

    def cached_parts(source: str, module: str, known: Any, bare: Any) -> Any:
        key = (source, module, frozenset(known), tuple(sorted(dict(bare).items())))
        if key not in parts:
            parts[key] = real_parts(source, module, known, bare)
        shared, fixtures = parts[key]
        return set(shared), list(fixtures)  # callers get their own copies

    real_requests = swarm_check._requests
    requests: dict[str, Any] = {}

    def cached_requests(text: str) -> Any:
        if text not in requests:
            requests[text] = real_requests(text)
        found, dynamic = requests[text]
        return set(found), dynamic

    real_pair = swarm_check.import_graph_pair
    graphs: dict[tuple[object, ...], Any] = {}

    def cached_pair(worktree: Path, extra_modules: Any = (), snapshot: Any = None) -> Any:
        walked = swarm_check.plan_snapshot(worktree) if snapshot is None else snapshot
        key = (worktree.resolve(), walked)
        if key not in graphs:
            graphs[key] = _cached_graph(real_pair, worktree, extra_modules, walked)
        return graphs[key]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(swarm_check, "_conftest_parts", cached_parts)
        mp.setattr(swarm_check, "_requests", cached_requests)
        mp.setattr(swarm_check, "import_graph_pair", cached_pair)
        cached_pair(_ROOT)  # the one real build, in setup: every real-tree plan below reuses it
        yield


def _cached_graph(real_pair: Any, worktree: Path, extra_modules: Any, walked: Any) -> Any:
    """`real_pair`'s (graph, importers) for the REAL tree, reused across RUNS from a temp file.

    Parsing ~720 files costs 2.5 s idle and 7 s on a loaded host, and it is the one thing every
    run of this file pays. The key is the plan's own snapshot (path, mtime_ns, size of every .py
    under vibe_quant/ and tests/) plus the selector's bytes, so any edit to the tree or to
    swarm_check.py rebuilds it — the same shape as test_power_layer2.py's tiny_root cache, and a
    stale file can only make a selection ASSERTION fail, never pass. A scratch tree (the small
    fixture repos below) is built, never cached.
    """
    if worktree.resolve() != _ROOT.resolve():
        return real_pair(worktree, extra_modules, walked)
    plugins_key = swarm_check._plugins_cache_key(worktree, walked)
    digest = hashlib.sha256(str(walked).encode())
    digest.update(Path(swarm_check.__file__).read_bytes())
    path = Path(tempfile.gettempdir()) / f"vq-selection-graph-{digest.hexdigest()[:16]}.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        swarm_check._TREE_CACHE[plugins_key] = saved["plugins"]
        return (
            {name: set(mods) for name, mods in saved["graph"].items()},
            {name: set(mods) for name, mods in saved["importers"].items()},
        )
    except (OSError, ValueError, KeyError):
        pass
    graph, importers = real_pair(worktree, extra_modules, walked)
    with contextlib.suppress(OSError):  # a read-only temp dir just costs the rebuild next run
        path.write_text(
            json.dumps(
                {
                    "graph": {name: sorted(mods) for name, mods in graph.items()},
                    "importers": {name: sorted(mods) for name, mods in importers.items()},
                    "plugins": swarm_check._TREE_CACHE.get(plugins_key),
                }
            ),
            encoding="utf-8",
        )
    return graph, importers


def _plan(worktree: Path, changed: list[str], *, full: bool = False) -> Any:
    cfg = swarm_check.Config(worktree=worktree, main_root=worktree, full=full)
    return swarm_check.plan_tests(cfg, swarm_check.GitState(changed=changed, untracked=[]))


class TestRealTreeSelection:
    """The selection on THIS repo's tree catches the failures that slipped past a gate."""

    def test_a_helper_the_root_conftest_imports_still_runs_the_full_suite(self) -> None:
        plan = _plan(_ROOT, ["tests/fixtures/known_results.py"])
        assert plan.full and "tests/conftest.py" in plan.notes[0]

    def test_a_production_resource_diff_runs_the_full_suite(self) -> None:
        """Binance_um_brackets.json is read by path; its reader tests
        (test_symbol_leverage.py) were not selected."""
        plan = _plan(_ROOT, ["vibe_quant/liquidation/binance_um_brackets.json"])
        assert plan.full and plan.files == ["tests/unit"]
        assert "non-Python resource" in plan.notes[0]
        assert _plan(_ROOT, ["tests/fixtures/known_results.json"]).full


def _scripts_repo(tmp_path: Path) -> Path:
    """scripts/agents/b.py loads a.py by file name; test_b.py loads b.py by path; the bus-like
    shim `scripts/agents/sh` has no suffix; test_other.py names nothing."""
    agents = tmp_path / "scripts" / "agents"
    agents.mkdir(parents=True)
    (agents / "a.py").write_text("A = 1\n")
    (agents / "b.py").write_text('SPEC = Path(__file__).with_name("a.py")\n')
    (agents / "sh").write_text('exec python3 "$(dirname "$0")/b.py" "$@"\n')
    (agents / "c.py").write_text("C = 1\n")
    # a chain against sorted order: q2.py <- z_mid.py <- a_top.py <- test_top.py (one pass in
    # path order sees a_top.py before z_mid.py is reached; the walk must iterate to a fixpoint)
    (agents / "q2.py").write_text("Q = 1\n")
    (agents / "z_mid.py").write_text('run("q2.py")\n')
    (agents / "a_top.py").write_text('run("z_mid.py")\n')
    unit = tmp_path / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_b.py").write_text('B = ROOT / "scripts" / "agents" / "b.py"\n')
    (unit / "test_shim.py").write_text('SHIM = "scripts/agents/sh"\n')
    (unit / "test_other.py").write_text('X = "sh is a word"\n')
    (unit / "test_top.py").write_text('T = "scripts/agents/a_top.py"\n')
    return tmp_path


class TestScriptConsumers:
    @pytest.mark.parametrize(
        ("changed", "expected"),
        [
            # a.py -> b.py names it -> test_b.py names b.py; the shim names b.py too
            (["scripts/agents/a.py"], ["tests/unit/test_b.py", "tests/unit/test_shim.py"]),
            (["scripts/agents/b.py"], ["tests/unit/test_b.py", "tests/unit/test_shim.py"]),
            # the shim is matched by "agents/sh", never by the bare word "sh"
            (["scripts/agents/sh"], ["tests/unit/test_shim.py"]),
            (["scripts/agents/c.py"], []),
            (["scripts/agents/q2.py"], ["tests/unit/test_top.py"]),
            (["vibe_quant/x.py", "docs/a.md"], []),
        ],
    )
    def test_a_script_reaches_the_tests_of_its_by_name_consumers(
        self, tmp_path: Path, changed: list[str], expected: list[str]
    ) -> None:
        repo = _scripts_repo(tmp_path)
        assert swarm_check.script_consumer_test_files(changed, repo) == expected

def _mini(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    return tmp_path


class TestReviewProbes:
    """The four review probes (rev-gate-speed-1), as selection assertions on scratch trees."""

    def test_a_string_loaded_module_selects_the_loader_tests(self, tmp_path: Path) -> None:
        repo = _mini(
            tmp_path,
            {
                "vibe_quant/plugins/alpha.py": "X = 2\n",
                "vibe_quant/loader.py": 'import importlib\nNAME = "vibe_quant.plugins.alpha"\n',
                "tests/unit/test_alpha_contract.py": "from vibe_quant.loader import NAME\n",
            },
        )
        plan = _plan(repo, ["vibe_quant/plugins/alpha.py"])
        assert "tests/unit/test_alpha_contract.py" in plan.explicit

class TestRound2Probes:
    """Review rev-gate-speed-2 F1-F3: what the gate does not model goes to the full suite."""

    @pytest.mark.parametrize(
        ("path", "full"),
        [
            ("settings.json", True),  # an unmodelled resource anywhere: full
            ("docs/orchestration/models.json", True),
            ("scripts/agents/hooks/x.json", False),  # scripts/**: the script-consumer walk
            ("frontend/src/App.tsx", False),  # inert: the frontend check covers it
            ("frontend/src/components/x/__fixtures__/y.json", True),  # python tests read these
            (".beads/issues.jsonl", False),
            ("docs/orchestration/README.md", False),  # .md: readers spell the suffix
            ("vibe_quant/strategies/templates/ema_ribbon.yaml", True),
        ],
    )
    def test_non_python_paths_are_full_unless_inert_or_modelled(
        self, path: str, full: bool
    ) -> None:
        assert _plan(_ROOT, [path]).full is full

    def test_an_unmodelled_directory_loader_runs_the_full_suite(self, tmp_path: Path) -> None:
        repo = _mini(
            tmp_path,
            {
                "vibe_quant/sources/__init__.py": "",
                "vibe_quant/sources/alpha.py": "X = 2\n",
                "vibe_quant/loader.py": "import pkgutil\nMODS = pkgutil.iter_modules([])\n",
                "tests/unit/test_contract.py": "from vibe_quant.loader import MODS\n",
            },
        )
        plan = _plan(repo, ["vibe_quant/sources/alpha.py"])
        assert plan.full and "vibe_quant/loader.py" in plan.notes[0]


_LOCK = "LOCK_DIR = '/real'\n\n\ndef where() -> str:\n    return LOCK_DIR\n"


def _fixture_repo(tmp_path: Path, conftest: str, tests: dict[str, str]) -> Path:
    files = {
        "tests/__init__.py": "",
        "tests/unit/__init__.py": "",
        "vibe_quant/__init__.py": "",
        "vibe_quant/lock.py": _LOCK,
        "vibe_quant/store.py": "def make() -> int:\n    return 1\n",
        "tests/unit/conftest.py": conftest,
        **tests,
    }
    return _mini(tmp_path, files)


class TestConftestFixtures:
    """Round-3 chief model: a conftest carries a module into a test only through a fixture."""

    def test_renaming_a_patched_attribute_fails_the_gate(self, tmp_path: Path) -> None:
        """Pin (chief): account_lock.PAPER_LOCK_DIR's shape - rename the attribute an autouse
        fixture patches, and the gate FAILs although no test requests or imports it."""
        conftest = (
            "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef guard(monkeypatch):\n"
            "    from vibe_quant import lock\n\n    monkeypatch.setattr(lock, 'LOCK_DIR', '/t')\n"
        )
        repo = _fixture_repo(
            tmp_path,
            conftest,
            {
                "tests/unit/test_unrelated.py": "def test_ok() -> None:\n    assert True\n",
            },
        )
        (repo / "vibe_quant" / "lock.py").write_text(_LOCK.replace("LOCK_DIR", "LOCK_PATH"))
        python = Path(sys.executable)

        def run(cmd: Any, cwd: Path, env: Any = None, timeout: int = 0) -> tuple[int, str]:
            c = [str(x) for x in cmd]
            if Path(c[0]).name in {"ruff", "mypy"}:
                return 0, "ok"
            if "pytest" in c:
                c = [str(python), *c[c.index("-m") :]]
            proc = subprocess.run(
                c,
                cwd=cwd,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    **(env or {}),
                    "PYTEST_ADDOPTS": "-p no:cacheprovider -o addopts=",
                },
            )
            return proc.returncode, proc.stdout + proc.stderr

        cfg = swarm_check.Config(
            worktree=repo, main_root=repo, base="HEAD", exactness="never", frontend="never"
        )
        state = swarm_check.GitState(changed=["vibe_quant/lock.py"], untracked=[])
        plan = swarm_check.plan_tests(cfg, state)
        assert not plan.full and plan.explicit == {"tests/unit/test_unrelated.py"}
        tools = {"pytest": [str(python), "-m", "pytest"]}
        result = swarm_check._run_tests(cfg, state, tools, run, {"PYTHONPATH": str(repo)})
        assert result.check.status == "FAIL", result.check.detail
        assert "test_unrelated.py::test_ok" in result.check.detail


class TestFullMode:
    def test_full_runs_the_whole_suite_and_every_timing_test_for_any_diff(self) -> None:
        plan = _plan(_ROOT, ["docs/orchestration/README.md"], full=True)
        assert plan.full
        assert [c[0] for c in plan.commands] == [
            ["tests/unit", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-m", "timing"],
        ]
        assert plan.notes == ["full tests/unit: --full (integration mode)"]

    def test_full_is_part_of_the_cache_key(self, tmp_path: Path) -> None:
        round_cfg = swarm_check.Config(worktree=tmp_path, main_root=tmp_path)
        full_cfg = swarm_check.Config(worktree=tmp_path, main_root=tmp_path, full=True)
        assert swarm_check.cache_key("t", "b", round_cfg, "x", "m") != swarm_check.cache_key(
            "t", "b", full_cfg, "x", "m"
        )
