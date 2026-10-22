"""Per-round test selection of scripts/agents/swarm_check.py (bead q2ijs).

The per-round gate runs the reverse import closure, the tests that reach a changed file by
path or name, the policy tests and a `-k` group restricted to files that can match it; the
whole tests/unit runs only under --full or a full_suite_reason. These tests pin what that
selection must still catch (on the REAL tree where a past miss happened) and the --full /
--list-tests modes.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

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


def _plan(worktree: Path, changed: list[str], *, full: bool = False) -> Any:
    cfg = swarm_check.Config(worktree=worktree, main_root=worktree, full=full)
    return swarm_check.plan_tests(cfg, swarm_check.GitState(changed=changed, untracked=[]))


class TestRealTreeSelection:
    """The selection on THIS repo's tree catches the failures that slipped past a gate."""

    def test_discovery_main_diff_selects_the_run_status_logging_test(self) -> None:
        """cc-l3-t7-1 (job 20261021-rabmo): a discovery/__main__.py change failed
        test_discovery_run_status_write_failure_logs_error outside the maker's targeted set."""
        plan = _plan(_ROOT, ["vibe_quant/discovery/__main__.py"])
        assert not plan.full
        assert "tests/unit/test_silent_run_status_logging.py" in plan.explicit
        assert "tests/unit" not in plan.files
        assert len(plan.commands) == 1  # one parallel file-list call, no timing call

    def test_handoff_lint_diff_selects_the_bus_tests(self) -> None:
        """Commit 8805dea8 changed scripts/agents/handoff_lint.py; its gate PASSed and
        tests/unit/test_swarm_bus.py failed 2 tests: bus.py loads handoff_lint.py by file name
        and the test loads bus.py by path - no import edge anywhere."""
        plan = _plan(_ROOT, ["scripts/agents/handoff_lint.py"])
        assert not plan.full
        assert "tests/unit/test_swarm_bus.py" in plan.explicit
        assert "tests/unit/test_handoff_lint.py" in plan.explicit

    def test_a_docs_only_diff_runs_the_policy_tests_and_the_docs_readers(self) -> None:
        plan = _plan(_ROOT, ["docs/orchestration/README.md"])
        assert not plan.full
        assert "tests/unit/test_steering_budget.py" in plan.explicit
        assert "tests/unit/test_exception_hygiene.py" not in plan.explicit  # scans vibe_quant

    def test_a_plugin_diff_selects_the_loader_consumers(self) -> None:
        """F1 r1: plugin_loader.py imports dsl/plugins/*.py by directory scan, so a rams.py
        diff (or an ADDED plugin) seeds the loader closure."""
        for path in ("vibe_quant/dsl/plugins/rams.py", "vibe_quant/dsl/plugins/brand_new.py"):
            plan = _plan(_ROOT, [path])
            assert not plan.full, plan.notes
            assert "tests/unit/test_indicator_compute.py" in plan.explicit

    def test_unit_conftest_autouse_patch_fixtures_do_not_force_the_full_suite(self) -> None:
        """Round-3 chief model: tests/unit/conftest.py's autouse fixtures only setattr on
        paper.account_lock / api.data_precheck, or act via sys.modules.get, so these
        modules (77 of 226 went full under the per-file conftest rule) stay targeted."""
        for path in (
            "vibe_quant/paper/node.py",
            "vibe_quant/paper/account_lock.py",
            "vibe_quant/api/data_precheck.py",
            "vibe_quant/dsl/aux_data.py",
            "vibe_quant/discovery/operators.py",
        ):
            plan = _plan(_ROOT, [path])
            assert not plan.full, (path, plan.notes)
            assert any(f.startswith("tests/unit/") for f in plan.explicit)  # autouse runs

    def test_a_helper_the_root_conftest_imports_still_runs_the_full_suite(self) -> None:
        plan = _plan(_ROOT, ["tests/fixtures/known_results.py"])
        assert plan.full and "tests/conftest.py" in plan.notes[0]

    def test_a_test_helper_diff_runs_its_importers_whole(self) -> None:
        """F2: the helper's consumers were only -k candidates (0 of their nodes ran)."""
        plan = _plan(_ROOT, ["tests/unit/pta_stream_harness.py"])
        assert {"tests/unit/test_indicator_compute.py", "tests/unit/test_pta_replay.py"} <= (
            plan.explicit
        )
        assert all("-k" not in args for args, _, _ in plan.commands)
        # the sys.path-insert form `from pta_stream_harness import ...` counts too
        assert "tests/unit/test_plugins/test_kama_exactness.py" in plan.explicit

    def test_a_changed_test_file_selects_the_tests_importing_it(self) -> None:
        plan = _plan(_ROOT, ["tests/unit/test_audit_paper_node.py"])
        assert "tests/unit/test_paper_watch_inherit.py" in plan.explicit

    def test_a_production_resource_diff_runs_the_full_suite(self) -> None:
        """F3: binance_um_brackets.json is read by path; its reader tests
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

    def test_generic_file_names_need_their_parent_dir(self, tmp_path: Path) -> None:
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "test_any_main.py").write_text('X = "__main__.py"\n')  # fixture text only
        (unit / "test_disc_main.py").write_text('X = "discovery/__main__.py"\n')
        got = swarm_check.path_mention_test_files(["vibe_quant/discovery/__main__.py"], tmp_path)
        assert got == ["tests/unit/test_disc_main.py"]
        assert swarm_check.file_needles("vibe_quant/discovery/__main__.py") == {
            "vibe_quant/discovery/__main__.py",
            "discovery/__main__.py",
        }
        assert swarm_check.file_needles("scripts/agents/bus.py") == {
            "scripts/agents/bus.py",
            "bus.py",
        }


class TestDynamicReferences:
    """String references to a reached module: no import edge, still a consumer."""

    @pytest.mark.parametrize(
        ("text", "selected"),
        [
            ('cmd = [sys.executable, "-m", "vibe_quant.disc", "--help"]\n', True),  # runs __main__
            ('monkeypatch.setattr("vibe_quant.disc.pipeline.run", fake)\n', True),  # patch target
            ("mod = importlib.import_module('vibe_quant.disc.pipeline')\n", True),
            ('logging.getLogger("vibe_quant.disc")\n', False),  # a package name, not `-m`
            ('ROOT / "vibe_quant"\n', False),
            ('x = "vibe_quant.disc.pipelinex"\n', False),  # another module, not a prefix
            ('cmd = ["-m", "vibe_quant.other"]\n', False),
        ],
    )
    def test_string_references_of_reached_modules(
        self, tmp_path: Path, text: str, selected: bool
    ) -> None:
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "test_ref.py").write_text(text)
        reached = {"vibe_quant.disc.__main__", "vibe_quant.disc.pipeline"}
        got = swarm_check.dynamic_reference_test_files(reached, tmp_path)
        assert got == (["tests/unit/test_ref.py"] if selected else [])

    def test_a_vibe_quant_change_runs_no_k_group(self) -> None:
        """`-k pipeline` ran test_screening_pipeline.py / test_overfitting_pipeline.py
        (namesakes) for a discovery/pipeline.py change; the selection is references only."""
        plan = _plan(_ROOT, ["vibe_quant/discovery/pipeline.py"])
        assert all("-k" not in args for args, _, _ in plan.commands)
        assert "tests/unit/test_screening_pipeline.py" not in plan.files
        assert "tests/unit/test_symbol_agg.py" in plan.explicit  # the import closure


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

    def test_a_helper_selects_its_importing_test_unfiltered(self, tmp_path: Path) -> None:
        repo = _mini(
            tmp_path,
            {
                "tests/__init__.py": "",
                "tests/unit/__init__.py": "",
                "tests/unit/fixture_bank.py": "X = 2\n",
                "tests/unit/test_contract.py": "from tests.unit.fixture_bank import X\n",
            },
        )
        plan = _plan(repo, ["tests/unit/fixture_bank.py"])
        assert plan.explicit == {"tests/unit/test_contract.py"}
        assert all("-k" not in args for args, _, _ in plan.commands)

    def test_a_runtime_json_goes_full(self, tmp_path: Path) -> None:
        repo = _mini(tmp_path, {"vibe_quant/rules.json": "{}\n", "vibe_quant/reader.py": "X = 1\n"})
        assert _plan(repo, ["vibe_quant/rules.json"]).full

    def test_a_string_patch_target_test_is_selected_by_plan_tests(self, tmp_path: Path) -> None:
        """Nit: plan_tests must keep calling dynamic_reference_test_files - this test has no
        import edge, no path mention and no stem in its name."""
        repo = _mini(
            tmp_path,
            {
                "vibe_quant/foo.py": "def f() -> int:\n    return 1\n",
                "tests/unit/test_patch_only.py": 'TARGET = "vibe_quant.foo.f"\n',
            },
        )
        plan = _plan(repo, ["vibe_quant/foo.py"])
        assert plan.explicit == {"tests/unit/test_patch_only.py"}


class TestRound2Probes:
    """Review rev-gate-speed-2 F1-F3: what the gate does not model goes to the full suite."""

    def test_a_helper_reaching_the_root_conftest_runs_the_full_suite(self, tmp_path: Path) -> None:
        repo = _mini(
            tmp_path,
            {
                "tests/__init__.py": "",
                "tests/unit/__init__.py": "",
                "tests/fixture_bank.py": "X = 2\n",
                "tests/conftest.py": "import pytest\nfrom tests.fixture_bank import X\n",
                "tests/unit/test_contract.py": "def test_value(registered_value): ...\n",
            },
        )
        plan = _plan(repo, ["tests/fixture_bank.py"])
        assert plan.full and "tests/conftest.py" in plan.notes[0]

    def test_a_helper_reaching_a_nested_conftest_runs_its_subtree(self, tmp_path: Path) -> None:
        repo = _mini(
            tmp_path,
            {
                "tests/__init__.py": "",
                "tests/unit/__init__.py": "",
                "tests/unit/api/__init__.py": "",
                "tests/fixture_bank.py": "X = 2\n",
                "tests/unit/api/conftest.py": "from tests.fixture_bank import X\n",
                "tests/unit/api/test_a.py": "def test_a(fx): ...\n",
                "tests/unit/api/deep/test_b.py": "def test_b(fx): ...\n",
                "tests/unit/test_outside.py": "def test_c(): ...\n",
            },
        )
        plan = _plan(repo, ["tests/fixture_bank.py"])
        assert not plan.full
        assert {"tests/unit/api/test_a.py", "tests/unit/api/deep/test_b.py"} <= plan.explicit
        assert "tests/unit/test_outside.py" not in plan.explicit

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

    def test_a_prompt_template_selects_the_tests_that_read_md_files(self) -> None:
        plan = _plan(_ROOT, ["docs/orchestration/prompts/cmd-task.md"])
        assert "tests/unit/test_mkbrief.py" in plan.explicit  # renders the prompts
        assert "tests/unit/test_models_route.py" in plan.explicit  # reads SKILL.md

    def test_an_added_research_source_selects_the_registry_tests(self) -> None:
        """F3 (real tree): research/sources/__init__.py loads its directory by iter_modules."""
        plan = _plan(_ROOT, ["vibe_quant/research/sources/zz_review_duplicate.py"])
        assert not plan.full
        assert "tests/unit/research/test_source_registry_reload.py" in plan.explicit

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


_STORE_FIXTURES = (
    "import pytest\n\n\n"
    "@pytest.fixture\ndef store():\n    from vibe_quant.store import make\n\n    return make()\n\n\n"
    "@pytest.fixture\ndef wrapped(store):\n    return store\n"
)


class TestConftestFixtures:
    """Round-3 chief model: a conftest carries a module into a test only through a fixture."""

    def test_a_non_autouse_fixture_selects_only_its_requesters(self, tmp_path: Path) -> None:
        repo = _fixture_repo(
            tmp_path,
            _STORE_FIXTURES,
            {
                "tests/unit/test_param.py": "def test_a(store):\n    assert store == 1\n",
                "tests/unit/test_transitive.py": "def test_b(wrapped):\n    assert wrapped\n",
                "tests/unit/test_usefix.py": '@pytest.mark.usefixtures("store")\ndef test_c(): ...\n',
                "tests/unit/test_value.py": 'def test_d(request):\n    request.getfixturevalue("store")\n',
                "tests/unit/test_other.py": "def test_e(tmp_path): ...\n",
            },
        )
        plan = _plan(repo, ["vibe_quant/store.py"])
        assert not plan.full
        assert plan.explicit == {
            "tests/unit/test_param.py",
            "tests/unit/test_transitive.py",
            "tests/unit/test_usefix.py",
            "tests/unit/test_value.py",
        }

    def test_a_dynamic_fixture_request_runs_the_subtree(self, tmp_path: Path) -> None:
        repo = _fixture_repo(
            tmp_path,
            _STORE_FIXTURES,
            {
                "tests/unit/test_dyn.py": "def test_d(request, n):\n    request.getfixturevalue(n)\n",
            },
        )
        plan = _plan(repo, ["vibe_quant/store.py"])
        assert plan.full and "dynamically" in plan.notes[0]

    @pytest.mark.parametrize(
        ("body", "full"),
        [
            # patch-only: setattr with a constant name, nothing called on the module
            (
                "    from vibe_quant import lock\n    monkeypatch.setattr(lock, 'LOCK_DIR', '/t')\n",
                False,
            ),
            # calls into the module
            ("    from vibe_quant import lock\n    lock.where()\n", True),
            # uses a return value of the patch / reads the module
            ("    from vibe_quant import lock\n    x = lock.LOCK_DIR\n", True),
            # acts only when the test already loaded it
            (
                "    import sys\n    m = sys.modules.get('vibe_quant.lock')\n    m and m.where()\n",
                False,
            ),
        ],
    )
    def test_autouse_fixtures(self, tmp_path: Path, body: str, full: bool) -> None:
        conftest = (
            "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef guard(monkeypatch):\n" + body
        )
        repo = _fixture_repo(tmp_path, conftest, {"tests/unit/test_x.py": "def test_x(): ...\n"})
        plan = _plan(repo, ["vibe_quant/lock.py"])
        assert plan.full is full, plan.notes
        if "setattr" in body:  # one test of the subtree still runs the patching fixture
            assert plan.explicit == {"tests/unit/test_x.py"}
        if "sys.modules" in body:  # no test loads lock, so nothing depends on it
            assert plan.explicit == set()

    def test_a_module_level_conftest_import_runs_the_full_suite(self, tmp_path: Path) -> None:
        repo = _fixture_repo(
            tmp_path,
            "from vibe_quant import lock\n",
            {
                "tests/unit/test_x.py": "def test_x(): ...\n",
            },
        )
        plan = _plan(repo, ["vibe_quant/lock.py"])
        assert plan.full and "module-level" in plan.notes[0]

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


class TestRound3Probes:
    """Review rev-gate-speed-3 F1-F2: runtime fixture requests and pytest_plugins."""

    _CONSUMERS = {
        "tests/unit/test_consumer.py": "def test_c(wrapped):\n    assert wrapped == 1\n",
        "tests/unit/test_smoke.py": "from vibe_quant.store import make\n",
        "tests/unit/test_other.py": "def test_o(tmp_path): ...\n",
    }
    _STORE = (
        "import pytest\n\n\n@pytest.fixture\ndef store():\n"
        "    from vibe_quant.store import make\n\n    return make()\n\n\n"
    )

    def test_a_constant_getfixturevalue_in_a_fixture_is_an_edge(self, tmp_path: Path) -> None:
        conftest = self._STORE + (
            '@pytest.fixture\ndef wrapped(request):\n    return request.getfixturevalue("store")\n'
        )
        repo = _fixture_repo(tmp_path, conftest, self._CONSUMERS)
        plan = _plan(repo, ["vibe_quant/store.py"])
        assert not plan.full
        assert plan.explicit == {"tests/unit/test_consumer.py", "tests/unit/test_smoke.py"}

    def test_a_constant_getfixturevalue_in_a_referenced_helper_is_an_edge(
        self, tmp_path: Path
    ) -> None:
        conftest = self._STORE + (
            'def _get(request):\n    return request.getfixturevalue("store")\n\n\n'
            "@pytest.fixture\ndef wrapped(request):\n    return _get(request)\n"
        )
        repo = _fixture_repo(tmp_path, conftest, self._CONSUMERS)
        assert "tests/unit/test_consumer.py" in _plan(repo, ["vibe_quant/store.py"]).explicit

    def test_a_keyword_getfixturevalue_in_a_fixture_is_an_edge(self, tmp_path: Path) -> None:
        for i, body in enumerate(
            (
                '    return request.getfixturevalue(argname="store")\n',
                "    return _get(request)\n",  # through a conftest helper
            )
        ):
            conftest = self._STORE + (
                'def _get(request):\n    return request.getfixturevalue(argname="store")\n\n\n'
                "@pytest.fixture\ndef wrapped(request):\n" + body
            )
            repo = _fixture_repo(tmp_path / str(i), conftest, self._CONSUMERS)
            plan = _plan(repo, ["vibe_quant/store.py"])
            assert not plan.full, body
            assert "tests/unit/test_consumer.py" in plan.explicit, body

    @pytest.mark.parametrize(
        "call",
        [
            "request.getfixturevalue(argname=name)",
            "request.getfixturevalue(**{'argname': 'store'})",
            "request.getfixturevalue(*['store'])",
        ],
    )
    def test_a_dynamic_keyword_or_unpacked_request_runs_the_subtree(
        self, tmp_path: Path, call: str
    ) -> None:
        conftest = self._STORE + (
            f"@pytest.fixture\ndef wrapped(request):\n    name = 'store'\n    return {call}\n"
        )
        repo = _fixture_repo(tmp_path, conftest, self._CONSUMERS)
        plan = _plan(repo, ["vibe_quant/store.py"])
        assert plan.full and "getfixturevalue" in plan.notes[0]

    @pytest.mark.parametrize(
        ("call", "dynamic"),
        [
            ('request.getfixturevalue(argname="store")', False),
            ("request.getfixturevalue(argname=n)", True),
            ("request.getfixturevalue(**kw)", True),
            ("request.getfixturevalue(*names)", True),
            ('pytest.mark.usefixtures("store")', False),
            ("pytest.mark.usefixtures(*names)", True),
        ],
    )
    def test_test_files_and_fixtures_accept_the_same_request_forms(
        self, call: str, dynamic: bool
    ) -> None:
        names, is_dynamic = swarm_check._requests(f"def t(request):\n    {call}\n")
        assert is_dynamic is dynamic
        if not dynamic:
            assert "store" in names
        runtime, runtime_dynamic = swarm_check._runtime_requests([ast.parse(call)])
        assert runtime_dynamic is dynamic
        assert ("store" in runtime) is (not dynamic)

    def test_a_variable_getfixturevalue_in_a_fixture_runs_the_subtree(self, tmp_path: Path) -> None:
        conftest = self._STORE + (
            "@pytest.fixture\ndef wrapped(request):\n"
            '    name = "store"\n    return request.getfixturevalue(name)\n'
        )
        repo = _fixture_repo(tmp_path, conftest, self._CONSUMERS)
        plan = _plan(repo, ["vibe_quant/store.py"])
        assert plan.full and "getfixturevalue" in plan.notes[0]

    @pytest.mark.parametrize(
        "decl",
        [
            'pytest_plugins = ("tests.fixture_bank",)\n',
            'pytest_plugins = ["tests.fixture_bank"]\n',
            'pytest_plugins = "tests.fixture_bank"\n',
        ],
    )
    def test_a_reached_registered_plugin_runs_the_full_suite(
        self, tmp_path: Path, decl: str
    ) -> None:
        repo = _fixture_repo(
            tmp_path,
            decl,
            {
                "tests/fixture_bank.py": (
                    "import pytest\n\n\n@pytest.fixture\ndef registered():\n"
                    "    from vibe_quant.store import make\n\n    return make()\n"
                ),
                "tests/unit/test_consumer.py": "def test_c(registered):\n    assert registered == 1\n",
            },
        )
        plan = _plan(repo, ["vibe_quant/store.py"])
        assert plan.full and "pytest_plugins" in plan.notes[0]

    def test_any_pytest_plugins_name_under_tests_runs_the_full_suite(self, tmp_path: Path) -> None:
        """r4 F2 (chief: blunt, no modelling): registration happens after module execution
        (`getattr(mod, "pytest_plugins", [])`), so a conditional/nested/computed one counts.
        Even a plugin the change does not reach goes full: tests/ registers none today."""
        decls = (
            'import os\nif os.environ.get("X") != "1":\n    pytest_plugins = ("tests.fixture_bank",)\n',
            'def _f():\n    global pytest_plugins\n    pytest_plugins = ["tests.x"]\n',
            'import sys\nsys.modules[__name__].pytest_plugins = ["tests.x"]\n',
            'pytest_plugins = ("tests.fixture_bank",)\n',  # its plugin is not reached
        )
        for i, decl in enumerate(decls):
            repo = _fixture_repo(
                tmp_path / str(i),
                decl,
                {
                    "tests/fixture_bank.py": "X = 1\n",
                    "tests/unit/test_smoke.py": "from vibe_quant.store import make\n",
                },
            )
            plan = _plan(repo, ["vibe_quant/store.py"])
            assert plan.full and "pytest_plugins" in plan.notes[0], decl

    def test_a_pytest_plugins_string_is_no_registration(self, tmp_path: Path) -> None:
        repo = _fixture_repo(
            tmp_path,
            'DOC = "pytest_plugins = (...)"\n',
            {
                "tests/unit/test_smoke.py": "from vibe_quant.store import make\n",
            },
        )
        assert not _plan(repo, ["vibe_quant/store.py"]).full

    def test_a_computed_plugin_registration_runs_the_full_suite(self, tmp_path: Path) -> None:
        repo = _fixture_repo(
            tmp_path,
            'NAMES = ["tests.x"]\npytest_plugins = NAMES\n',
            {
                "tests/unit/test_smoke.py": "from vibe_quant.store import make\n",
            },
        )
        plan = _plan(repo, ["vibe_quant/store.py"])
        assert plan.full and "pytest_plugins" in plan.notes[0]


class TestKCandidates:
    def test_only_files_that_can_match_are_collected(self, tmp_path: Path) -> None:
        unit = tmp_path / "tests" / "unit"
        (unit / "sub").mkdir(parents=True)
        (unit / "test_a.py").write_text("def test_Pipeline_runs() -> None: ...\n")  # case
        (unit / "test_pipeline_x.py").write_text("def test_x() -> None: ...\n")  # path
        (unit / "sub" / "test_c.py").write_text("def test_fitness() -> None: ...\n")
        (unit / "test_d.py").write_text("def test_other() -> None: ...\n")
        got = swarm_check.k_candidate_files(tmp_path, "pipeline or fitness", set())
        assert got == [
            "tests/unit/sub/test_c.py",
            "tests/unit/test_a.py",
            "tests/unit/test_pipeline_x.py",
        ]
        excluded = {"tests/unit/test_a.py"}
        assert "tests/unit/test_a.py" not in swarm_check.k_candidate_files(
            tmp_path, "pipeline", excluded
        )
        assert swarm_check.k_candidate_files(tmp_path, "", set()) == []


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


class TestListTests:
    def test_list_tests_prints_the_selection_and_runs_no_tool(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = tmp_path / "wt"
        (repo / "vibe_quant").mkdir(parents=True)
        (repo / "tests" / "unit").mkdir(parents=True)
        (repo / "vibe_quant" / "foo.py").write_text("F = 1\n")
        (repo / "tests" / "unit" / "test_caller.py").write_text("from vibe_quant.foo import F\n")
        (repo / "tests" / "unit" / "test_none.py").write_text("X = 1\n")
        _git(tmp_path, "init", "-q", "-b", "main", str(repo))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        (repo / "vibe_quant" / "foo.py").write_text("F = 2\n")
        calls: list[list[str]] = []

        def run(cmd: object, cwd: Path, env: object = None, timeout: int = 0) -> tuple[int, str]:
            c = [str(x) for x in cmd]  # type: ignore[attr-defined]
            calls.append(c)
            proc = subprocess.run(c, cwd=str(cwd), capture_output=True, text=True)
            return proc.returncode, proc.stdout + proc.stderr

        rc = swarm_check.main([str(repo), "--base", "HEAD", "--list-tests"], run=run)
        out, err = capsys.readouterr()
        assert rc == 0
        assert out.splitlines() == ["tests/unit/test_caller.py"]
        assert "# pytest <1 files> -n auto --dist worksteal -m not timing" in err
        assert all(c[0] == "git" for c in calls)  # no ruff/mypy/pytest ran


def test_xdist_floor_is_half_the_cores_under_load() -> None:
    """3 concurrent gates at load 10 got 2 workers each and crawled (bead q2ijs)."""
    workers = swarm_check.xdist_auto_workers
    assert workers({}, 5.0, 10) is None
    assert workers({}, 5.1, 10) == "5"
    assert workers({}, 29.5, 10) == "5"
    assert workers({}, 2.0, 10) is None
    assert workers({}, 1.0, 4) is None
    assert workers({}, 3.0, 4) == "2"
