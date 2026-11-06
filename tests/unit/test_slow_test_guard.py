"""The pre-review gate's 2 s rule: a NEW or CHANGED unit test that takes > 2 s alone fails
the gate unless it carries `@pytest.mark.slow(reason=...)` (job 20261105-prune2 lane G, user
decision 2026-11-05). tests/unit grew +395 tests in a week after the prune; the serial suite
is 754 s against a ~120 s goal, so a new slow test is a decision, not an accident.

Per-test times come from a junit xml each pytest call writes (`--junitxml` in that call's
PYTEST_ADDOPTS, so the argv the gate passes stays byte-identical), the "alone" time from a
serial re-run of the reported node id (`-p no:randomly -n0`, at most 5, slowest first). A test
unchanged since the base is existing debt and is never judged; timing that cannot be read
fails closed. Tests call slow_tests_check directly: run_checks costs ~0.8 s of real git.
"""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

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

JUNIT_OPT = "--junitxml="


def _git(repo: Path, *args: str) -> None:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _repo(tmp_path: Path, base_test: str = "def test_base() -> None:\n    pass\n") -> Path:
    """A git repo whose HEAD holds one committed test; every later edit is the diff."""
    repo = tmp_path / "wt"
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / "vibe_quant").mkdir()
    (repo / "vibe_quant" / "keep.py").write_text("X = 1\n")
    (repo / "tests" / "unit" / "test_base.py").write_text(base_test)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=m@example.com", "-c", "user.name=m", "commit", "-qm", "base")
    return repo


def _cfg(repo: Path) -> object:
    return swarm_check.Config(
        worktree=repo, main_root=repo, base="HEAD", exactness="never", frontend="never"
    )


def _classname(file: str, cls: str) -> str:
    module = file[: -len(".py")].replace("/", ".") if file.endswith(".py") else file
    return f"{module}.{cls}" if cls else module


def _junit_rows(path: Path, rows: list[tuple[str, float]]) -> None:
    """Write a pytest junit xml: one testcase per node id, `time` = the test's own duration."""
    cases = []
    for node_id, secs in rows:
        parts = node_id.split("::")
        file, name = parts[0], parts[-1]
        cls = parts[1] if len(parts) == 3 else ""
        cases.append(
            f'<testcase classname="{_classname(file, cls)}" name="{name}" time="{secs}"/>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest" tests="{len(cases)}">{"".join(cases)}</testsuite></testsuites>'
    )


def _junit_path(cmd: Sequence[str], env: Mapping[str, str] | None) -> Path | None:
    args = [*cmd, *shlex.split((env or {}).get("PYTEST_ADDOPTS", ""))]
    for arg in args:
        if str(arg).startswith(JUNIT_OPT):
            return Path(str(arg)[len(JUNIT_OPT):])
    return None


class Runner:
    """git delegates to the real binary; a pytest re-run of one node id writes a junit xml with
    the declared time. `alone` maps node id -> seconds in the serial re-run (default fast, i.e.
    load noise); `alone_rc` / `alone_xml` break the re-run (non-zero exit / no report)."""

    def __init__(
        self,
        alone: dict[str, float] | None = None,
        *,
        alone_rc: int = 0,
        alone_xml: bool = True,
    ) -> None:
        self.alone = alone or {}
        self.alone_rc = alone_rc
        self.alone_xml = alone_xml
        self.calls: list[list[str]] = []
        self.timeouts: list[int] = []

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: int = 0,
    ) -> tuple[int, str]:
        c = [str(x) for x in cmd]
        self.calls.append(c)
        if c[0] == "git":
            proc = subprocess.run(c, cwd=str(cwd), capture_output=True, text=True, check=False)
            return proc.returncode, proc.stdout + proc.stderr
        self.timeouts.append(timeout)
        node = next(a for a in c if "::" in a)
        xml = _junit_path(c, env)
        if xml is not None and self.alone_xml:
            _junit_rows(xml, [(node, self.alone.get(node, 0.01))])
        if self.alone_rc:
            return self.alone_rc, "ERROR at setup of test_slow\nFixture boom"
        return 0, "1 passed in 0.01s"


BASE = "tests/unit/test_base.py"
NEW_FILE = "tests/unit/test_new.py"
NEW_SLOW = "def test_slow() -> None:\n    time.sleep(2.5)\n"
NODE = f"{NEW_FILE}::test_slow"


def _state(*changed: str) -> object:
    """The diff the gate would see, declared (the real git_state costs 4 spawns per test and
    is exercised by the two wiring tests): NEW_FILE is the only untracked path."""
    return swarm_check.GitState(list(changed), [p for p in changed if p == NEW_FILE])


def _slow(
    repo: Path,
    runner: Callable[..., tuple[int, str]],
    gate: dict[str, float],
    gate_xml: str = "ok",
    changed: tuple[str, ...] = (NEW_FILE,),
) -> object:
    """The slow-tests check over a gate junit holding `gate` (node id -> seconds); `gate_xml`
    breaks that report: "missing", "malformed", "foreign" (valid rows of other tests)."""
    xml = repo.parent / "gate.xml"
    if gate_xml == "malformed":
        xml.write_text("<testsuites><broken")
    elif gate_xml != "missing":
        rows = [("tests/unit/test_other.py::test_other", 0.01)] if gate_xml == "foreign" else gate
        _junit_rows(xml, list(rows.items()) if isinstance(rows, dict) else rows)
    return swarm_check.slow_tests_check(
        _cfg(repo), _state(*changed), {"pytest": ["pytest"]}, runner, {}, [xml]
    )


def _alone_runs(runner: Runner) -> list[list[str]]:
    return [c for c in runner.calls if c[0] != "git"]



def test_an_added_slow_test_fails_and_names_it_with_both_seconds(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    slow = _slow(repo, Runner({NODE: 2.6}), {NODE: 2.5})
    assert slow.status == "FAIL", slow.detail
    assert NODE in slow.detail
    assert "2.5" in slow.detail and "2.6" in slow.detail
    assert '@pytest.mark.slow(reason="...")' in slow.detail


def test_an_added_slow_test_marked_with_a_reason_passes_and_is_listed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(
        'import pytest\n\n\n@pytest.mark.slow(reason="needs a real NT run")\n'
        "def test_slow() -> None:\n    time.sleep(2.5)\n"
    )
    runner = Runner()
    slow = _slow(repo, runner, {NODE: 2.5})
    assert slow.status == "PASS", slow.detail
    assert NODE in slow.detail
    assert _alone_runs(runner) == []


def test_a_slow_mark_with_no_reason_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    for decorator in ("@pytest.mark.slow\n", '@pytest.mark.slow(reason="")\n'):
        (repo / NEW_FILE).write_text(
            f"import pytest\n\n\n{decorator}def test_slow() -> None:\n    time.sleep(2.5)\n"
        )
        slow = _slow(repo, Runner({NODE: 2.6}), {NODE: 2.5})
        assert slow.status == "FAIL", slow.detail
        assert NODE in slow.detail and "reason" in slow.detail, decorator


def test_a_test_slow_only_under_the_parallel_gate_passes_as_load_noise(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    runner = Runner({NODE: 0.4})
    slow = _slow(repo, runner, {NODE: 2.5})
    assert slow.status == "PASS", slow.detail
    assert NODE in slow.detail
    assert "2.5" in slow.detail and "0.4" in slow.detail
    run = _alone_runs(runner)
    assert len(run) == 1
    assert "-n0" in run[0] and "no:randomly" in run[0] and runner.timeouts == [60]


def test_an_unchanged_slow_test_in_a_changed_file_is_left_alone(tmp_path: Path) -> None:
    """Existing debt: only the function the diff touched is judged, not its file or others."""
    old = "import time\n\n\ndef test_old_slow() -> None:\n    time.sleep(3.0)\n\n\n"
    repo = _repo(tmp_path, old + "def test_fast() -> None:\n    assert 1\n")
    (repo / "tests" / "unit" / "test_base.py").write_text(
        old + "def test_fast() -> None:\n    assert 2\n"
    )
    runner = Runner()
    slow = _slow(
        repo,
        runner,
        {f"{BASE}::test_old_slow": 3.0, f"{BASE}::test_fast": 0.01},
        changed=(BASE,),
    )
    assert slow.status == "PASS", slow.detail
    assert "test_old_slow" not in slow.detail
    assert _alone_runs(runner) == []


def test_a_changed_parametrized_test_judges_every_row(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(
        "import pytest\n\n\n@pytest.mark.parametrize('n', [1, 2])\n"
        "def test_slow(n: int) -> None:\n    time.sleep(2.5)\n"
    )
    slow = _slow(
        repo,
        Runner({f"{NODE}[1]": 0.1, f"{NODE}[2]": 2.6}),
        {f"{NODE}[1]": 2.4, f"{NODE}[2]": 2.7},
    )
    assert slow.status == "FAIL", slow.detail
    assert f"{NODE}[2] 2.7s in the gate, 2.6s alone" in slow.detail  # slowest row re-run first


@pytest.mark.parametrize(
    ("base", "head", "node"),
    [
        (
            "import pytest\n@pytest.mark.parametrize('x',[1])\ndef test_base(x):\n    pass\n",
            "import pytest\n@pytest.mark.parametrize('x',[1,2])\ndef test_base(x):\n    pass\n",
            "test_base[2]",
        ),
        (
            "class TestA:\n    def test_base(self):\n        assert 1\n",
            "class TestA:\n    def test_base(self):\n        assert 2\n",
            "TestA::test_base",
        ),
        ("def test_base():\n    pass\n", "def test_renamed():\n    pass\n", "test_renamed"),
    ],
    ids=["decorator", "class-method", "rename"],
)
def test_a_changed_decorator_class_method_or_rename_is_judged(
    tmp_path: Path, base: str, head: str, node: str
) -> None:
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text(head)
    full = f"{BASE}::{node}"
    slow = _slow(repo, Runner({full: 3.0}), {full: 3.0}, changed=(BASE,))
    assert slow.status == "FAIL", slow.detail
    assert full in slow.detail


def test_no_python_test_changed_is_a_skip(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "vibe_quant" / "keep.py").write_text("X = 2\n")
    slow = _slow(repo, Runner(), {}, changed=("vibe_quant/keep.py",))
    assert slow.status == "SKIP", slow.detail
    assert "no python test changed" in slow.detail


def test_a_changed_test_with_no_usable_timing_report_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    for mode, wanted in (("missing", "no usable junit"), ("malformed", "no usable junit"), ("foreign", NODE)):
        runner = Runner()
        slow = _slow(repo, runner, {NODE: 3.0}, gate_xml=mode)
        assert slow.status == "FAIL", f"{mode}: {slow.detail}"
        assert wanted in slow.detail, mode
        assert _alone_runs(runner) == [], mode


def test_a_testcase_with_an_invalid_time_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    xml = tmp_path / "gate.xml"
    runner = Runner()
    for bad in ('time="-1"', 'time="nan"', 'time="x"', ""):
        xml.write_text(
            f'<testsuites><testcase classname="tests.unit.test_new" name="test_slow" {bad}/></testsuites>'
        )
        slow = swarm_check.slow_tests_check(
            _cfg(repo), _state(NEW_FILE), {"pytest": ["pytest"]}, runner, {}, [xml]
        )
        assert slow.status == "FAIL", f"{bad}: {slow.detail}"
        assert "valid time" in slow.detail, bad


def test_a_serial_rerun_that_fails_or_reports_nothing_is_a_fail_not_noise(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    for rc, has_xml, why in ((1, True, "exited 1"), (1, False, "exited 1"), (124, False, "exited 124"), (0, False, "junit")):
        runner = Runner({NODE: 0.01}, alone_rc=rc, alone_xml=has_xml)
        slow = _slow(repo, runner, {NODE: 3.0})
        assert slow.status == "FAIL", f"rc={rc} xml={has_xml}: {slow.detail}"
        assert NODE in slow.detail and why in slow.detail, (rc, has_xml)
        assert "load noise" not in slow.detail


def test_deleting_a_line_of_an_existing_slow_test_makes_it_a_changed_test(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        "import time\n\n\ndef test_base() -> None:\n    assert True\n    time.sleep(3)\n",
    )
    (repo / BASE).write_text(
        "import time\n\n\ndef test_base() -> None:\n    time.sleep(3)\n"
    )
    node = f"{BASE}::test_base"
    slow = _slow(repo, Runner({node: 3.0}), {node: 3.0}, changed=(BASE,))
    assert slow.status == "FAIL", slow.detail
    assert node in slow.detail


def test_deleting_a_neighbour_test_does_not_judge_the_unchanged_slow_one(tmp_path: Path) -> None:
    keep = "def test_old_slow() -> None:\n    time.sleep(3)\n"
    repo = _repo(tmp_path, f"import time\n\n\ndef test_gone() -> None:\n    pass\n\n\n{keep}")
    (repo / BASE).write_text(f"import time\n\n\n{keep}")
    node = f"{BASE}::test_old_slow"
    runner = Runner({node: 3.0})
    slow = _slow(repo, runner, {node: 3.0}, changed=(BASE,))
    assert slow.status == "SKIP", slow.detail
    assert _alone_runs(runner) == []


_DECORATED = (
    "import pytest\nimport time\n\n\n"
    '@pytest.mark.slow(reason="needs a real NT run")\ndef test_x() -> None:\n    time.sleep(3)\n'
)


def _changed_names(repo: Path, changed: tuple[str, ...] = (BASE,)) -> list[str]:
    found = swarm_check.changed_test_functions(_cfg(repo), _state(*changed), Runner())
    return [f"{t.cls + '::' if t.cls else ''}{t.name}" for t in found]


def test_deleting_only_the_slow_mark_of_an_existing_test_judges_it(tmp_path: Path) -> None:
    """The hunk's head line is the blank line ABOVE the decorator: the next test is the one
    that lost its mark, so it is judged (and fails when it is slow and now unmarked)."""
    repo = _repo(tmp_path, _DECORATED)
    (repo / BASE).write_text(_DECORATED.replace('@pytest.mark.slow(reason="needs a real NT run")\n', ""))
    assert _changed_names(repo) == ["test_x"]
    node = f"{BASE}::test_x"
    slow = _slow(repo, Runner({node: 3.0}), {node: 3.0}, changed=(BASE,))
    assert slow.status == "FAIL", slow.detail
    assert node in slow.detail


def test_deleting_the_slow_mark_of_the_first_test_in_the_file_judges_it(tmp_path: Path) -> None:
    base = '@pytest.mark.slow(reason="r")\ndef test_x() -> None:\n    time.sleep(3)\n'
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text("def test_x() -> None:\n    time.sleep(3)\n")
    assert _changed_names(repo) == ["test_x"]


def test_replacing_the_slow_mark_with_a_comment_judges_the_test(tmp_path: Path) -> None:
    """A hunk that removes the decorator AND adds a line above the def is not a bare deletion."""
    repo = _repo(tmp_path, _DECORATED)
    (repo / BASE).write_text(
        _DECORATED.replace('@pytest.mark.slow(reason="needs a real NT run")', "# was slow")
    )
    assert _changed_names(repo) == ["test_x"]


def test_deleting_the_slow_mark_of_a_second_test_leaves_the_first_alone(tmp_path: Path) -> None:
    first = "def test_a() -> None:\n    pass\n\n\n"
    repo = _repo(tmp_path, "import pytest\n\n\n" + first + _DECORATED.split("\n\n\n", 1)[1])
    (repo / BASE).write_text(
        "import pytest\n\n\n" + first + "def test_x() -> None:\n    time.sleep(3)\n"
    )
    assert _changed_names(repo) == ["test_x"]


def test_deleting_the_slow_mark_of_a_test_class_judges_its_methods(tmp_path: Path) -> None:
    base = (
        '@pytest.mark.slow(reason="r")\nclass TestA:\n    def test_x(self) -> None:\n'
        "        time.sleep(3)\n"
    )
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text(base.split("\n", 1)[1])
    assert _changed_names(repo) == ["TestA::test_x"]


def test_deleting_a_non_decorator_line_judges_only_the_test_it_ends(tmp_path: Path) -> None:
    """A removed trailing line of test_a maps to test_a (its enclosing span); the next test_x is
    not judged, so the neighbour stays existing debt."""
    base = "def test_a() -> None:\n    pass\n    # note\n\n\ndef test_x() -> None:\n    time.sleep(3)\n"
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text(base.replace("    # note\n", ""))
    assert _changed_names(repo) == ["test_a"]


def test_deleting_a_blank_line_between_tests_judges_no_neighbour(tmp_path: Path) -> None:
    base = "def test_a() -> None:\n    pass\n\n\n\ndef test_x() -> None:\n    time.sleep(3)\n"
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text(base.replace("\n\n\n\n", "\n\n\n", 1))
    assert "test_x" not in _changed_names(repo)


def test_deleting_a_whole_decorated_neighbour_does_not_judge_the_next_test(tmp_path: Path) -> None:
    """The removed lines hold the neighbour's decorator AND its def: that test is gone, the
    unchanged slow one below it is existing debt."""
    keep = "def test_old_slow() -> None:\n    time.sleep(3)\n"
    gone = "@pytest.mark.slow(reason='r')\ndef test_gone() -> None:\n    pass\n\n\n"
    repo = _repo(tmp_path, f"import time\n\n\n{gone}{keep}")
    (repo / BASE).write_text(f"import time\n\n\n{keep}")
    assert _changed_names(repo) == []


_MARK = '@pytest.mark.slow(reason="r")\n'


@pytest.mark.parametrize(
    ("base", "head", "expected"),
    [
        pytest.param(
            f"import pytest\n\n{_MARK}# retained\ndef test_x():\n    pass\n",
            "import pytest\n\n# retained\ndef test_x():\n    pass\n",
            "test_x",
            id="comment-between-decorator-and-def",
        ),
        pytest.param(
            f"import pytest\n\n{_MARK}\ndef test_x():\n    pass\n",
            "import pytest\n\n\ndef test_x():\n    pass\n",
            "test_x",
            id="blank-between-decorator-and-def",
        ),
        pytest.param(
            f"import pytest\n\n{_MARK}# retained\nclass TestA:\n    def test_x(self):\n        pass\n",
            "import pytest\n\n# retained\nclass TestA:\n    def test_x(self):\n        pass\n",
            "TestA::test_x",
            id="comment-before-class",
        ),
        pytest.param(
            f"import pytest\n\ndef test_gone():\n    pass\n\n{_MARK}def test_x():\n    pass\n",
            "import pytest\n\ndef test_x():\n    pass\n",
            "test_x",
            id="neighbour-removed-in-same-hunk",
        ),
        pytest.param(
            "import pytest\n\n@pytest.mark.parametrize('a', [1])\n"
            f"{_MARK}# why\ndef test_x(a):\n    pass\n",
            "import pytest\n\n@pytest.mark.parametrize('a', [1])\n# why\ndef test_x(a):\n    pass\n",
            "test_x",
            id="second-decorator-removed-keeps-first",
        ),
        pytest.param(
            f"import pytest\n\nclass TestA:\n    {_MARK}    def test_x(self):\n        pass\n",
            "import pytest\n\nclass TestA:\n    def test_x(self):\n        pass\n",
            "TestA::test_x",
            id="first-method-of-class",
        ),
    ],
)
def test_a_deleted_slow_mark_judges_the_test_that_survives_it(
    tmp_path: Path, base: str, head: str, expected: str
) -> None:
    """The surviving def/class header is found across blank/comment trivia and across a removed
    neighbour in the same hunk: its slow mark is gone, so it is judged (and fails when slow)."""
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text(head)
    assert _changed_names(repo) == [expected]
    node = f"{BASE}::{expected}"
    slow = _slow(repo, Runner({node: 3.0}), {node: 3.0}, changed=(BASE,))
    assert slow.status == "FAIL", slow.detail
    assert node in slow.detail


def test_removing_a_neighbour_and_an_unrelated_decorator_judges_only_the_marked_test(
    tmp_path: Path,
) -> None:
    """test_gone (with its own decorator) goes in the same hunk as test_x's decorator: test_x is
    judged, the unchanged slow test below it is not."""
    old_slow = "def test_old_slow():\n    time.sleep(3)\n"
    base = (
        f"import pytest\n\n{_MARK}def test_gone():\n    pass\n\n{_MARK}def test_x():\n    pass\n\n\n{old_slow}"
    )
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text(f"import pytest\n\ndef test_x():\n    pass\n\n\n{old_slow}")
    assert _changed_names(repo) == ["test_x"]


def test_a_one_line_neighbour_directly_above_the_deleted_mark_does_not_hide_the_test(
    tmp_path: Path,
) -> None:
    """The hunk's head line is the neighbour's own `def` line: the survivor search starts after
    it, not at it, so test_x (not only test_a) is judged."""
    base = f"import pytest\n\ndef test_a(): pass\n{_MARK}def test_x():\n    pass\n"
    repo = _repo(tmp_path, base)
    (repo / BASE).write_text(base.replace(_MARK, ""))
    assert "test_x" in _changed_names(repo)


def _many_slow(tmp_path: Path, count: int) -> tuple[Path, dict[str, float]]:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text("".join(f"def test_{i}() -> None:\n    pass\n\n\n" for i in range(count)))
    return repo, {f"{NEW_FILE}::test_{i}": 3.0 + i / 100 for i in range(count)}


def test_a_confirmed_slow_test_stops_the_reruns_and_lists_the_rest(tmp_path: Path) -> None:
    repo, rows = _many_slow(tmp_path, 30)
    runner = Runner(rows)
    slow = _slow(repo, runner, rows)
    assert slow.status == "FAIL", slow.detail
    assert len(_alone_runs(runner)) == 1
    assert f"{NEW_FILE}::test_29 " in slow.detail  # the slowest candidate is re-run first
    assert "> 2 s in gate run, not re-run (cap 5)" in slow.detail
    assert "(+24 more)" in slow.detail


def test_all_noise_reruns_are_capped_and_the_rest_fails_unverified(tmp_path: Path) -> None:
    repo, rows = _many_slow(tmp_path, 8)
    runner = Runner({node: 0.01 for node in rows})
    slow = _slow(repo, runner, rows)
    assert slow.status == "FAIL", slow.detail
    assert len(_alone_runs(runner)) == 5
    assert "> 2 s in gate run, not re-run (cap 5)" in slow.detail
    assert f"{NEW_FILE}::test_2" in slow.detail  # the 3 slowest-last are unverified
    assert f"{NEW_FILE}::test_3" not in slow.detail  # re-run, noise: not listed


def test_a_real_pytest_junit_maps_and_judges_a_new_slow_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real git and a real pytest junit of a planted test that really sleeps (threshold lowered
    to 0.2 s; the child runs xdist only, no plugin autoload, so this test stays far below the
    2 s rule it enforces). The serial re-run hands back that same real report."""
    monkeypatch.setattr(swarm_check, "SLOW_TEST_SECONDS", 0.2)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text("import time\n\n\ndef test_slow() -> None:\n    time.sleep(0.22)\n")
    xml = tmp_path / "gate.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "xdist", NEW_FILE, "-q", "-n0", f"{JUNIT_OPT}{xml}"],
        cwd=repo, capture_output=True, check=True,
    )

    def rerun(cmd, cwd, env=None, timeout=0):  # type: ignore[no-untyped-def]
        if cmd[0] == "git":
            return swarm_check.real_run(cmd, cwd, env, timeout)
        shutil.copy(xml, _junit_path(cmd, env) or xml)
        return 0, "1 passed"

    slow = swarm_check.slow_tests_check(
        _cfg(repo), _state(NEW_FILE), {"pytest": ["pytest"]}, rerun, {}, [xml]
    )
    assert slow.status == "FAIL", slow.detail
    assert NODE in slow.detail and "alone" in slow.detail


def test_the_gate_run_writes_junit_through_the_env_and_feeds_the_slow_check(tmp_path: Path) -> None:
    """The gate's pytest argv must not grow: the junit file goes through PYTEST_ADDOPTS."""
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    alone = Runner({NODE: 2.6})
    gate_calls: list[list[str]] = []

    def runner(cmd, cwd, env=None, timeout=0):  # type: ignore[no-untyped-def]
        c = [str(x) for x in cmd]
        if c[0] == "git" or any("::" in a for a in c):
            return alone(cmd, cwd, env, timeout)
        if "pytest" in c:
            gate_calls.append(c)
            xml = _junit_path(c, env)
            assert xml is not None
            _junit_rows(xml, [(NODE, 2.5)])
            return 0, "1 passed in 0.01s"
        return 0, "ok"

    checks = {c.name: c for c in swarm_check.run_checks(_cfg(repo), runner)}
    assert checks["slow-tests"].status == "FAIL", checks["slow-tests"].detail
    assert NODE in checks["slow-tests"].detail
    assert gate_calls
    assert all(not any(a.startswith(JUNIT_OPT) for a in c) for c in gate_calls)
