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
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
    load noise), `then` -> seconds in any later re-run of it; `alone_rc` / `alone_xml` break
    the re-run (non-zero exit / no report)."""

    def __init__(
        self,
        alone: dict[str, float] | None = None,
        *,
        then: dict[str, float] | None = None,
        alone_rc: int = 0,
        alone_xml: bool = True,
    ) -> None:
        self.alone = alone or {}
        self.then = then or {}
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
            again = sum(node in call for call in self.calls[:-1])
            secs = self.then[node] if again and node in self.then else self.alone.get(node, 0.01)
            _junit_rows(xml, [(node, secs)])
        if self.alone_rc:
            return self.alone_rc, "ERROR at setup of test_slow\nFixture boom"
        return 0, "1 passed in 0.01s"


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


def test_a_changed_test_with_no_usable_timing_report_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    for mode, wanted in (("missing", "no usable junit"), ("malformed", "no usable junit"), ("foreign", NODE)):
        runner = Runner()
        slow = _slow(repo, runner, {NODE: 3.0}, gate_xml=mode)
        assert slow.status == "FAIL", f"{mode}: {slow.detail}"
        assert wanted in slow.detail, mode
        assert _alone_runs(runner) == [], mode


def test_a_serial_rerun_that_fails_or_reports_nothing_is_a_fail_not_noise(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    for rc, has_xml, why in ((1, True, "exited 1"), (1, False, "exited 1"), (124, False, "exited 124"), (0, False, "junit")):
        runner = Runner({NODE: 0.01}, alone_rc=rc, alone_xml=has_xml)
        slow = _slow(repo, runner, {NODE: 3.0})
        assert slow.status == "FAIL", f"rc={rc} xml={has_xml}: {slow.detail}"
        assert NODE in slow.detail and why in slow.detail, (rc, has_xml)
        assert "load noise" not in slow.detail


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


SETUP_SLOW = (
    "import time\n\nimport pytest\n\n\n"
    "@pytest.fixture(scope='module')\ndef built() -> None:\n    time.sleep(0.3)\n\n\n"
    "def test_slow(built: None) -> None:\n    pass\n"
)
# -n0 needs xdist; every other plugin only costs start-up in the real re-runs below
LEAN_PYTEST = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_PLUGINS": "xdist.plugin"}
BODY_SLOW = "import time\n\n\ndef test_slow() -> None:\n    time.sleep(0.3)\n"


def _real_alone(tmp_path: Path, monkeypatch: object, source: str) -> object:
    """slow-tests over a REAL serial pytest re-run of NODE (bar lowered to 0.2 s, plugins cut
    to xdist so the case stays cheap): what the alone time counts is pytest's doing."""
    monkeypatch.setattr(swarm_check, "SLOW_TEST_SECONDS", 0.2)  # type: ignore[attr-defined]
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(source)
    tools = {"pytest": [sys.executable, "-m", "pytest"]}
    return swarm_check.slow_tests_check(
        _cfg(repo), _state(NEW_FILE), tools, swarm_check.real_run, LEAN_PYTEST, [_gate_xml(repo, 3.0)]
    )


def _gate_xml(repo: Path, secs: float) -> Path:
    xml = repo.parent / "gate.xml"
    _junit_rows(xml, [(NODE, secs)])
    return xml


def test_a_slow_module_fixture_with_a_fast_body_is_not_slow_alone(tmp_path: Path, monkeypatch: object) -> None:
    slow = _real_alone(tmp_path, monkeypatch, SETUP_SLOW)
    assert slow.status == "PASS", slow.detail
    assert "load noise" in slow.detail and NODE in slow.detail


def test_a_slow_test_body_still_fails_alone(tmp_path: Path, monkeypatch: object) -> None:
    slow = _real_alone(tmp_path, monkeypatch, BODY_SLOW)
    assert slow.status == "FAIL", slow.detail
    assert NODE in slow.detail and "alone" in slow.detail


def test_a_first_alone_time_over_the_bar_is_measured_again_and_the_minimum_judged(tmp_path: Path) -> None:
    """Host load inflates one re-run (2.3-5.9 s for 0.15-0.6 s tests); the second decides."""
    repo = _repo(tmp_path)
    (repo / NEW_FILE).write_text(NEW_SLOW)
    runner = Runner({NODE: 2.6}, then={NODE: 0.4})
    slow = _slow(repo, runner, {NODE: 3.0})
    assert slow.status == "PASS", slow.detail
    assert len(_alone_runs(runner)) == 2
    assert "2.6" in slow.detail and "0.4" in slow.detail and "load" in slow.detail
