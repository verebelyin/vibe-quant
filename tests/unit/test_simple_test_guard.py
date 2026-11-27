"""The gate's simple-target rule (CLAUDE.md § Tests prove outcomes): a NEW test whose run
touches only functions with cognitive complexity <= 5 that are not kill-zone/money code and
hold no regex/SQL fails the gate, unless it carries `@pytest.mark.simple_target(reason=...)`.

Each test drives simple_tests_check against a tmp git repo with real pytest, coverage and the
pinned cccc, so the verdict is the one a worker's gate would print.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

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

TOOL = """\
import re


def double(x):
    return x * 2


def parse(text):
    return re.findall(r"\\d+", text)


def route(kind, n):
    if kind == "a":
        for i in range(n):
            if i % 2:
                if i > 3:
                    return i
    elif kind == "b":
        while n:
            n -= 1
            if n == 2 or n == 5:
                return n
    return None
"""
HEADER = """\
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("tool", Path(__file__).parents[2] / "scripts/agents/tool.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_base():
    assert tool.double(1) == 2
"""
NEW_FILE = "tests/unit/test_tool.py"


def _git(repo: Path, *args: str) -> None:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _check(tmp_path: Path, new_tests: str) -> object:
    """simple_tests_check over a repo whose base holds test_base; `new_tests` is the diff."""
    repo = tmp_path / "wt"
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / "scripts" / "agents").mkdir(parents=True)
    (repo / "scripts" / "agents" / "tool.py").write_text(TOOL)
    shutil.copy2(_ROOT / swarm_check.CCCC_SCRIPT, repo / swarm_check.CCCC_SCRIPT)
    (repo / NEW_FILE).write_text(HEADER)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=m@example.com", "-c", "user.name=m", "commit", "-qm", "base")
    (repo / NEW_FILE).write_text(HEADER + new_tests)
    cfg = swarm_check.Config(
        worktree=repo, main_root=repo, base="HEAD", exactness="never", frontend="never"
    )
    state = swarm_check.GitState([NEW_FILE], [])
    tools = {"pytest": [sys.executable, "-m", "pytest"]}
    return swarm_check.simple_tests_check(cfg, state, tools, swarm_check.real_run, {})


def test_a_new_test_of_only_simple_code_fails_and_a_reasoned_mark_keeps_it(tmp_path: Path) -> None:
    new = "\n\ndef test_double():\n    assert tool.double(3) == 6\n"
    check = _check(tmp_path / "plain", new)
    assert check.status == "FAIL"
    assert f"{NEW_FILE}::test_double (scripts/agents/tool.py:double cog=0)" in check.detail
    assert "test_base" not in check.detail  # unchanged since the base: not judged
    marked = '\n\n@pytest.mark.simple_target(reason="pins the API")\ndef test_double():\n    assert tool.double(3) == 6\n'
    assert _check(tmp_path / "marked", marked).status == "SKIP"


def test_complex_or_regex_code_passes(tmp_path: Path) -> None:
    new = (
        "\n\ndef test_route():\n    assert tool.route('b', 9) == 5\n"
        "\n\ndef test_parse():\n    assert tool.parse('a1b22') == ['1', '22']\n"
    )
    check = _check(tmp_path, new)
    assert (check.status, check.detail) == ("PASS", "2 new tests, none only on simple code")


def test_a_failing_new_test_fails_closed(tmp_path: Path) -> None:
    check = _check(tmp_path, "\n\ndef test_bad():\n    assert tool.double(1) == 3\n")
    assert check.status == "FAIL"
    assert "the coverage run of 1 new tests exited 1" in check.detail


def test_coverage_decode_runs_under_the_gate_interpreter(tmp_path: Path) -> None:
    """swarm-check.sh runs swarm_check.py with the system python3, which has no `coverage`:
    decoding a line_bits row must still work there (it crashed the gate on 2026-11-20)."""
    import sqlite3

    db = tmp_path / "cov.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        "create table file (id integer, path text);"
        "create table context (id integer, context text);"
        "create table line_bits (file_id integer, context_id integer, numbits blob);"
    )
    con.execute("insert into file values (1, ?)", (str(tmp_path / "tool.py"),))
    con.execute("insert into context values (1, 'tests/test_t.py::test_x[1]|run')")
    con.execute("insert into line_bits values (1, 1, ?)", (bytes([0b00000110, 0b10000001]),))
    con.commit()
    con.close()
    code = (
        "import importlib.util, sys; from pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('sc', {str(_SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['sc'] = m; spec.loader.exec_module(m)\n"
        f"out = m._covered_by_test(Path({str(db)!r}), [Path({str(tmp_path)!r})])\n"
        "print({k: {f: sorted(v) for f, v in d.items()} for k, d in out.items()})\n"
    )
    gate_python = shutil.which("python3", path="/opt/homebrew/bin:/usr/local/bin:/usr/bin")
    assert gate_python is not None
    proc = subprocess.run([gate_python, "-I", "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "{'tests/test_t.py::test_x': {'tool.py': [1, 2, 8, 15]}}"
