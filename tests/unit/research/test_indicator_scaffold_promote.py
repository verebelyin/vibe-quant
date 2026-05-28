"""Unit tests for bd-3p1k.3 — promote indicator (rename to drop prefix).

Covers the four pure helpers (``strip_auto_generated_header``,
``write_promoted_plugin``, ``bd_remember_indicator``,
``git_commit_promotion``) plus the ``promote_indicator`` orchestrator.
Subprocesses are monkeypatched except for the focused integration test
against a real ``git init`` repo, which covers the happy commit path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.research import indicator_scaffold
from vibe_quant.research.indicator_scaffold import (
    CO_AUTHOR_TRAILER,
    PromoteError,
    PromoteResult,
    bd_remember_indicator,
    git_commit_promotion,
    promote_indicator,
    promoted_path_for,
    proposed_path_for,
    strip_auto_generated_header,
    write_promoted_plugin,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------- strip_auto_generated_header ----------


def test_strip_header_removes_auto_generated_docstring() -> None:
    src = (
        '"""AUTO-GENERATED FROM EXTRACTION 42 ON 2026-01-01T00:00:00+00:00 — review.\n'
        "\n"
        "RANGES: period=llm\n"
        'Display: My Ind\n'
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
    )
    out = strip_auto_generated_header(src)
    assert "AUTO-GENERATED" not in out
    assert out.startswith("from __future__")


def test_strip_header_keeps_source_when_no_leading_docstring() -> None:
    src = "import pandas as pd\n# regular code\n"
    assert strip_auto_generated_header(src) == src


def test_strip_header_handles_leading_blank_lines() -> None:
    src = '\n\n"""AUTO-GENERATED header text"""\n\nimport pd\n'
    out = strip_auto_generated_header(src)
    assert "AUTO-GENERATED" not in out
    assert out.startswith("import pd")


# ---------- write_promoted_plugin ----------


def _seed_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    monkeypatch.setattr(indicator_scaffold, "PLUGINS_DIR", plugins)
    return plugins


def test_write_promoted_plugin_renames_strips_and_deletes_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _seed_plugins(tmp_path, monkeypatch)
    src = (
        '"""AUTO-GENERATED FROM EXTRACTION 1 ON x — review."""\n'
        "\n"
        "from __future__ import annotations\n"
        "MY = 1\n"
    )
    proposed = plugins / "proposed_my_ind.py"
    proposed.write_text(src)

    old, new = write_promoted_plugin(name="MY_IND")
    assert old == proposed
    assert new == plugins / "my_ind.py"
    assert not proposed.exists()
    assert new.exists()
    body = new.read_text()
    assert "AUTO-GENERATED" not in body
    assert "MY = 1" in body


def test_write_promoted_plugin_refuses_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _seed_plugins(tmp_path, monkeypatch)
    (plugins / "proposed_x.py").write_text('"""h"""\n')
    (plugins / "x.py").write_text("# already promoted\n")

    with pytest.raises(PromoteError) as excinfo:
        write_promoted_plugin(name="X")
    assert excinfo.value.code == "collision"
    # Old file must still exist — we didn't touch anything.
    assert (plugins / "proposed_x.py").exists()


def test_write_promoted_plugin_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plugins(tmp_path, monkeypatch)
    with pytest.raises(PromoteError) as excinfo:
        write_promoted_plugin(name="GHOST")
    assert excinfo.value.code == "not_found"


def test_write_promoted_plugin_rejects_invalid_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_plugins(tmp_path, monkeypatch)
    with pytest.raises(PromoteError) as excinfo:
        write_promoted_plugin(name="lowercase")
    assert excinfo.value.code == "invalid_name"


def test_write_promoted_plugin_force_overwrites_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _seed_plugins(tmp_path, monkeypatch)
    (plugins / "proposed_x.py").write_text('"""h"""\nNEW = 2\n')
    (plugins / "x.py").write_text("OLD = 1\n")

    _old, new = write_promoted_plugin(name="X", force=True)
    assert "NEW = 2" in new.read_text()
    assert not proposed_path_for("X").exists()


# ---------- bd_remember_indicator ----------


def test_bd_remember_missing_binary_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(indicator_scaffold.shutil, "which", lambda _b: None)
    ok, msg = bd_remember_indicator(
        name="X", extraction_id=1, source_url="http://x"
    )
    assert ok is False
    assert "not on PATH" in msg


def test_bd_remember_invokes_with_correct_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(*args, **kwargs):
        captured["argv"] = args[0]
        return FakeProc()

    monkeypatch.setattr(indicator_scaffold.shutil, "which", lambda _b: "/usr/bin/bd")
    monkeypatch.setattr(indicator_scaffold.subprocess, "run", fake_run)

    ok, _ = bd_remember_indicator(
        name="MY_IND", extraction_id=42, source_url="https://example/post"
    )
    assert ok is True
    argv = captured["argv"]
    assert argv[0] == "/usr/bin/bd"
    assert argv[1] == "remember"
    # fact embeds name, extraction, url
    assert "indicator:my_ind" in argv[2]
    assert "extraction 42" in argv[2]
    assert "https://example/post" in argv[2]
    assert argv[3:] == ["--key", "indicator:my_ind"]


def test_bd_remember_omits_extraction_and_url_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        captured["argv"] = args[0]
        return FakeProc()

    monkeypatch.setattr(indicator_scaffold.shutil, "which", lambda _b: "/usr/bin/bd")
    monkeypatch.setattr(indicator_scaffold.subprocess, "run", fake_run)

    bd_remember_indicator(name="ABC", extraction_id=None, source_url=None)
    fact = captured["argv"][2]
    assert "extraction" not in fact
    assert "source" not in fact


def test_bd_remember_returns_false_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "bd error: nope"

    monkeypatch.setattr(indicator_scaffold.shutil, "which", lambda _b: "/usr/bin/bd")
    monkeypatch.setattr(
        indicator_scaffold.subprocess, "run", lambda *_a, **_kw: FakeProc()
    )
    ok, msg = bd_remember_indicator(name="X", extraction_id=1, source_url=None)
    assert ok is False
    assert "nope" in msg


# ---------- git_commit_promotion (real git, integration) ----------


def _init_repo(tmp_path: Path) -> Path:
    import subprocess as sp

    sp.run(["git", "init", "-q", str(tmp_path)], check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "README").write_text("seed\n")
    sp.run(["git", "-C", str(tmp_path), "add", "README"], check=True)
    sp.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "seed"], check=True
    )
    return tmp_path


def test_git_commit_promotion_records_rename_with_expected_message(
    tmp_path: Path,
) -> None:
    import subprocess as sp

    repo = _init_repo(tmp_path)
    # Pre-commit a file at the proposed path so the deletion is real.
    proposed = repo / "proposed_my_ind.py"
    proposed.write_text("# initial\n")
    sp.run(["git", "-C", str(repo), "add", str(proposed)], check=True)
    sp.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "add proposed"],
        check=True,
    )

    # Simulate the rename: write the new file, delete the old.
    new = repo / "my_ind.py"
    new.write_text("# promoted\n")
    proposed.unlink()

    sha = git_commit_promotion(proposed, new, name="MY_IND", repo_root=repo)
    assert len(sha) == 40

    msg = sp.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert msg == "chore: promote indicator MY_IND (bd-3p1k)"

    body = sp.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%B"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert CO_AUTHOR_TRAILER in body

    # Both add + delete in the diff.
    stat = sp.run(
        ["git", "-C", str(repo), "show", "--stat", "--format=", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "my_ind.py" in stat
    assert "proposed_my_ind.py" in stat


def test_git_commit_promotion_reverts_staging_on_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    proposed = repo / "proposed_x.py"
    proposed.write_text("# x\n")
    new = repo / "x.py"
    new.write_text("# x\n")

    calls: list[list[str]] = []
    real_git = indicator_scaffold._git

    def fake_git(*args, cwd=None):
        calls.append(list(args))
        if args[0] == "commit":
            return 1, "", "hook rejected"
        return real_git(*args, cwd=cwd)

    monkeypatch.setattr(indicator_scaffold, "_git", fake_git)
    with pytest.raises(PromoteError) as excinfo:
        git_commit_promotion(proposed, new, name="X", repo_root=repo)
    assert excinfo.value.code == "commit_failed"
    assert any(a[0] == "restore" and "--staged" in a for a in calls)


# ---------- promote_indicator orchestrator ----------


def test_promote_indicator_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _seed_plugins(tmp_path, monkeypatch)
    (plugins / "proposed_demo.py").write_text(
        '"""AUTO-GENERATED FROM EXTRACTION 7 ON x — review."""\nFOO=1\n'
    )

    monkeypatch.setattr(
        indicator_scaffold,
        "git_commit_promotion",
        lambda _o, _n, *, name, repo_root=None: "c0ffee" * 6 + "abcd",  # 40 chars
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "bd_remember_indicator",
        lambda **_kw: (True, "saved"),
    )

    result = promote_indicator("DEMO", extraction_id=7, source_url="u")
    assert isinstance(result, PromoteResult)
    assert result.old_path == proposed_path_for("DEMO")
    assert result.new_path == promoted_path_for("DEMO")
    assert result.bd_remember_ok is True
    assert (plugins / "demo.py").exists()
    assert not (plugins / "proposed_demo.py").exists()
    assert "AUTO-GENERATED" not in (plugins / "demo.py").read_text()


def test_promote_indicator_rolls_back_on_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _seed_plugins(tmp_path, monkeypatch)
    (plugins / "proposed_demo.py").write_text('"""h"""\nFOO=1\n')

    def fail_commit(*_a, **_kw):
        raise PromoteError("commit_failed", "hook nope")

    monkeypatch.setattr(indicator_scaffold, "git_commit_promotion", fail_commit)

    with pytest.raises(PromoteError) as excinfo:
        promote_indicator("DEMO", extraction_id=None, source_url=None)
    assert excinfo.value.code == "commit_failed"
    # Rolled back: the new file is gone. (We do NOT recreate the old
    # file — the user can re-run scaffold from the extraction.)
    assert not (plugins / "demo.py").exists()


def test_promote_indicator_propagates_bd_failure_in_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _seed_plugins(tmp_path, monkeypatch)
    (plugins / "proposed_demo.py").write_text('"""h"""\n')
    monkeypatch.setattr(
        indicator_scaffold,
        "git_commit_promotion",
        lambda _o, _n, *, name, repo_root=None: "a" * 40,
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "bd_remember_indicator",
        lambda **_kw: (False, "bd not on PATH"),
    )
    result = promote_indicator("DEMO", extraction_id=None, source_url=None)
    assert result.bd_remember_ok is False
    assert "bd not on PATH" in result.bd_remember_output
