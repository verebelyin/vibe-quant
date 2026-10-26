"""Single-writer scan for run/job status writes (rabmo.12 / T18, R50/R51).

Every write to ``backtest_runs.status`` and to ``background_jobs`` lives in
``vibe_quant/db/run_lifecycle.py``. This scan reads the package source rather
than a live database, so a new writer anywhere else fails here at unit speed.
"""

from __future__ import annotations

from pathlib import Path

import vibe_quant

LIFECYCLE_PATH = "db/run_lifecycle.py"

_RUN_UPDATE = "UPDATE backtest_runs SET"
_RUN_PARAMETERS_UPDATE = "UPDATE backtest_runs SET parameters"
_JOB_WRITES = (
    "UPDATE background_jobs",
    "INSERT INTO background_jobs",
    "DELETE FROM background_jobs",
)

# Writes the scan keeps tolerating, plus a marker each allowlisted file must
# still contain so a stale allowlist entry fails loudly.
# ``INSERT INTO backtest_runs`` is allowed anywhere and is therefore not a
# flagged pattern at all.
ALLOWED_WRITES: dict[str, tuple[str, ...]] = {
    "db/schema.py": ("UPDATE backtest_runs SET", "UPDATE background_jobs"),
    "screening/__main__.py": (_RUN_PARAMETERS_UPDATE,),
    "db/state_manager.py": (_RUN_PARAMETERS_UPDATE,),
}


def offenders(sources: dict[str, str]) -> list[str]:
    """Flag run/job status writes outside the lifecycle module.

    ``sources`` maps a package-relative path (``"jobs/manager.py"``) to its
    source text. Whitespace collapses first, so a multi-line or dynamic
    (f-string) statement matches the same patterns as a single-line one.
    """
    found: list[str] = []
    for path, text in sources.items():
        if path == LIFECYCLE_PATH:
            continue
        collapsed = " ".join(text.split())
        for allowed in ALLOWED_WRITES.get(path, ()):
            collapsed = collapsed.replace(allowed, "")
        if _RUN_UPDATE in collapsed:
            found.append(f"{path}: {_RUN_UPDATE}")
        for write in _JOB_WRITES:
            if write in collapsed:
                found.append(f"{path}: {write}")
    return sorted(found)


def _package_sources() -> dict[str, str]:
    root = Path(vibe_quant.__file__).parent
    return {path.relative_to(root).as_posix(): path.read_text() for path in root.rglob("*.py")}


def test_no_status_writes_outside_lifecycle() -> None:
    """R50: only db/run_lifecycle.py writes run status or background_jobs rows."""
    assert offenders(_package_sources()) == []


def test_multiline_run_status_update_is_flagged() -> None:
    """A ``UPDATE backtest_runs`` split across lines is still a status write."""
    source = 'conn.execute(\n    """UPDATE backtest_runs\n       SET status = ?\n    """,\n)'
    assert offenders({"jobs/other.py": source}) == [f"jobs/other.py: {_RUN_UPDATE}"]


def test_dynamic_run_update_is_flagged() -> None:
    """The f-string SET form of update_backtest_run_status is flagged too."""
    source = 'f"UPDATE backtest_runs SET {\', \'.join(updates)} WHERE id = ?{guard}"'
    assert offenders({"db/state_manager.py": source}) == [f"db/state_manager.py: {_RUN_UPDATE}"]


def test_job_row_writes_are_flagged() -> None:
    """Every background_jobs write shape outside the lifecycle module is flagged."""
    sources = {
        "a.py": "UPDATE background_jobs SET pid = ?",
        "b.py": "INSERT INTO background_jobs (run_id, pid) VALUES (?, ?)",
        "c.py": "DELETE FROM background_jobs WHERE run_id = ?",
    }
    expected = [
        "a.py: UPDATE background_jobs",
        "b.py: INSERT INTO background_jobs",
        "c.py: DELETE FROM background_jobs",
    ]
    assert offenders(sources) == expected


def test_parameters_write_is_allowed_only_in_its_own_files() -> None:
    """``UPDATE backtest_runs SET parameters`` passes in the two allowlisted files only."""
    source = 'conn.execute("UPDATE backtest_runs SET parameters = ? WHERE id = ?", (raw, run_id))'
    assert offenders({"db/state_manager.py": source}) == []
    assert offenders({"screening/__main__.py": source}) == []
    assert offenders({"jobs/manager.py": source}) == [f"jobs/manager.py: {_RUN_UPDATE}"]


def test_insert_into_backtest_runs_is_allowed_anywhere() -> None:
    """create_backtest_run's INSERT is not a status write, wherever it lives."""
    source = 'conn.execute("INSERT INTO backtest_runs (run_mode) VALUES (?)", ("screening",))'
    assert offenders({"db/state_manager.py": source}) == []


def test_lifecycle_module_is_exempt() -> None:
    """The lifecycle module itself is the single writer and is not scanned."""
    source = "UPDATE backtest_runs SET status = ?; INSERT INTO background_jobs (run_id) VALUES (?)"
    assert offenders({LIFECYCLE_PATH: source}) == []


def test_allowlist_entries_are_live() -> None:
    """Every allowlisted write still exists in its file (no stale allowlist)."""
    sources = _package_sources()
    for path, markers in ALLOWED_WRITES.items():
        assert path in sources, path
        collapsed = " ".join(sources[path].split())
        for marker in markers:
            assert marker in collapsed, f"stale allowlist entry: {path}: {marker}"


def test_lifecycle_opens_no_connection() -> None:
    """R51: run_lifecycle takes the caller's connection and opens none."""
    assert "connect(" not in _package_sources()[LIFECYCLE_PATH]
