"""Single-writer scan for run/job status writes (rabmo.12 / T18, R50/R51).

Every write to ``backtest_runs.status`` and to ``background_jobs`` lives in
``vibe_quant/db/run_lifecycle.py``. This scan reads the package source rather
than a live database, so a new writer anywhere else fails here at unit speed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import vibe_quant

LIFECYCLE_PATH = "db/run_lifecycle.py"

_RUN_UPDATE = "UPDATE backtest_runs SET"
_JOB_WRITES = (
    "UPDATE background_jobs",
    "INSERT INTO background_jobs",
    "DELETE FROM background_jobs",
)

# Columns that carry the run lifecycle. A SET touching any of them is a status
# write even when it also sets ``parameters``.
_LIFECYCLE_COLUMNS = re.compile(
    r"\b(status|pid|started_at|heartbeat_at|completed_at|error_message)\b", re.IGNORECASE
)
# ``UPDATE backtest_runs SET parameters ...`` up to its WHERE (or closing quote).
_PARAMETERS_UPDATE = re.compile(r"UPDATE backtest_runs SET parameters\b(?P<rest>.*?)(?: WHERE\b|\"|$)")

# db/schema.py is exempt as a whole: migrations and repairs rewrite lifecycle rows.
# The markers keep the exemption honest: each must still be in the file.
SCHEMA_PATH = "db/schema.py"
SCHEMA_MARKERS = ("UPDATE backtest_runs SET", "UPDATE background_jobs")

# Files that may write ``backtest_runs.parameters`` (and nothing lifecycle).
# ``INSERT INTO backtest_runs`` is allowed anywhere and is therefore not a
# flagged pattern at all.
PARAMETERS_WRITERS: dict[str, str] = {
    "screening/__main__.py": "UPDATE backtest_runs SET parameters = ?",
    "db/state_manager.py": "UPDATE backtest_runs SET parameters = ?",
}


def _strip_parameters_updates(collapsed: str) -> str:
    """Drop ``SET parameters ...`` statements that touch no lifecycle column."""

    def keep_if_lifecycle(match: re.Match[str]) -> str:
        return match.group(0) if _LIFECYCLE_COLUMNS.search(match.group("rest")) else ""

    return _PARAMETERS_UPDATE.sub(keep_if_lifecycle, collapsed)


def offenders(sources: dict[str, str]) -> list[str]:
    """Flag run/job status writes outside the lifecycle module.

    ``sources`` maps a package-relative path (``"jobs/manager.py"``) to its
    source text. Whitespace collapses first, so a multi-line or dynamic
    (f-string) statement matches the same patterns as a single-line one.
    """
    found: list[str] = []
    for path, text in sources.items():
        if path in (LIFECYCLE_PATH, SCHEMA_PATH):
            continue
        collapsed = " ".join(text.split())
        if path in PARAMETERS_WRITERS:
            collapsed = _strip_parameters_updates(collapsed)
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


@pytest.mark.parametrize("path", sorted(PARAMETERS_WRITERS))
@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE backtest_runs SET parameters = ?, status = ? WHERE id = ?",
        "UPDATE backtest_runs SET parameters = ?, heartbeat_at = datetime('now') WHERE id = ?",
        "UPDATE backtest_runs SET parameters = ?,\n   error_message = NULL WHERE id = ?",
        "UPDATE backtest_runs SET parameters = ?, completed_at = NULL, pid = NULL",
        'f"UPDATE backtest_runs SET parameters = ?, {extra} status = \'failed\'"',
    ],
    ids=["status", "heartbeat", "error-multiline", "no-where", "fstring"],
)
def test_parameters_write_that_also_sets_a_lifecycle_column_is_flagged(
    path: str, statement: str
) -> None:
    """The parameters exemption covers SETs touching no lifecycle column."""
    assert offenders({path: f"conn.execute({statement!r})"}) == [f"{path}: {_RUN_UPDATE}"]


@pytest.mark.parametrize("path", sorted(PARAMETERS_WRITERS))
def test_parameters_write_next_to_a_status_write_is_still_flagged(path: str) -> None:
    """An exempt parameters UPDATE does not mask a second, real status UPDATE."""
    source = (
        'conn.execute("UPDATE backtest_runs SET parameters = ? WHERE id = ?", (raw, run_id))\n'
        'conn.execute("UPDATE backtest_runs SET status = ? WHERE id = ?", (st, run_id))'
    )
    assert offenders({path: source}) == [f"{path}: {_RUN_UPDATE}"]


def test_schema_module_is_exempt_as_a_whole() -> None:
    """db/schema.py migrations may write lifecycle rows of either table."""
    source = (
        "UPDATE backtest_runs SET status = 'failed' WHERE id = ?; "
        "INSERT INTO background_jobs (run_id) VALUES (?); "
        "DELETE FROM background_jobs WHERE run_id = ?"
    )
    assert offenders({SCHEMA_PATH: source}) == []
    assert offenders({"db/other.py": source}) == [
        "db/other.py: DELETE FROM background_jobs",
        "db/other.py: INSERT INTO background_jobs",
        f"db/other.py: {_RUN_UPDATE}",
    ]


def test_insert_into_backtest_runs_is_allowed_anywhere() -> None:
    """create_backtest_run's INSERT is not a status write, wherever it lives."""
    source = 'conn.execute("INSERT INTO backtest_runs (run_mode) VALUES (?)", ("screening",))'
    assert offenders({"db/state_manager.py": source}) == []


def test_lifecycle_module_is_exempt() -> None:
    """The lifecycle module itself is the single writer and is not scanned."""
    source = "UPDATE backtest_runs SET status = ?; INSERT INTO background_jobs (run_id) VALUES (?)"
    assert offenders({LIFECYCLE_PATH: source}) == []


def test_allowlist_entries_are_live() -> None:
    """Every exempted write still exists in its file (no stale allowlist)."""
    sources = _package_sources()
    markers = {SCHEMA_PATH: SCHEMA_MARKERS} | {
        path: (marker,) for path, marker in PARAMETERS_WRITERS.items()
    }
    for path, expected in markers.items():
        assert path in sources, path
        collapsed = " ".join(sources[path].split())
        for marker in expected:
            assert marker in collapsed, f"stale allowlist entry: {path}: {marker}"


def test_lifecycle_opens_no_connection() -> None:
    """R51: run_lifecycle takes the caller's connection and opens none."""
    assert "connect(" not in _package_sources()[LIFECYCLE_PATH]
