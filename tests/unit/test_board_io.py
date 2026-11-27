"""board_io.py: the one-off replay of the old swarm bus into the Vercel Board, and its check.

The fixture (tests/fixtures/board_legacy) holds one row of every shape the old JSONL carries:
plain post, reply, reply to a reply, retire, two supersedes of one post, a needs-user answer
relayed by the chief, an `info` reply to a question and an `info` row with a `[done]` prefix
(kind labelled where the reader would derive another), a DM pair, a broadcast, two mirrored
posts (one retired on the persistent board only), an orphan
ref, a `--` body, an empty and a whitespace-only body, a chat post a search word hits, and
three author names the board does not keep as they are. Every outcome is read back from the
session's throwaway `vb serve` (``board_server``) through board_store, each run in its own
channel namespace (``SWARM_BOARD_NS``).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from tests.unit.conftest import BoardServer

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "agents" / "board_io.py"
_FIX = _ROOT / "tests" / "fixtures" / "board_legacy"
_PERSISTENT = _FIX / "messages.jsonl"
_JOBS = _FIX / "jobs"
_SRC = ("--persistent", str(_PERSISTENT), "--jobs-root", str(_JOBS))
Report = dict[str, Any]  # a JSON report as board_io prints it


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("board_io", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["board_io"] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


bio = _load()
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))
bs: ModuleType = importlib.import_module(
    "board_store"
)  # the store board_io writes through (same module)


def _on(mp: pytest.MonkeyPatch, server: BoardServer, ns: str) -> None:
    """Point board_store (and `vb`) at the test board, in namespace ``ns``."""
    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        mp.setenv(key, server.env[key])
    mp.setenv("SWARM_BOARD_NS", ns)


def _run(*args: str) -> tuple[int, Report]:
    """``board_io.main`` in this process: (exit code, the JSON report it printed)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = bio.main(list(args))
    report = json.loads(out.getvalue())
    assert isinstance(report, dict)
    return code, report


def _import(map_file: Path, *extra: str) -> tuple[int, Report]:
    return _run("import-legacy", *_SRC, "--map", str(map_file), *extra)


def _verify(map_file: Path, *queries: str, src: tuple[str, ...] = _SRC) -> tuple[int, Report]:
    return _run("verify", *src, "--map", str(map_file), "--query", *queries)


def _msg(new: str) -> dict[str, Any]:
    """One board message by board id (read through its thread)."""
    for m in bs.read_thread(new):
        if m["id"] == new:
            return dict(m)
    raise AssertionError(f"{new} not in its own thread")


def _count() -> int:
    return len(bio.board_raw())


@dataclass
class Imported:
    server: BoardServer
    ns: str
    map_file: Path
    ids: dict[str, str]
    first: Report
    second: Report

    def msg(self, old: str) -> dict[str, Any]:
        return _msg(self.ids[old])

    def labels(self, old: str) -> dict[str, str]:
        labels = self.msg(old)["labels"]
        assert isinstance(labels, dict)
        return labels


@pytest.fixture(scope="module")
def imported(board_server: BoardServer, tmp_path_factory: pytest.TempPathFactory) -> Imported:
    """Persistent board + job a (a run that stopped there), then everything."""
    ns = "bio" + uuid.uuid4().hex[:10]
    map_file = tmp_path_factory.mktemp("board_io") / "legacy-ids.json"
    with pytest.MonkeyPatch.context() as mp:
        _on(mp, board_server, ns)
        code1, first = _import(map_file, "--limit-jobs", "1")
        code2, second = _import(map_file)
    assert (code1, code2) == (0, 0), (first["rejects"], second["rejects"])
    return Imported(board_server, ns, map_file, json.loads(map_file.read_text()), first, second)


@pytest.fixture
def board(imported: Imported, monkeypatch: pytest.MonkeyPatch) -> Imported:
    """The imported fixture board, its namespace in the environment."""
    _on(monkeypatch, imported.server, imported.ns)
    return imported


@pytest.fixture
def fresh(
    board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Path]:
    """An empty namespace of its own (a test that damages the board); yields a map path."""
    _on(monkeypatch, board_server, "bio" + uuid.uuid4().hex[:10])
    yield tmp_path / "map.json"


def _rows_file(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[str, ...]:
    """A one-file persistent source (and an empty jobs root) as CLI arguments."""
    (tmp_path / "board").mkdir()
    (tmp_path / "jobs").mkdir()
    path = tmp_path / "board" / "messages.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ("--persistent", str(path), "--jobs-root", str(tmp_path / "jobs"))


# ------------------------------------------------------------------ pure mapping (no board)


class TestMapping:
    def test_a_secret_text_or_label_would_be_refused(self) -> None:
        def row(**kw: object) -> Any:
            return bio.Row(
                {
                    "id": "s1",
                    "ts": "t",
                    "from": "w",
                    "to": "board",
                    "kind": "info",
                    "body": "fine",
                    "topic": "chat",
                    **kw,
                },
                "persistent",
            )

        assert [
            bio.would_be_refused(r)
            for r in (
                row(),
                row(body="API_TOKEN=abcdefghijklmnop"),
                row(by="API_TOKEN=abcdefghijklmnop"),
            )
        ] == [False, True, True]


# ------------------------------------------------------------------ import against a real board


class TestImport:
    def test_second_run_adds_nothing(self, board: Imported) -> None:
        before = _count()
        code, report = _import(board.map_file)
        assert code == 0, report["rejects"]
        assert (
            report["posted"],
            report["remapped"],
            report["already_mapped"],
            report["board_added"],
        ) == (0, 0, 37, 0)
        assert _count() == before
        assert json.loads(board.map_file.read_text()) == board.ids


# ------------------------------------------------------------------ verify


class TestVerify:
    def test_fails_on_a_missing_map_entry_and_a_wrong_one(
        self, board: Imported, tmp_path: Path
    ) -> None:
        map_file = tmp_path / "less.json"
        map_file.write_text(
            json.dumps(
                {**{k: v for k, v in board.ids.items() if k != "a02"}, "p01": board.ids["p02"]}
            )
        )
        code, report = _verify(map_file)
        assert code == 1
        assert [m for m in report["mismatches"] if m["check"].startswith("map-")] == [
            {"check": "map-missing", "ids": ["a02"]},
            {"check": "map-wrong", "ids": ["p01"]},
        ]

    def test_a_key_held_by_another_author_is_a_reject_never_a_second_post(
        self, fresh: Path
    ) -> None:
        """The key reused for another channel and profile: 409 duplicate_id, nobody's post (I5)."""
        bs.post_raw(f"{bs.ns()}lobby", "squatter", profile="someone", key=bio.idem_key("a03"))
        before = _count()
        code, report = _import(fresh)
        assert code == 1
        assert [(r["id"], r["step"]) for r in report["rejects"]["items"]] == [("a03", "post")]
        assert "used by someone else" in report["rejects"]["items"][0]["error"]
        assert (_count() - before, "a03" in json.loads(fresh.read_text())) == (36, False)
        code, check = _verify(fresh)
        assert code == 1
        assert {"check": "map-missing", "ids": ["a03"]} in check["mismatches"]
        assert {"check": "board-missing", "ids": ["a03"]} in check["mismatches"]

    def test_a_secret_row_is_listed_and_expected_missing(self, fresh: Path, tmp_path: Path) -> None:
        src = _rows_file(
            tmp_path,
            [
                {
                    "id": "s1",
                    "ts": "2026-10-01T08:00:00",
                    "from": "w",
                    "to": "board",
                    "kind": "info",
                    "body": "API_TOKEN=abcdefghijklmnop",
                    "topic": "chat",
                },
                {
                    "id": "s2",
                    "ts": "2026-10-01T08:01:00",
                    "from": "w",
                    "to": "board",
                    "kind": "info",
                    "body": "fine",
                    "topic": "chat",
                    "ref": "s1",
                },
            ],
        )
        code, report = _run("import-legacy", *src, "--map", str(fresh))
        assert (code, report["secret_refused"], report["posted"]) == (
            0,
            {"count": 1, "ids": ["s1"]},
            1,
        )
        assert report["orphan_refs"] == {"count": 1, "items": [{"id": "s2", "ref": "s1"}]}
        code, check = _verify(fresh, src=src)
        assert (code, check["secret_refused_ids"], check["mismatches"]) == (0, ["s1"], [])


def test_cli_needs_a_board_and_writes_its_report_atomically(
    board_server: BoardServer, tmp_path: Path
) -> None:
    src = _rows_file(
        tmp_path,
        [
            {
                "id": "x1",
                "ts": "2026-10-01T08:00:00",
                "from": "w",
                "to": "board",
                "kind": "info",
                "body": "x",
                "topic": "chat",
            }
        ],
    )
    env = {**board_server.env, "SWARM_BOARD_NS": "bio" + uuid.uuid4().hex[:10]}
    url = env.pop("BOARD_URL")
    cmd = [sys.executable, str(_SCRIPT), "import-legacy", *src, "--map", str(tmp_path / "m.json")]
    none = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert (none.returncode, "no board" in none.stderr) == (2, True)
    report = tmp_path / "import.json"
    report.write_text("an older run's report")
    got = subprocess.run(
        [*cmd, "--url", url, "--report", str(report)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert got.returncode == 0, got.stderr
    assert json.loads(report.read_text()) == json.loads(got.stdout)
    assert json.loads(got.stdout)["posted"] == 1 and sorted(p.name for p in tmp_path.iterdir()) == [
        "board",
        "import.json",
        "jobs",
        "m.json",
    ]
