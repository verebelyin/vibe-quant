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
_QUERIES = ("widget.py", "gizmo.py", "gamma", "sprocket")
_ALL_IDS = {
    *(f"p{i:02d}" for i in range(1, 26)),
    *(f"a{i:02d}" for i in range(1, 9)),
    *(f"b{i:02d}" for i in range(1, 5)),
}
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
bs: ModuleType = importlib.import_module("board_store")  # the store board_io writes through (same module)


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
def fresh(board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
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


def _rows() -> dict[str, Any]:
    return {r.id: r for r in bio.load_rows(_PERSISTENT, _JOBS)}


class TestMapping:
    def test_root_channels_by_topic_and_job(self) -> None:
        rows = _rows()
        got = {i: bio.root_channel(rows[i]) for i in ("p01", "p04", "p08", "p10", "p12", "p13", "p14", "p15", "p17", "a04", "b03")}
        assert got == {
            "p01": "kb/gotchas", "p04": "kb/findings", "p08": "kb/decisions", "p10": "needs-user",
            "p12": "lobby", "p13": "chat", "p14": "kb/thoughts", "p15": "kb/model-notes",
            "p17": "kb/self-improvement", "a04": "job/20260101-a", "b03": "job/20260102-b",
        }
        assert bio.root_channel(bio.Row({"id": "x"}, "20261120-Board.Pin")) == "job/20261120-board-pin"

    def test_plan_labels_kind_where_the_reader_derives_another(self) -> None:
        rows = _rows()
        question = {"id": "B10", "channel": "needs-user", "labels": {"kind": "question"}}
        p11 = bio.plan(rows["p11"], {"p10": question})
        assert (p11.re, p11.channel, p11.profile, p11.key) == ("B10", None, "user", "legacy-p11")  # no namespace
        assert p11.labels == {  # kind answer: the reader derives it under a question
            "legacy-id": "p11", "ts": "2026-10-04T09:00:00", "user-answer": "true", "relayed-by": "chief",
        }
        assert bio.plan(rows["p20"], {"p10": question}).labels == {
            "legacy-id": "p20", "ts": "2026-10-04T08:30:00", "kind": "info",  # an info reply to a question
        }
        p21 = bio.plan(rows["p21"], {})
        assert (p21.channel, p21.labels) == ("lobby", {
            "legacy-id": "p21", "ts": "2026-10-04T10:30:00", "to": "chief", "kind": "info",  # `[done]` prefix
        })
        assert bio.plan(rows["a01"], {}).labels == {
            "legacy-id": "a01", "ts": "2026-10-06T07:00:00", "job": "20260101-a", "to": "chief", "kind": "question",
        }

    def test_plan_authors_and_orphan_ref(self) -> None:
        rows = _rows()
        got = {i: (bio.plan(rows[i], {}).profile, bio.plan(rows[i], {}).labels.get("from")) for i in ("p01", "p15", "a07", "a08", "p02")}
        assert got == {
            "p01": ("arch-l2", "arch-L2"), "p15": ("arch-l2-", "Arch L2!"),
            "a07": ("drev-x0axf-14-1", "drev-x0axf.14-1"), "a08": ("a9lives", "9lives"), "p02": ("bob", None),
        }
        a05 = bio.plan(rows["a05"], {})
        assert (a05.channel, a05.re, a05.labels["ref"]) == ("job/20260101-a", None, "zz-missing")

    def test_import_order_parent_first_skips_mirrors_and_underscore_dirs(self) -> None:
        order, skipped = bio.import_order(bio.load_rows(_PERSISTENT, _JOBS))
        ids = [r.id for r in order]
        assert len(ids) == len(set(ids)) == 37 and set(ids) == _ALL_IDS
        assert ids.index("b03") < ids.index("b02")  # the parent comes first
        assert [(r.source, r.id) for r in skipped] == [("20260101-a", "p16"), ("20260101-a", "p23")]
        assert "g01" not in ids

    def test_limit_jobs_reads_the_first_jobs_only(self) -> None:
        assert {r.source for r in bio.load_rows(_PERSISTENT, _JOBS, limit_jobs=1)} == {"persistent", "20260101-a"}

    def test_text_never_empty_or_blank(self) -> None:
        assert [bio.post_text(b) for b in ("", "\n", " \t ", None, "a\n")] == ["(empty)"] * 4 + ["a\n"]

    def test_a_secret_text_or_label_would_be_refused(self) -> None:
        def row(**kw: object) -> Any:
            return bio.Row({"id": "s1", "ts": "t", "from": "w", "to": "board", "kind": "info", "body": "fine", "topic": "chat", **kw}, "persistent")

        assert [bio.would_be_refused(r) for r in (row(), row(body="API_TOKEN=abcdefghijklmnop"), row(by="API_TOKEN=abcdefghijklmnop"))] == [False, True, True]


# ------------------------------------------------------------------ import against a real board


class TestImport:
    def test_channel_of_each_row(self, board: Imported) -> None:
        ns = board.ns + "/"
        got = {old: str(board.msg(old)["channel"]).removeprefix(ns) for old in sorted(_ALL_IDS)}
        kb, a, b = "kb/", "job/20260101-a", "job/20260102-b"
        assert got == {
            "p01": kb + "gotchas", "p02": kb + "gotchas", "p03": kb + "gotchas",
            "p04": kb + "findings", "p05": kb + "findings", "p06": kb + "findings",
            "p07": kb + "decisions", "p08": kb + "decisions", "p09": kb + "decisions",
            "p10": "needs-user", "p11": "needs-user", "p12": "lobby", "p13": "chat",
            "p14": kb + "thoughts", "p15": kb + "model-notes", "p16": kb + "gotchas",
            "p17": kb + "self-improvement", "p18": kb + "self-improvement", "p19": kb + "self-improvement",
            "p20": "needs-user", "p21": "lobby", "p22": "chat", "p23": kb + "gotchas", "p24": kb + "gotchas",
            "p25": "chat",
            "a01": a, "a02": a, "a03": a, "a04": a, "a05": a, "a06": a, "a07": a, "a08": a,
            "b01": b, "b02": b, "b03": b, "b04": b,
        }

    def test_thread_shape_reply_to_exact_parent(self, board: Imported) -> None:
        ids = board.ids
        shape = {old: (board.msg(old)["re"], board.msg(old)["replyTo"]) for old in ("p01", "p02", "p03", "p19", "b02", "a05")}
        assert shape == {
            "p01": (None, None),
            "p02": (ids["p01"], None),  # the board leaves replyTo null at the root
            "p03": (ids["p01"], ids["p02"]),
            "p19": (ids["p17"], ids["p18"]),
            "b02": (ids["b03"], None),
            "a05": (None, None),  # orphan: a root with a ref label
        }

    def test_labels_on_each_message(self, board: Imported) -> None:
        ts = {r.id: r.rec["ts"] for r in bio.load_rows(_PERSISTENT, _JOBS)}
        want = {
            "p01": {"job": "20260101-a", "importance": "3", "state": "live", "from": "arch-L2"},
            "p02": {},
            "p10": {"job": "20260101-a", "kind": "question"},
            "p11": {"user-answer": "true", "relayed-by": "chief"},
            "p12": {"kind": "done", "to": "chief"},
            "p13": {},
            "p15": {"from": "Arch L2!", "state": "live"},
            "p16": {"job": "20260101-a", "state": "live"},
            "p20": {"kind": "info"},
            "p21": {"kind": "info", "to": "chief"},
            "a01": {"job": "20260101-a", "kind": "question", "to": "chief"},
            "a02": {"job": "20260101-a", "to": "worker-a"},
            "a03": {"job": "20260101-a", "to": "all"},
            "a05": {"job": "20260101-a", "to": "chief", "ref": "zz-missing"},
            "a07": {"job": "20260101-a", "importance": "1", "to": "chief", "from": "drev-x0axf.14-1"},
            "a08": {"job": "20260101-a", "to": "chief", "from": "9lives"},
            "b01": {"job": "20260102-b", "to": "chief"},
            "b04": {"job": "20260102-b", "kind": "done", "to": "chief"},
        }
        got = {old: board.labels(old) for old in want}
        assert got == {old: {"legacy-id": old, "ts": ts[old], **extra} for old, extra in want.items()}
        assert board.labels("a06") == {
            "legacy-id": "a06", "ts": ts["a06"], "job": "20260101-a", "topic": "design", "retires": board.ids["a04"],
        }

    def test_retire_and_supersede_states(self, board: Imported) -> None:
        ids = board.ids
        assert board.labels("p04") == {
            "legacy-id": "p04", "ts": "2026-10-02T09:00:00", "state": "retired", "reason": "wrong numbers see beta v2",
        }
        assert board.labels("p06") == {"legacy-id": "p06", "ts": "2026-10-02T10:00:00", "retires": ids["p04"]}
        assert (board.msg("p06")["re"], board.msg("p06")["replyTo"]) == (ids["p04"], None)
        assert board.labels("p18")["reason"] == "gizmo.py claim was wrong"
        a04 = board.labels("a04")
        assert (a04["state"], a04["reason"], a04["topic"]) == ("retired", "design dropped", "design")
        # the first supersede wins; the second leaves the target alone
        assert board.labels("p07") == {
            "legacy-id": "p07", "ts": "2026-10-03T11:00:00", "state": "superseded", "superseded-by": ids["p08"],
        }
        assert board.labels("p08") == {
            "legacy-id": "p08", "ts": "2026-10-03T12:00:00", "importance": "2", "supersedes": ids["p07"], "state": "live",
        }
        assert board.labels("p09") == {"legacy-id": "p09", "ts": "2026-10-03T13:00:00", "supersedes": ids["p07"], "state": "live"}

    def test_mirrors_exist_once(self, board: Imported) -> None:
        legacy = [bio._labels(m).get("legacy-id") for m in bio.board_raw().values()]
        assert (legacy.count("p16"), legacy.count("p23"), len(legacy)) == (1, 1, 37)

    def test_texts_are_exact(self, board: Imported) -> None:
        got = {old: board.msg(old)["text"] for old in ("p13", "p14", "p22", "p01")}
        assert got == {
            "p13": "--flag looks like an option but is text", "p14": "(empty)", "p22": "(empty)",
            "p01": "RULE: fixture alpha reads widget.py before gizmo setup",
        }

    def test_authors(self, board: Imported) -> None:
        got = {old: board.msg(old)["by"]["profile"] for old in ("p15", "a07", "a08", "p01", "p11")}
        assert got == {"p15": "arch-l2-", "a07": "drev-x0axf-14-1", "a08": "a9lives", "p01": "arch-l2", "p11": "user"}

    def test_kind_reads_back_as_written(self, board: Imported) -> None:
        view = {m["id"]: m for m in bs.read(None)}
        assert {i: view[i]["kind"] for i in ("p20", "p21", "p11", "p10", "p12")} == {
            "p20": "info", "p21": "info", "p11": "answer", "p10": "question", "p12": "done",
        }

    def test_map_has_one_entry_per_distinct_id(self, board: Imported) -> None:
        assert set(board.ids) == _ALL_IDS and len(set(board.ids.values())) == 37

    def test_reports(self, board: Imported) -> None:
        first, second = board.first, board.second
        assert (first["rows_read"], first["posted"], first["mirror_skipped"], first["board_added"]) == (35, 33, 2, 33)
        assert first["sanitised_authors"] == {"count": 3, "items": [
            {"id": "p15", "from": "Arch L2!", "as": "arch-l2-"},
            {"id": "a07", "from": "drev-x0axf.14-1", "as": "drev-x0axf-14-1"},
            {"id": "a08", "from": "9lives", "as": "a9lives"},
        ]}
        assert first["orphan_refs"] == {"count": 1, "items": [{"id": "a05", "ref": "zz-missing"}]}
        assert (first["rejects"], first["secret_refused"], first["supersede_first_wins"]) == (
            {"count": 0, "items": []}, {"count": 0, "ids": []}, 1)
        assert (second["rows_read"], second["already_mapped"], second["posted"], second["board_added"]) == (39, 33, 4, 4)
        assert (second["map_entries"], second["distinct_ids"], second["remapped"]) == (37, 37, 0)

    def test_second_run_adds_nothing(self, board: Imported) -> None:
        before = _count()
        code, report = _import(board.map_file)
        assert code == 0, report["rejects"]
        assert (report["posted"], report["remapped"], report["already_mapped"], report["board_added"]) == (0, 0, 37, 0)
        assert _count() == before
        assert json.loads(board.map_file.read_text()) == board.ids

    def test_resume_from_half_a_map_reposts_under_the_same_key_and_adds_nothing(self, board: Imported, tmp_path: Path) -> None:
        map_file = tmp_path / "half.json"
        map_file.write_text(json.dumps(dict(sorted(board.ids.items())[:18])))
        before = _count()
        code, report = _import(map_file)
        assert code == 0, report["rejects"]
        assert (report["posted"], report["remapped"], report["board_added"]) == (0, 19, 0)
        assert _count() == before
        assert json.loads(map_file.read_text()) == board.ids


# ------------------------------------------------------------------ verify


class TestVerify:
    def test_passes_on_the_imported_fixture(self, board: Imported) -> None:
        code, report = _verify(board.map_file, *_QUERIES)
        assert code == 0, report["mismatches"]
        assert report["counts"] == {
            "source_rows": 39, "distinct_ids": 37, "expected_ids": 37, "secret_refused": 0,
            "map_entries": 37, "board_legacy_messages": 37, "readback_rows": 39, "mirror_hidden": 1,
        }
        hidden = ["p04", "p05", "p06", "p07", "p18", "p19", "p23", "p24"]
        assert report["hidden"] == {
            "persistent": {"old": hidden, "new": hidden},
            "20260101-a": {"old": ["a04", "a06", "p23"], "new": ["a04", "a06", "p23"]},
            "20260102-b": {"old": [], "new": []},
        }
        assert report["mirror_hidden_ids"] == ["p23"]  # MIRROR_HIDDEN
        search = report["search"]
        assert {w: (s["old"], s["new"], s["top_level"]) for w, s in search.items()} == {
            "widget.py": (1, 1, []), "gizmo.py": (0, 1, []), "gamma": (2, 2, []), "sprocket": (0, 0, ["p25"]),
        }
        extra = search["gizmo.py"]["extras"]
        assert [e["root"] for e in extra] == ["p17"]
        assert {m["id"]: m["why"] for m in extra[0]["matched"]} == {"p18": "hidden: retired", "p19": "hidden: retire event"}

    def test_fails_on_a_missing_map_entry_and_a_wrong_one(self, board: Imported, tmp_path: Path) -> None:
        map_file = tmp_path / "less.json"
        map_file.write_text(json.dumps({**{k: v for k, v in board.ids.items() if k != "a02"}, "p01": board.ids["p02"]}))
        code, report = _verify(map_file)
        assert code == 1
        assert [m for m in report["mismatches"] if m["check"].startswith("map-")] == [
            {"check": "map-missing", "ids": ["a02"]}, {"check": "map-wrong", "ids": ["p01"]},
        ]

    def test_names_each_damaged_field_of_the_read_back(self, fresh: Path, board_server: BoardServer) -> None:
        """The reviewer's damage probe: a dropped kind, a fake ts, a wrong `to` and `job`."""
        assert _import(fresh)[0] == 0
        ids = json.loads(fresh.read_text())
        assert board_server.vb("label", ids["a01"], "--unset", "kind").returncode == 0
        assert board_server.vb("label", ids["a02"], "ts=1999-01-01T00:00:00").returncode == 0
        assert board_server.vb("label", ids["a03"], "to=nobody", "job=wrong").returncode == 0
        code, report = _verify(fresh)
        assert code == 1
        assert report["mismatches"] == [
            {"check": "readback", "view": "20260101-a", "id": "a01", "field": "kind", "old": "question", "new": "info"},
            {"check": "readback", "view": "20260101-a", "id": "a02", "field": "ts", "old": "2026-10-06T07:05:00", "new": "1999-01-01T00:00:00"},
            {"check": "readback", "view": "20260101-a", "id": "a02", "field": "kind", "old": "answer", "new": "info"},  # its parent lost its kind
            {"check": "readback", "view": "20260101-a", "id": "a03", "field": "to", "old": "all", "new": "nobody"},
            {"check": "readback", "view": "20260101-a", "id": "a03", "field": "job", "old": "20260101-a", "new": "wrong"},
        ]

    def test_fails_on_a_duplicate_on_the_board(self, fresh: Path) -> None:
        assert _import(fresh)[0] == 0
        bs.post_raw(f"{bs.ns()}lobby", "a second copy", labels={"legacy-id": "p02"}, profile="w")
        code, report = _verify(fresh)
        assert code == 1 and {"check": "board-duplicate", "ids": ["p02"]} in report["mismatches"]

    def test_fails_on_flipped_hidden_states_in_every_source(self, fresh: Path, board_server: BoardServer) -> None:
        assert _import(fresh)[0] == 0
        ids = json.loads(fresh.read_text())
        assert board_server.vb("label", ids["p07"], "state=live").returncode == 0  # persistent
        assert board_server.vb("label", ids["a04"], "--unset", "state").returncode == 0  # job a
        assert board_server.vb("label", ids["p23"], "state=live").returncode == 0  # a mirror the board shows again
        code, report = _verify(fresh)
        assert code == 1
        assert [m for m in report["mismatches"] if m["check"] == "hidden-parity"] == [
            {"check": "hidden-parity", "source": "persistent", "only_old": ["p07", "p23"], "only_new": []},
            {"check": "hidden-parity", "source": "20260101-a", "only_old": ["a04", "p23"], "only_new": []},
        ]

    def test_resumed_import_reapplies_a_retire(self, fresh: Path, board_server: BoardServer) -> None:
        assert _import(fresh)[0] == 0
        p18 = json.loads(fresh.read_text())["p18"]
        assert board_server.vb("label", p18, "--unset", "state", "--unset", "reason").returncode == 0
        assert _verify(fresh)[0] == 1
        code, report = _import(fresh)
        assert (code, report["posted"], report["board_added"]) == (0, 0, 0)
        assert {k: v for k, v in _msg(p18)["labels"].items() if k in ("state", "reason")} == {
            "state": "retired", "reason": "gizmo.py claim was wrong",
        }
        assert _verify(fresh)[0] == 0

    def test_a_key_that_answers_another_text_is_a_reject_and_a_body_mismatch(self, fresh: Path) -> None:
        """Same key, same channel and profile: the board answers the first post (I5)."""
        bs.post_raw(f"{bs.ns()}job/20260101-a", "another text", labels={"legacy-id": "a03"}, profile="chief", key=bio.idem_key("a03"))
        before = _count()
        code, report = _import(fresh)
        assert code == 1
        assert [(r["id"], r["step"]) for r in report["rejects"]["items"]] == [("a03", "key-reused")]
        assert _count() - before == 36  # every row but a03
        code, check = _verify(fresh)
        body = [m for m in check["mismatches"] if m.get("field") == "body"]
        assert code == 1 and [(m["id"], m["old"]["len"], m["new"]["len"]) for m in body] == [("a03", 24, 12)]
        assert "another text" not in json.dumps(check)  # a body is never printed

    def test_a_key_held_by_another_author_is_a_reject_never_a_second_post(self, fresh: Path) -> None:
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
        src = _rows_file(tmp_path, [
            {"id": "s1", "ts": "2026-10-01T08:00:00", "from": "w", "to": "board", "kind": "info", "body": "API_TOKEN=abcdefghijklmnop", "topic": "chat"},
            {"id": "s2", "ts": "2026-10-01T08:01:00", "from": "w", "to": "board", "kind": "info", "body": "fine", "topic": "chat", "ref": "s1"},
        ])
        code, report = _run("import-legacy", *src, "--map", str(fresh))
        assert (code, report["secret_refused"], report["posted"]) == (0, {"count": 1, "ids": ["s1"]}, 1)
        assert report["orphan_refs"] == {"count": 1, "items": [{"id": "s2", "ref": "s1"}]}
        code, check = _verify(fresh, src=src)
        assert (code, check["secret_refused_ids"], check["mismatches"]) == (0, ["s1"], [])

    def test_a_search_hit_outside_kb_and_top_level_fails(self, fresh: Path, tmp_path: Path) -> None:
        src = _rows_file(tmp_path, [
            {"id": "l1", "ts": "2026-10-01T08:00:00", "from": "w", "to": "board", "kind": "info", "body": "lobby note on sprocket", "topic": "design"},
            {"id": "c1", "ts": "2026-10-01T08:01:00", "from": "w", "to": "board", "kind": "info", "body": "chat on sprocket", "topic": "chat"},
        ])
        assert _run("import-legacy", *src, "--map", str(fresh))[0] == 0
        code, check = _verify(fresh, "sprocket", src=src)
        assert code == 1 and check["mismatches"] == [{"check": "search-outside-kb", "word": "sprocket", "ids": ["l1"]}]
        assert check["search"]["sprocket"]["top_level"] == ["c1"]


def test_cli_needs_a_board_and_writes_its_report_atomically(board_server: BoardServer, tmp_path: Path) -> None:
    src = _rows_file(tmp_path, [{"id": "x1", "ts": "2026-10-01T08:00:00", "from": "w", "to": "board", "kind": "info", "body": "x", "topic": "chat"}])
    env = {**board_server.env, "SWARM_BOARD_NS": "bio" + uuid.uuid4().hex[:10]}
    url = env.pop("BOARD_URL")
    cmd = [sys.executable, str(_SCRIPT), "import-legacy", *src, "--map", str(tmp_path / "m.json")]
    none = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert (none.returncode, "no board" in none.stderr) == (2, True)
    report = tmp_path / "import.json"
    report.write_text("an older run's report")
    got = subprocess.run([*cmd, "--url", url, "--report", str(report)], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert got.returncode == 0, got.stderr
    assert json.loads(report.read_text()) == json.loads(got.stdout)
    assert json.loads(got.stdout)["posted"] == 1 and sorted(p.name for p in tmp_path.iterdir()) == ["board", "import.json", "jobs", "m.json"]


def test_a_job_reply_into_a_global_thread_is_named_by_verify(fresh: Path, tmp_path: Path) -> None:
    """Known board_store gap (0 real rows): a reply reads its topic from the thread's channel,
    so a topic-less job row under a chat post reads back topic=chat. verify must say so."""
    src = _rows_file(tmp_path, [
        {"id": "c1", "ts": "2026-10-01T08:00:00", "from": "w", "to": "board", "kind": "info", "body": "chat root", "topic": "chat"},
    ])
    job = tmp_path / "jobs" / "20261001-j" / "bus"
    job.mkdir(parents=True)
    (job / "messages.jsonl").write_text(json.dumps(
        {"id": "j1", "ts": "2026-10-01T08:01:00", "from": "w", "to": "chief", "kind": "info", "body": "job reply", "ref": "c1"}) + "\n")
    assert _run("import-legacy", *src, "--map", str(fresh))[0] == 0
    code, check = _verify(fresh, src=src)
    assert code == 1 and check["mismatches"] == [
        {"check": "readback", "view": "20261001-j", "id": "j1", "field": "topic", "old": None, "new": "chat"},
    ]
