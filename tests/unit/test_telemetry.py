"""scripts/agents/telemetry.py — cross-job swarm task ledger (append-only JSONL)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_TELEMETRY_PATH = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "telemetry.py"

MODEL_A = "xiaomi/mimo-v2.6-pro"
MODEL_B = "deepseek/deepseek-v4.1-flash"

RESULT_SUCCESS = (
    '{"type":"result","subtype":"success","stopReason":"end_turn","durationMs":315493,'
    '"usage":{"inputTokens":508685,"outputTokens":7881},"finalText":"DONE cmd-report.md"}'
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swarm_telemetry", _TELEMETRY_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swarm_telemetry"] = mod
    spec.loader.exec_module(mod)
    return mod


tel = _load()


@pytest.fixture(autouse=True)
def _isolated_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point SWARM_TELEMETRY into tmp_path — a test must never write the real ledger."""
    ledger = tmp_path / "tasks.jsonl"
    monkeypatch.setenv("SWARM_TELEMETRY", str(ledger))
    return ledger


def _write_log(log: Path, text: str, encoding: str = "utf-8", age_s: float = 600) -> None:
    """Write a transcript and age its mtime past the 120 s "still running" guard."""
    log.write_text(text, encoding=encoding)
    old = time.time() - age_s
    os.utime(log, (old, old))


def _rec(**kw: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "ts": "2026-10-09T10:00:00",
        "job": "j1",
        "task": "t1",
        "agent": "w1",
        "runtime": "cmd",
        "model": MODEL_A,
        "round": 1,
        "outcome": None,
        "review": None,
        "tokens_in": None,
        "tokens_out": None,
        "wall_s": None,
        "bead": None,
        "note": None,
    }
    rec.update(kw)
    return rec


def _seed(ledger: Path, records: list[dict[str, Any]]) -> None:
    ledger.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def _ledger_rows(ledger: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


# hand-built report fixture: 2 models, 2 rounds, review updates, one superseded record
LEDGER = [
    _rec(
        ts="2026-10-09T10:00:00Z",
        task="t1",
        agent="w1",
        model=MODEL_A,
        round=1,
        outcome="done",
        review="PASS",
        tokens_in=100,
        tokens_out=10,
        wall_s=10.0,
    ),
    _rec(
        ts="2026-10-09T11:00:00",
        task="t2",
        agent="w1",
        model=MODEL_A,
        round=1,
        outcome="error",
        review="SHOULD-FIX",
        tokens_in=200,
        tokens_out=20,
        wall_s=20.0,
    ),
    _rec(
        ts="2026-10-09T12:00:00",
        task="t2",
        agent="w2",
        runtime="claude",
        model=MODEL_B,
        round=2,
        outcome="done",
        review="PASS",
        tokens_in=300,
        tokens_out=30,
        wall_s=30.0,
    ),
    _rec(
        ts="2026-10-08T09:00:00",
        task="t3",
        agent="w2",
        model=MODEL_B,
        round=1,
        outcome="done",
        review="BLOCK",
        tokens_in=400,
        tokens_out=40,
        wall_s=40.0,
    ),
    # review update: copy of the previous record with review set (supersedes it)
    _rec(
        ts="2026-10-09T13:00:00",
        task="t3",
        agent="w2",
        model=MODEL_B,
        round=1,
        outcome="done",
        review="PASS",
        tokens_in=400,
        tokens_out=40,
        wall_s=40.0,
    ),
    _rec(
        ts="2026-10-08T10:00:00",
        task="t4",
        agent="w1",
        model=MODEL_A,
        round=1,
        outcome="stalled",
        tokens_in=500,
        tokens_out=50,
        wall_s=50.0,
    ),
]

ROW_B = {
    "group": MODEL_B,
    "tasks": 2,
    "pass1_passes": 1,
    "pass1_attempts": 1,
    "pass1_rate": 1.0,
    "mean_rounds": 1.0,
    "med_tokens_in": 350.0,
    "med_wall_s": 35.0,
    "stalls_errors": 0,
}
ROW_A = {
    "group": MODEL_A,
    "tasks": 3,
    "pass1_passes": 1,
    "pass1_attempts": 2,
    "pass1_rate": 0.5,
    "mean_rounds": 1.5,
    "med_tokens_in": 200.0,
    "med_wall_s": 20.0,
    "stalls_errors": 2,
}


def test_record_writes_full_schema(_isolated_ledger: Path) -> None:
    rc = tel.main(
        [
            "record",
            "--job",
            "j1",
            "--task",
            "t1",
            "--agent",
            "w4",
            "--runtime",
            "cmd",
            "--model",
            MODEL_A,
            "--round",
            "2",
            "--outcome",
            "done",
            "--review",
            "PASS",
            "--tokens-in",
            "100",
            "--tokens-out",
            "5",
            "--wall-s",
            "12.5",
            "--bead",
            "vibe-quant-1",
            "--note",
            "hello",
        ]
    )
    assert rc == 0
    (rec,) = _ledger_rows(_isolated_ledger)
    assert rec == {
        "ts": rec["ts"],
        "job": "j1",
        "task": "t1",
        "agent": "w4",
        "runtime": "cmd",
        "model": MODEL_A,
        "round": 2,
        "outcome": "done",
        "review": "PASS",
        "tokens_in": 100,
        "tokens_out": 5,
        "wall_s": 12.5,
        "bead": "vibe-quant-1",
        "note": "hello",
        "source": None,
        "comms_asked": None,
        "comms_dms": None,
        "comms_replies": None,
        "board_relied": None,
    }
    assert list(rec) == [
        "ts",
        "job",
        "task",
        "agent",
        "runtime",
        "model",
        "round",
        "outcome",
        "review",
        "tokens_in",
        "tokens_out",
        "wall_s",
        "bead",
        "note",
        "source",
        "comms_asked",
        "comms_dms",
        "comms_replies",
        "board_relied",
    ]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rec["ts"])  # UTC ISO


def test_record_defaults_round1_and_nulls(_isolated_ledger: Path) -> None:
    tel.main(
        [
            "record",
            "--job",
            "j1",
            "--task",
            "t1",
            "--agent",
            "w4",
            "--runtime",
            "claude",
            "--model",
            MODEL_B,
        ]
    )
    (rec,) = _ledger_rows(_isolated_ledger)
    assert rec["round"] == 1
    for key in ("outcome", "review", "tokens_in", "tokens_out", "wall_s", "bead", "note"):
        assert rec[key] is None, key
    assert rec["runtime"] == "claude"


def test_record_rejects_unknown_enums(_isolated_ledger: Path) -> None:
    base = [
        "record",
        "--job",
        "j1",
        "--task",
        "t1",
        "--agent",
        "w4",
        "--runtime",
        "cmd",
        "--model",
        MODEL_A,
    ]
    tel.main(
        base
    )  # seed: the review subcommand must fail on the enum itself, not on "unknown task"
    for extra in (["--runtime", "gemini"], ["--outcome", "maybe"], ["--review", "MAYBE"]):
        with pytest.raises(SystemExit):
            tel.main([*base, *extra])
    with pytest.raises(SystemExit):
        tel.main(["review", "--job", "j1", "--task", "t1", "--review", "MAYBE"])
    assert len(_ledger_rows(_isolated_ledger)) == 1  # bad inputs wrote nothing


def test_ingest_success_parses_result(_isolated_ledger: Path, tmp_path: Path) -> None:
    log = tmp_path / "20261009-120000-t1-xiaomi_mimo-v2.6-pro.ndjson"
    _write_log(
        log,
        '{"type":"event","event":{"type":"run_start"}}\n' + RESULT_SUCCESS + "\n",
        encoding="utf-8",
    )
    rc = tel.main(
        [
            "ingest-cmd-log",
            str(log),
            "--job",
            "j1",
            "--task",
            "t1",
            "--agent",
            "w4",
        ]
    )
    assert rc == 0
    (rec,) = _ledger_rows(_isolated_ledger)
    assert rec == {
        "ts": rec["ts"],
        "job": "j1",
        "task": "t1",
        "agent": "w4",
        "runtime": "cmd",
        "model": MODEL_A,
        "round": 1,
        "outcome": "done",
        "review": None,
        "tokens_in": 508685,
        "tokens_out": 7881,
        "wall_s": 315.493,
        "bead": None,
        "note": None,
        "source": None,
        "comms_asked": None,
        "comms_dms": None,
        "comms_replies": None,
        "board_relied": None,
    }


def test_ingest_max_turns_sets_note(_isolated_ledger: Path, tmp_path: Path) -> None:
    log = tmp_path / "20261009-120000-t1-xiaomi_mimo-v2.6-pro.ndjson"
    _write_log(
        log,
        '{"type":"result","subtype":"error","stopReason":"max_turns","durationMs":5000,'
        '"usage":{"inputTokens":10,"outputTokens":2}}\n',
        encoding="utf-8",
    )
    tel.main(["ingest-cmd-log", str(log), "--job", "j1", "--task", "t1", "--agent", "w4"])
    (rec,) = _ledger_rows(_isolated_ledger)
    assert rec["outcome"] == "error"
    assert rec["note"] == "stopReason: max_turns"
    assert rec["wall_s"] == 5.0


def test_ingest_no_result_event(_isolated_ledger: Path, tmp_path: Path) -> None:
    log = tmp_path / "20261009-120000-t1-xiaomi_mimo-v2.6-pro.ndjson"
    _write_log(log, '{"type":"event","event":{"type":"run_start"}}\n', encoding="utf-8")
    tel.main(["ingest-cmd-log", str(log), "--job", "j1", "--task", "t1", "--agent", "w4"])
    (rec,) = _ledger_rows(_isolated_ledger)
    assert rec["outcome"] == "stalled"
    assert rec["note"] == "no result event (running or killed)"
    assert rec["tokens_in"] is None
    assert rec["wall_s"] is None


def test_ingest_last_result_wins_and_round(_isolated_ledger: Path, tmp_path: Path) -> None:
    log = tmp_path / "20261009-120000-t1-xiaomi_mimo-v2.6-pro.ndjson"
    first = (
        '{"type":"result","subtype":"error","stopReason":"max_turns","durationMs":1000,'
        '"usage":{"inputTokens":1,"outputTokens":1}}'
    )
    _write_log(log, first + "\n" + RESULT_SUCCESS + "\n", encoding="utf-8")
    tel.main(
        [
            "ingest-cmd-log",
            str(log),
            "--job",
            "j1",
            "--task",
            "t1",
            "--agent",
            "w4",
            "--round",
            "3",
        ]
    )
    (rec,) = _ledger_rows(_isolated_ledger)
    assert rec["outcome"] == "done"
    assert rec["note"] is None
    assert rec["tokens_in"] == 508685
    assert rec["round"] == 3


def test_ingest_model_precedence(_isolated_ledger: Path, tmp_path: Path) -> None:
    events = '{"type":"event","event":{"type":"model_request_start","model":"zai-org/GLM-5.3"}}\n'
    result = '{"type":"result","subtype":"success","stopReason":"end_turn","durationMs":1,'
    result += '"usage":{"inputTokens":1,"outputTokens":1}}\n'

    def ingest(name: str, body: str, extra: list[str] | None = None) -> dict[str, Any]:
        log = tmp_path / name
        _write_log(log, body, encoding="utf-8")
        tel.main(
            [
                "ingest-cmd-log",
                str(log),
                "--job",
                "j1",
                "--task",
                "t1",
                "--agent",
                "w4",
                *(extra or []),
            ]
        )
        return _ledger_rows(_isolated_ledger)[-1]

    # (1) --model flag wins over both event and filename
    rec = ingest(
        "20261009-120000-t1-xiaomi_mimo-v2.6-pro.ndjson",
        events + result,
        ["--model", "flag/model"],
    )
    assert rec["model"] == "flag/model"
    # (2) event field wins over a parseable filename
    rec = ingest("20261009-120000-t1-xiaomi_mimo-v2.6-pro.ndjson", events + result)
    assert rec["model"] == "zai-org/glm-5.3"  # lowercased once in make_record
    # (3) filename fallback (dash-in-vendor brief name)
    rec = ingest("20261008-190858-t23-mock-uid-zai-org_glm-5.3.ndjson", result)
    assert rec["model"] == "zai-org/glm-5.3"
    # (4) unknown -> null + note
    rec = ingest("20261009-120000-t1-acme_widget-9.0.ndjson", result)
    assert rec["model"] is None
    assert rec["note"] == "model unknown"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("20261009-192736-w4-xiaomi_mimo-v2.6-pro.ndjson", "xiaomi/mimo-v2.6-pro"),
        (
            "20261008-205820-p1-fix2-deepseek_deepseek-v4.1-flash-fast.ndjson",
            "deepseek/deepseek-v4.1-flash-fast",
        ),
        ("20261008-190858-t23-mock-uid-zai-org_glm-5.3.ndjson", "zai-org/glm-5.3"),
        ("20261008-222044-q-ui-qwen_qwen3.8-max-0902.ndjson", "qwen/qwen3.8-max-0902"),
        # brief name containing both '_' and '-' must not break the split
        ("20261009-120000-my_fix-run-xiaomi_mimo-v2.6-pro.ndjson", "xiaomi/mimo-v2.6-pro"),
        # brief name containing '-deepseek_': the LAST '-<vendor>_' occurrence wins
        ("20261009-120000-fix-deepseek_log-xiaomi_mimo-v2.6-pro.ndjson", "xiaomi/mimo-v2.6-pro"),
        ("some-log-qwen_qwen3.8-max-0902.ndjson", "qwen/qwen3.8-max-0902"),
        ("20261009-120000-t1-acme_widget-9.0.ndjson", None),
    ],
)
def test_model_from_logname(name: str, expected: str | None) -> None:
    assert tel.model_from_logname(name) == expected


def test_review_appends_update_record(_isolated_ledger: Path) -> None:
    base = [
        "record",
        "--job",
        "j1",
        "--task",
        "t1",
        "--agent",
        "w4",
        "--runtime",
        "cmd",
        "--model",
        MODEL_A,
        "--outcome",
        "done",
    ]
    tel.main(base)
    tel.main(["review", "--job", "j1", "--task", "t1", "--review", "PASS"])
    tel.main(["review", "--job", "j1", "--task", "t1", "--review", "BLOCK"])
    rows = _ledger_rows(_isolated_ledger)
    assert len(rows) == 3  # append-only: updates are new records
    assert rows[0]["review"] is None
    assert rows[1]["review"] == "PASS"
    assert rows[2]["review"] == "BLOCK"
    # the update is a copy of the latest record for (job, task) with review set
    for key in ("job", "task", "agent", "runtime", "model", "round", "outcome", "bead", "note"):
        assert rows[2][key] == rows[1][key], key


def test_review_copies_latest_for_task(_isolated_ledger: Path) -> None:
    base = ["record", "--job", "j1", "--task", "t1", "--agent", "w4", "--runtime", "cmd"]
    tel.main([*base, "--model", MODEL_A, "--round", "1", "--outcome", "error"])
    tel.main([*base, "--model", MODEL_B, "--round", "2", "--outcome", "done"])
    tel.main(["review", "--job", "j1", "--task", "t1", "--review", "PASS"])
    rows = _ledger_rows(_isolated_ledger)
    assert len(rows) == 3
    assert rows[-1]["round"] == 2
    assert rows[-1]["outcome"] == "done"
    assert rows[-1]["model"] == MODEL_B
    assert rows[-1]["review"] == "PASS"


def test_review_unknown_task_fails(_isolated_ledger: Path) -> None:
    with pytest.raises(SystemExit):
        tel.main(["review", "--job", "j1", "--task", "nope", "--review", "PASS"])
    assert not _isolated_ledger.exists()


def test_report_by_model_math(_isolated_ledger: Path) -> None:
    _seed(_isolated_ledger, LEDGER)
    rows = tel.report_rows(_ledger_rows(_isolated_ledger), by="model")
    assert rows == [ROW_B, ROW_A]  # sorted by group name


def test_report_by_runtime_math(_isolated_ledger: Path) -> None:
    _seed(_isolated_ledger, LEDGER)
    rows = tel.report_rows(_ledger_rows(_isolated_ledger), by="runtime")
    assert rows == [
        {
            "group": "claude",
            "tasks": 1,
            "pass1_passes": 0,
            "pass1_attempts": 0,
            "pass1_rate": None,
            "mean_rounds": None,  # t2's round 1 was cmd: the task is charged to the cmd group
            "med_tokens_in": 300.0,
            "med_wall_s": 30.0,
            "stalls_errors": 0,
        },
        {
            "group": "cmd",
            "tasks": 4,
            "pass1_passes": 2,
            "pass1_attempts": 3,
            "pass1_rate": 2 / 3,
            "mean_rounds": 4 / 3,
            "med_tokens_in": 300.0,
            "med_wall_s": 30.0,
            "stalls_errors": 2,
        },
    ]


def test_report_since_filters_before_dedupe(_isolated_ledger: Path) -> None:
    _seed(_isolated_ledger, LEDGER)
    rows = tel.report_rows(_ledger_rows(_isolated_ledger), by="model", since="2026-10-09")
    assert rows == [
        ROW_B,
        {
            "group": MODEL_A,
            "tasks": 2,
            "pass1_passes": 1,
            "pass1_attempts": 2,
            "pass1_rate": 0.5,
            "mean_rounds": 1.5,
            "med_tokens_in": 150.0,
            "med_wall_s": 15.0,
            "stalls_errors": 1,
        },
    ]


def test_report_output_tables(_isolated_ledger: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed(_isolated_ledger, LEDGER)
    tel.main(["report", "--by", "model"])
    plain = capsys.readouterr().out
    assert "group" in plain.splitlines()[0]
    assert "1/2 (0.50)" in plain
    assert MODEL_A in plain

    tel.main(["report", "--by", "model", "--markdown"])
    md = capsys.readouterr().out
    assert md.splitlines()[0].startswith("| group |")
    assert "| --- |" in md
    assert f"| {MODEL_A} | 3 | 1/2 (0.50) | 1.50 | 200 | 20 | 2 |" in md


def _row(rows: list[dict[str, Any]], group: str) -> dict[str, Any] | None:
    return next((r for r in rows if r["group"] == group), None)


def test_model_normalised_lowercase(_isolated_ledger: Path, tmp_path: Path) -> None:
    """B1: event 'Qwen/X' and flag 'qwen/x' are one report group."""
    log = tmp_path / "20261009-120000-t1-acme_widget-9.0.ndjson"
    ev = '{"type":"event","event":{"type":"model_request_start","model":"Qwen/X"}}\n'
    _write_log(log, ev + RESULT_SUCCESS + "\n")
    tel.main(["ingest-cmd-log", str(log), "--job", "j1", "--task", "t1", "--agent", "w"])
    tel.main(
        [
            "record",
            "--job",
            "j1",
            "--task",
            "t2",
            "--agent",
            "w",
            "--runtime",
            "cmd",
            "--model",
            " qwen/x ",
        ]
    )
    rows = _ledger_rows(_isolated_ledger)
    assert [r["model"] for r in rows] == ["qwen/x", "qwen/x"]
    # legacy mixed-case rows already in the ledger merge too
    rows[0]["model"] = "Qwen/X"
    out = tel.report_rows(rows, by="model")
    assert [(r["group"], r["tasks"]) for r in out] == [("qwen/x", 2)]


def test_rounds_charged_to_round1_model_and_need_pass() -> None:
    """B2: X BLOCK at r1, Y PASS at r2 -> X mean_rounds 2, Y excluded."""
    recs = [
        _rec(task="t", model="x/m", round=1, outcome="done", review="BLOCK"),
        _rec(task="t", model="y/m", round=2, outcome="done", review="PASS"),
    ]
    rows = tel.report_rows(recs, by="model")
    assert _row(rows, "x/m")["mean_rounds"] == 2.0  # type: ignore[index]
    assert _row(rows, "y/m")["mean_rounds"] is None  # type: ignore[index]


def test_rounds_unreviewed_task_excluded_and_earliest_pass_wins() -> None:
    recs = [
        _rec(task="a", round=1, outcome="done"),  # no review: excluded
        _rec(task="b", round=1, outcome="done", review="PASS"),
        _rec(task="b", round=2, outcome="done", review="PASS"),  # PASS at 1 and 2 -> 1
    ]
    (row,) = tel.report_rows(recs, by="model")
    assert row["mean_rounds"] == 1.0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("w4-fix", ("w4", 2)),
        ("w4-fix2", ("w4", 3)),
        ("w4-r3", ("w4", 3)),
        ("w4-r", ("w4", 2)),
        ("a-fix-b-fix", ("a-fix-b", 2)),
        ("w4", None),
        ("x-r0", None),
        ("x-fix0", None),
        ("x-r1", ("x", 1)),
        ("prefix", None),
    ],
)
def test_derive_round(name: str, expected: tuple[str, int] | None) -> None:
    assert tel.derive_round(name) == expected


def test_ingest_auto_round(
    _isolated_ledger: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "20261009-120000-w4-fix2-xiaomi_mimo-v2.6-pro.ndjson"
    _write_log(log, RESULT_SUCCESS + "\n")
    args = ["ingest-cmd-log", str(log), "--job", "j1", "--agent", "w", "--auto-round"]
    tel.main([*args, "--task", "w4-fix2"])
    (rec,) = _ledger_rows(_isolated_ledger)
    assert (rec["task"], rec["round"]) == ("w4", 3)
    assert rec["note"] == "round derived from name"
    captured = capsys.readouterr()
    assert "task='w4' round=3" in captured.err
    # explicit --round is authoritative, even with --auto-round
    tel.main([*args, "--task", "w4", "--round", "5"])
    assert _ledger_rows(_isolated_ledger)[-1]["round"] == 5
    assert _ledger_rows(_isolated_ledger)[-1]["note"] is None


def test_review_round_selects_record(_isolated_ledger: Path) -> None:
    base = ["record", "--job", "j1", "--task", "t1", "--agent", "w", "--runtime", "cmd"]
    tel.main([*base, "--model", MODEL_A, "--round", "1", "--outcome", "done"])
    tel.main([*base, "--model", MODEL_B, "--round", "2", "--outcome", "done"])
    tel.main(["review", "--job", "j1", "--task", "t1", "--review", "BLOCK", "--round", "1"])
    last = _ledger_rows(_isolated_ledger)[-1]
    assert (last["round"], last["model"], last["review"]) == (1, MODEL_A, "BLOCK")
    with pytest.raises(SystemExit):
        tel.main(["review", "--job", "j1", "--task", "t1", "--review", "PASS", "--round", "9"])


def test_since_validated() -> None:
    with pytest.raises(SystemExit):
        tel.main(["report", "--since", "yesterday"])


def test_fresh_log_refused_unless_force(_isolated_ledger: Path, tmp_path: Path) -> None:
    log = tmp_path / "20261009-120000-t1-xiaomi_mimo-v2.6-pro.ndjson"
    _write_log(log, RESULT_SUCCESS + "\n", age_s=5)
    args = ["ingest-cmd-log", str(log), "--job", "j1", "--task", "t1", "--agent", "w"]
    with pytest.raises(SystemExit):
        tel.main(args)
    assert not _isolated_ledger.exists()
    tel.main([*args, "--force"])
    assert len(_ledger_rows(_isolated_ledger)) == 1


def test_report_does_not_create_ledger_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SWARM_TELEMETRY", str(tmp_path / "newdir" / "tasks.jsonl"))
    tel.main(["report"])
    assert not (tmp_path / "newdir").exists()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )  # fmt: skip


def _copy_script(dst_root: Path) -> ModuleType:
    d = dst_root / "scripts" / "agents"
    d.mkdir(parents=True, exist_ok=True)
    (d / "telemetry.py").write_text(_TELEMETRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("swarm_telemetry_copy", d / "telemetry.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ledger_path_uses_main_checkout_from_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SWARM_TELEMETRY", raising=False)
    main = (tmp_path / "main").resolve()
    main.mkdir()
    _git(main, "init", "-q")
    _copy_script(main)
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", str(wt))
    mod = _copy_script(wt)  # the worktree's own checkout of the script
    expected = main / "docs" / "orchestration" / "telemetry" / "tasks.jsonl"
    assert mod.ledger_path().resolve() == expected.resolve()


def test_ledger_path_outside_git_is_clean_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SWARM_TELEMETRY", raising=False)
    mod = _copy_script(tmp_path / "nogit")
    with pytest.raises(SystemExit) as exc:
        mod.ledger_path()
    assert "SWARM_TELEMETRY" in str(exc.value)


def test_since_does_not_change_round_attribution() -> None:
    """SF1: X BLOCK r1 (10-08), Y PASS r2 (10-09), --since 10-09 still charges X 2.0."""
    recs = [
        _rec(ts="2026-10-08T10:00:00Z", model="x/m", round=1, outcome="done", review="BLOCK"),
        _rec(ts="2026-10-09T10:00:00Z", model="y/m", round=2, outcome="done", review="PASS"),
    ]
    rows = tel.report_rows(recs, by="model", since="2026-10-09")
    assert _row(rows, "y/m")["mean_rounds"] is None  # type: ignore[index]
    assert _row(rows, "x/m")["mean_rounds"] == 2.0  # type: ignore[index]
    both = tel.report_rows(
        [*recs, _rec(ts="2026-10-09T11:00:00Z", model="x/m", task="u", round=1, review="PASS")],
        by="model",
        since="2026-10-09",
    )
    assert _row(both, "x/m")["mean_rounds"] == 1.5  # type: ignore[index]
    # A task entirely before --since is not charged (only the tasks count shows it).
    old = [
        _rec(ts="2026-10-08T09:00:00Z", model="x/m", task="old", round=1, review="BLOCK"),
        _rec(ts="2026-10-08T09:30:00Z", model="x/m", task="old", round=2, review="PASS"),
    ]
    windowed = tel.report_rows([*recs, *old], by="model", since="2026-10-09")
    assert _row(windowed, "x/m")["tasks"] == 1  # type: ignore[index]


def test_task_without_round1_excluded_from_rounds() -> None:
    """SF2: only a round-2 record exists -> not charged to anyone."""
    recs = [_rec(task="sa", model="y/m", round=2, outcome="done", review="PASS")]
    (row,) = tel.report_rows(recs, by="model")
    assert row["mean_rounds"] is None


def test_latest_attempts_keeps_retry_and_collapses_review_copy() -> None:
    """A retry is its own attempt; a review update is a copy of one (vibe-quant-kzbc6)."""
    first = _rec(task="t", model="mimo/m", round=1, outcome="stalled")
    second = _rec(task="t", model="deepseek/m", round=1, outcome="done")
    review_copy = _rec(
        ts="2026-10-09T11:00:00Z", task="t", model="deepseek/m", round=1,
        outcome="done", review="PASS",
    )  # fmt: skip
    attempts = tel.latest_attempts([first, second, review_copy])
    assert [r["model"] for r in attempts] == ["mimo/m", "deepseek/m"]
    assert attempts[-1]["review"] == "PASS"  # the review update still supersedes its source


def test_report_keeps_retry_attempts_per_model() -> None:
    """MiMo stall -> DeepSeek done+PASS: both attempts stay, each under its own model."""
    recs = [
        _rec(task="t", model="mimo/m", round=1, outcome="stalled", tokens_in=100, wall_s=10.0),
        _rec(
            task="t",
            model="deepseek/m",
            round=1,
            outcome="done",
            review="PASS",
            tokens_in=300,
            wall_s=30.0,
        ),  # fmt: skip
    ]
    rows = tel.report_rows(recs, by="model")
    stalled = _row(rows, "mimo/m")
    done = _row(rows, "deepseek/m")
    assert stalled is not None and done is not None
    # every attempt is counted under its own model
    assert (stalled["tasks"], stalled["stalls_errors"]) == (1, 1)
    assert (stalled["med_tokens_in"], stalled["med_wall_s"]) == (100, 10)
    # pass-1/rounds follow the LATEST attempt (unchanged single-attempt behaviour)
    assert (stalled["pass1_passes"], stalled["pass1_attempts"]) == (0, 0)
    assert (done["pass1_passes"], done["pass1_attempts"]) == (1, 1)
    assert done["mean_rounds"] == 1.0
    assert (done["tasks"], done["stalls_errors"]) == (1, 0)
    assert (done["med_tokens_in"], done["med_wall_s"]) == (300, 30)


def test_report_by_model_shows_attempts_column(
    _isolated_ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(_isolated_ledger, LEDGER)
    tel.main(["report", "--by", "model", "--markdown"])
    md = capsys.readouterr().out
    assert md.splitlines()[0].endswith("| attempts |")
    row_a = next(line for line in md.splitlines() if line.startswith(f"| {MODEL_A} "))
    assert row_a.endswith("| 3 |")  # MODEL_A has 3 attempts in LEDGER


def test_pass1_uses_latest_attempt_only() -> None:
    """Same (job, task, round): mimo SHOULD-FIX then deepseek PASS. The superseded attempt must
    not count as a mimo pass-1 failure; the pass-1 attempt belongs to deepseek alone."""
    recs = [
        _rec(task="t", model="mimo/m", round=1, outcome="done", review="SHOULD-FIX"),
        _rec(task="t", model="deepseek/m", round=1, outcome="done", review="PASS"),
    ]
    rows = tel.report_rows(recs, by="model")
    mimo = _row(rows, "mimo/m")
    deepseek = _row(rows, "deepseek/m")
    assert mimo is not None and deepseek is not None
    assert (mimo["pass1_passes"], mimo["pass1_attempts"]) == (0, 0)
    assert mimo["tasks"] == 1  # but the attempt still counts as a task of mimo
    assert (deepseek["pass1_passes"], deepseek["pass1_attempts"]) == (1, 1)
    assert deepseek["pass1_rate"] == 1.0
    assert deepseek["mean_rounds"] == 1.0
    assert mimo["mean_rounds"] is None


def test_attempt_counts_count_every_attempt_per_group() -> None:
    recs = [
        _rec(task="t", model="mimo/m", round=1, outcome="stalled"),
        _rec(task="t", model="deepseek/m", round=1, outcome="done"),
    ]
    assert tel.attempt_counts(recs, "model") == {"mimo/m": 1, "deepseek/m": 1}
    # the same retry pair: one agent, two attempts
    assert tel.attempt_counts(recs, "agent") == {"w1": 2}


def test_identical_stalled_retries_count_as_separate_attempts() -> None:
    """Two stalled records identical apart from their ts (a relaunch that stalls the same way)
    are two attempts and two stalls: only a review copy may collapse onto its source."""
    recs = [
        _rec(ts="2026-10-09T10:00:00", task="t", model="mimo/m", round=1, outcome="stalled"),
        _rec(ts="2026-10-09T10:30:00", task="t", model="mimo/m", round=1, outcome="stalled"),
    ]
    assert tel.attempt_counts(recs, "model") == {"mimo/m": 2}
    row = _row(tel.report_rows(recs, by="model"), "mimo/m")
    assert row is not None
    assert row["stalls_errors"] == 2


def test_review_copy_collapses_onto_its_source_not_the_latest_attempt() -> None:
    """A review copy replaces the earlier attempt it matches, not whatever came last: a
    round-1 review arriving after a round-2 record still collapses onto round 1."""
    recs = [
        _rec(task="t", model="mimo/m", round=1, outcome="done"),
        _rec(task="t", model="deepseek/d", round=2, outcome="done"),
        _rec(ts="2026-10-09T11:00:00", task="t", model="mimo/m", round=1, outcome="done",
             review="BLOCK"),
    ]
    assert tel.attempt_counts(recs, "model") == {"mimo/m": 1, "deepseek/d": 1}


def test_attempt_counts_respect_since_window() -> None:
    recs = [
        _rec(ts="2026-10-08T09:00:00", task="old", model="mimo/m", round=1, outcome="stalled"),
        _rec(ts="2026-10-09T09:00:00", task="new", model="mimo/m", round=1, outcome="done"),
        _rec(ts="2026-10-09T09:30:00", task="new", model="deepseek/m", round=1, outcome="done"),
    ]
    assert tel.attempt_counts(recs, "model") == {"mimo/m": 2, "deepseek/m": 1}
    assert tel.attempt_counts(recs, "model", since="2026-10-09") == {
        "mimo/m": 1,
        "deepseek/m": 1,
    }


def test_attempts_column_printed_for_every_group_by(
    _isolated_ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The attempts column is not model-specific (documented in the module docstring)."""
    _seed(
        _isolated_ledger,
        [
            _rec(task="t", model="mimo/m", round=1, outcome="stalled"),
            _rec(task="t", model="deepseek/m", round=1, outcome="done"),
        ],
    )
    tel.main(["report", "--by", "agent", "--markdown"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("| attempts |")
    row = next(line for line in lines if line.startswith("| w1 "))
    assert row.endswith("| 2 |")


# --- ingest-runs: T3 delegated tasks, Command Code sessions, Claude subagent transcripts -------

JOB = "20261014-demo"
HANDOFF_COMMS = (
    "status: done\ncomms: asked=1 dms=2 replies=0; board posts relied on: "
    "18de7ab5245b2778-1430, 18de7ad6d75e3ec0-0531 (fee gotcha)\nimprove: none"
)


def _t3_db(path: Path, rows: list[dict[str, Any]]) -> Path:
    import sqlite3

    con = sqlite3.connect(path)
    con.execute(
        "create table orchestration_v2_projection_subagents (subagent_id text primary key, "
        "provider text, origin text, status text, started_at text, completed_at text, payload_json text)"
    )
    for i, r in enumerate(rows):
        cid = r["cid"]
        payload = {
            "id": f"node:delegated-task:command%3Amcp%3Aabc%3Adelegate-task%3A{cid}",
            "prompt": r["prompt"],
            "model": r["model"],
            "result": r.get("result"),
            "usage": r.get("usage"),
        }
        con.execute(
            "insert into orchestration_v2_projection_subagents values (?,?,?,?,?,?,?)",
            (f"s{i}", r["provider"], "app_owned", r["status"], r["start"], r.get("end"), json.dumps(payload)),
        )
    con.commit()
    con.close()
    return path


def _cmd_session(d: Path, start: str, prompt: str, usages: list[tuple[int, int]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        {"type": "session", "id": "x", "timestamp": start, "cwd": "/repo"},
        {"type": "message", "id": "u1", "timestamp": start,
         "message": {"role": "user", "content": [{"type": "text", "text": f"<t3_code_instructions>mode</t3_code_instructions>\n{prompt}"}]}},
    ]
    for k, (i, o) in enumerate(usages):
        lines.append({"type": "message", "id": f"a{k}", "timestamp": start,
                      "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                      "usage": {"inputTokens": i, "outputTokens": o, "cacheReadTokens": i - 10}})
    (d / f"{start[-6:-1]}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def _pointer_prompt(brief_id: str, task: str, rnd: int = 1) -> str:
    """A real T3 dispatch prompt: a short pointer whose first 80 chars every sibling shares."""
    return (
        "Act as the implementation sub-agent for this task.\n\n"
        f"Read your brief at /Users/verebelyin/projects/vibe-quant/data/swarm/{JOB}/prompts/{brief_id}.md"
        f" and follow it exactly.\nTELEMETRY: task={task} round={rnd}"
    )


def _claude_subagent(d: Path, agent_id: str, name: str, prompt: str, final: str, error: str | None = None) -> None:
    sub = d / "sess1" / "subagents"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"agent-{agent_id}.meta.json").write_text(json.dumps({"agentType": "implementer", "name": name}), encoding="utf-8")
    msgs: list[dict[str, Any]] = [
        {"type": "user", "timestamp": "2026-10-14T10:00:00.000Z", "message": {"role": "user", "content": prompt}},
        # the same API message is logged once per content block: usage must be counted once
        {"type": "assistant", "timestamp": "2026-10-14T10:01:00.000Z", "message": {"id": "m1", "model": "claude-sonnet-5-5",
         "content": [{"type": "thinking", "thinking": "x"}], "usage": {"input_tokens": 3, "cache_creation_input_tokens": 100,
         "cache_read_input_tokens": 1000, "output_tokens": 50}}},
        {"type": "assistant", "timestamp": "2026-10-14T10:01:00.000Z", "message": {"id": "m1", "model": "claude-sonnet-5-5",
         "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}], "usage": {"input_tokens": 3,
         "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": 50}}},
        {"type": "assistant", "timestamp": "2026-10-14T10:05:30.000Z", "message": {"id": "m2", "model": "claude-sonnet-5-5",
         "content": [{"type": "text", "text": final}], "usage": {"input_tokens": 2, "cache_creation_input_tokens": 0,
         "cache_read_input_tokens": 2000, "output_tokens": 70}}},
    ]
    if error:
        msgs.append({"type": "assistant", "timestamp": "2026-10-14T10:06:00.000Z", "isApiErrorMessage": True,
                     "message": {"id": "e", "model": "<synthetic>", "content": [{"type": "text", "text": error}]}})
    (sub / f"agent-{agent_id}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in msgs), encoding="utf-8")
    old = time.time() - 3600
    os.utime(sub / f"agent-{agent_id}.jsonl", (old, old))


def _sources(tmp_path: Path) -> dict[str, Path]:
    db = _t3_db(tmp_path / "t3.sqlite", [
        {"cid": "cc-liq-e3-2", "provider": "commandcode_command", "status": "completed",
         "start": "2026-10-14T09:00:00.000Z", "end": "2026-10-14T09:07:30.000Z", "model": "deepseek/deepseek-v4.1-flash",
         "prompt": f"ROLE: Command Code worker (agent id cc-liq-e3), job {JOB}. Fix the docs.", "result": HANDOFF_COMMS,
         "usage": {"totalTokens": 73591, "approximate": True}},
        {"cid": "rev-liq-e3-r1-20261014", "provider": "codex", "status": "failed",
         "start": "2026-10-14T09:10:00.000Z", "end": "2026-10-14T09:10:04.000Z", "model": "gpt-6.1-sol",
         "prompt": f"You are the reviewer (agent id rev-liq-e3-r1), job {JOB}.",
         "result": "You've hit your usage limit. Try again at 1:07 AM.", "usage": {"inputTokens": 900, "outputTokens": 9}},
        {"cid": "cc-other-1", "provider": "commandcode_command", "status": "completed",
         "start": "2026-10-14T09:00:00.000Z", "end": "2026-10-14T09:01:00.000Z", "model": "deepseek/deepseek-v4.1-flash",
         "prompt": "job 20261001-unrelated", "result": "done"},
        {"cid": "cc-running-1", "provider": "commandcode_command", "status": "running",
         "start": "2026-10-14T09:00:00.000Z", "model": "deepseek/deepseek-v4.1-flash", "prompt": f"job {JOB}"},
    ])
    cmd = tmp_path / "cmd"
    _cmd_session(cmd / "proj", "2026-10-14T09:00:01.500Z",
                 f"ROLE: Command Code worker (agent id cc-liq-e3), job {JOB}. Fix the docs.", [(1000, 20), (2000, 30)])
    claude = tmp_path / "claude"
    _claude_subagent(claude, "a1", "impl-liq-t2a-opus5",
                     f"TELEMETRY: task=liq-t2a round=5\nROLE: implementer, job {JOB}", HANDOFF_COMMS)
    _claude_subagent(claude, "a2", "rev-liq-t3-r2", f"ROLE: reviewer, job {JOB}", "verdict BLOCK\ncomms: asked=0 dms=1 replies=0; board posts relied on: none",
                     error="API Error: You've hit your session limit")
    return {"t3_db": db, "cmd_dir": cmd, "claude_dir": claude}


def test_parse_comms_line() -> None:
    assert tel.parse_comms(HANDOFF_COMMS) == {"comms_asked": 1, "comms_dms": 2, "comms_replies": 0, "board_relied": 2}
    assert tel.parse_comms("comms: asked=0 dms=0 replies=0; board posts relied on: none")["board_relied"] == 0
    assert tel.parse_comms("comms: asked=3 dms=0 replies=1; board posts relied on: `18dd86415580cc90-8734` (x)")["board_relied"] == 1
    assert tel.parse_comms("no comms line here") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cc-liq-e3-2", ("cc-liq-e3", "liq-e3", 2)),  # T3 clientRequestId cc-<slug>-<round>
        ("cc-liq-e3-1", ("cc-liq-e3", "liq-e3", 1)),
        ("cc-liq-t1-r4-1", ("cc-liq-t1-r4", "liq-t1", 4)),  # round in the slug wins over the retry digit
        ("rev-liq-e3-r1-20261014", ("rev-liq-e3-r1", "review:liq-e3", 1)),  # dated Codex id
        ("rev-liq-t3-r2", ("rev-liq-t3-r2", "review:liq-t3", 2)),
        ("revlite-liq-e3", ("revlite-liq-e3", "review:liq-e3", 1)),
        ("impl-liq-t2a-opus5", ("impl-liq-t2a-opus5", "liq-t2a", 5)),
        ("impl-sqlite", ("impl-sqlite", "sqlite", 1)),
        ("verifier-p1d", ("verifier-p1d", "verify:p1d", 1)),
        ("architect-liq", ("architect-liq", "design:liq", 1)),
        ("impl-d4-fix2", ("impl-d4-fix2", "d4", 3)),
    ],
)
def test_identity_from_name(name: str, expected: tuple[str, str, int]) -> None:
    assert tel.identity_from_name(name) == expected


def test_ingest_runs_reads_t3_command_code_and_claude(_isolated_ledger: Path, tmp_path: Path) -> None:
    src = _sources(tmp_path)
    added = tel.ingest_runs(JOB, **src)
    rows = {r["source"]: r for r in _ledger_rows(_isolated_ledger)}
    assert len(added) == 4 and set(rows) == {"t3:cc-liq-e3-2", "t3:rev-liq-e3-r1-20261014", "claude:a1", "claude:a2"}
    cc = rows["t3:cc-liq-e3-2"]
    assert (cc["job"], cc["task"], cc["agent"], cc["round"], cc["runtime"]) == (JOB, "liq-e3", "cc-liq-e3", 2, "cmd")
    # T3's Command Code usage is a last-context approximation: tokens come from the session file
    assert (cc["tokens_in"], cc["tokens_out"], cc["wall_s"], cc["outcome"]) == (3000, 50, 450.0, "done")
    assert (cc["comms_asked"], cc["comms_dms"], cc["comms_replies"], cc["board_relied"]) == (1, 2, 0, 2)
    cdx = rows["t3:rev-liq-e3-r1-20261014"]
    assert (cdx["runtime"], cdx["task"], cdx["outcome"], cdx["tokens_in"], cdx["tokens_out"]) == ("codex", "review:liq-e3", "error", 900, 9)
    assert "usage limit" in cdx["note"]
    a1 = rows["claude:a1"]  # TELEMETRY line beats the name heuristic
    assert (a1["task"], a1["round"], a1["agent"], a1["model"]) == ("liq-t2a", 5, "impl-liq-t2a-opus5", "claude-sonnet-5-5")
    assert (a1["tokens_in"], a1["tokens_out"], a1["wall_s"]) == (1103 + 2002, 120, 330.0)  # m1 counted once
    assert a1["board_relied"] == 2
    a2 = rows["claude:a2"]
    assert (a2["outcome"], a2["task"], a2["round"], a2["board_relied"]) == ("error", "review:liq-t3", 2, 0)
    assert "session limit" in a2["note"]


def test_ingest_runs_is_idempotent_and_skips_manual_records(_isolated_ledger: Path, tmp_path: Path) -> None:
    src = _sources(tmp_path)
    tel.main(["record", "--job", JOB, "--task", "liq-e3", "--agent", "cc-liq-e3", "--runtime", "cmd", "--round", "2"])
    first = tel.ingest_runs(JOB, **src)
    assert {r["source"] for r in first} == {"t3:rev-liq-e3-r1-20261014", "claude:a1", "claude:a2"}
    assert tel.ingest_runs(JOB, **src) == []  # nothing new on a re-run
    assert len(_ledger_rows(_isolated_ledger)) == 4


def test_ingest_runs_dry_run_writes_nothing(_isolated_ledger: Path, tmp_path: Path) -> None:
    src = _sources(tmp_path)
    assert len(tel.ingest_runs(JOB, dry_run=True, **src)) == 4
    assert not _isolated_ledger.exists()


def _t3_run(cid: str, start: str, end: str, prompt: str) -> dict[str, Any]:
    return {"cid": cid, "provider": "commandcode_command", "status": "completed",
            "start": start, "end": end, "model": MODEL_B, "prompt": prompt, "result": "done"}  # fmt: skip


def test_ingest_runs_gives_each_cc_child_its_own_session(_isolated_ledger: Path, tmp_path: Path) -> None:
    # Sibling dispatch prompts share their first 80 chars (only the brief id / TELEMETRY tail
    # differ), so matching on that prefix handed every child the same session's tokens.
    p1 = _pointer_prompt("cc-sh-telemetry-1", "sh-telemetry")
    p2 = _pointer_prompt("cc-sh-telemetry-2", "sh-telemetry")
    p3 = _pointer_prompt("cc-sh-telemetry-3", "sh-telemetry")
    assert p1[:80] == p2[:80] == p3[:80] and len({p1, p2, p3}) == 3
    db = _t3_db(tmp_path / "t3.sqlite", [
        _t3_run("cc-sh-telemetry-1", "2026-10-17T15:00:00.000Z", "2026-10-17T15:10:00.000Z", p1),
        _t3_run("cc-sh-telemetry-2", "2026-10-17T15:00:04.000Z", "2026-10-17T15:10:04.000Z", p2),
        _t3_run("cc-sh-telemetry-3", "2026-10-17T15:00:02.000Z", "2026-10-17T15:10:02.000Z", p3),  # no session
    ])
    cmd = tmp_path / "cmd"
    # The child that launched first (p1) is the LAST to open its session, so its closest session
    # is the sibling's (proj-b): only the whole-prompt match routes it to its own file.
    _cmd_session(cmd / "proj-a", "2026-10-17T15:00:10.000Z", p1, [(1000, 10)])
    _cmd_session(cmd / "proj-b", "2026-10-17T15:00:01.000Z", p2, [(2000, 20)])
    added = tel.ingest_runs(JOB, t3_db=db, cmd_dir=cmd, claude_dir=tmp_path / "nocl")
    toks = {r["source"]: (r["tokens_in"], r["tokens_out"]) for r in added}
    assert toks == {
        "t3:cc-sh-telemetry-1": (1000, 10),
        "t3:cc-sh-telemetry-2": (2000, 20),
        "t3:cc-sh-telemetry-3": (None, None),  # unmatched keeps None, never a sibling's numbers
    }
    assert next(r for r in added if r["source"] == "t3:cc-sh-telemetry-3")["note"] == "tokens unknown"


def test_ingest_runs_pairs_a_rerun_with_the_closest_unclaimed_session(
    _isolated_ledger: Path, tmp_path: Path
) -> None:
    # One task dispatched twice with the same pointer: each attempt takes its own closest session
    # and no session file is counted twice.
    p = _pointer_prompt("cc-sh-retry-1", "sh-retry")
    db = _t3_db(tmp_path / "t3.sqlite", [
        _t3_run("cc-sh-retry-1", "2026-10-17T16:00:00.000Z", "2026-10-17T16:10:00.000Z", p),
        _t3_run("cc-sh-retry-2", "2026-10-17T16:00:04.000Z", "2026-10-17T16:10:04.000Z", p),
    ])
    cmd = tmp_path / "cmd"
    _cmd_session(cmd / "proj-a", "2026-10-17T16:00:01.000Z", p, [(100, 1)])
    _cmd_session(cmd / "proj-b", "2026-10-17T16:00:31.000Z", p, [(300, 3)])
    added = tel.ingest_runs(JOB, t3_db=db, cmd_dir=cmd, claude_dir=tmp_path / "nocl")
    toks = {r["source"]: (r["tokens_in"], r["tokens_out"]) for r in added}
    assert toks == {"t3:cc-sh-retry-1": (100, 1), "t3:cc-sh-retry-2": (300, 3)}


def test_cmd_session_usage_takes_the_closest_start_not_the_first_path(tmp_path: Path) -> None:
    # Two session files carry the same whole prompt. The one that starts 90 s late but sorts
    # FIRST by path must not win over the +2 s file: the "first match wins" mutant survives
    # unless the closest start is asserted (proj-a sorts before proj-b).
    prompt = _pointer_prompt("cc-sh-near-1", "sh-near")
    start = tel._iso_s("2026-10-17T16:00:00.000Z")
    assert start is not None
    cmd = tmp_path / "cmd"
    _cmd_session(cmd / "proj-a", "2026-10-17T16:01:30.000Z", prompt, [(900, 90)])  # +90 s, sorts first
    _cmd_session(cmd / "proj-b", "2026-10-17T16:00:02.000Z", prompt, [(200, 2)])  # +2 s
    claimed: set[Path] = set()
    assert tel._cmd_session_usage(cmd, start, prompt, claimed) == (200, 2)
    assert claimed == {cmd / "proj-b" / "2.000.jsonl"}


def test_report_by_board_and_asked(_isolated_ledger: Path) -> None:
    recs = [
        _rec(task="a", round=1, review="PASS", board_relied=2, comms_asked=0),
        _rec(task="b", round=1, review="BLOCK", board_relied=0, comms_asked=1),
        _rec(task="c", round=1, review="PASS", board_relied=0, comms_asked=0),
        _rec(task="d", round=1, review="PASS"),
    ]
    by_board = {r["group"]: (r["pass1_passes"], r["pass1_attempts"]) for r in tel.report_rows(recs, by="board")}
    assert by_board == {"relied>0": (1, 1), "relied=0": (1, 2), "unknown": (1, 1)}
    by_asked = {r["group"]: (r["pass1_passes"], r["pass1_attempts"]) for r in tel.report_rows(recs, by="asked")}
    assert by_asked == {"asked>0": (0, 1), "asked=0": (2, 2), "unknown": (1, 1)}


# --- round suffixes without a "-r"/"-fix" marker (cc-e6-sweep-2 was recorded as round 1) ------


def test_derive_round_reads_a_bare_round_suffix() -> None:
    assert tel.derive_round("cc-e6-sweep-2") == ("cc-e6-sweep", 2)
    assert tel.derive_round("cc-e6-sweep-1") == ("cc-e6-sweep", 1)
    assert tel.derive_round("e6-sweep-2") == ("e6-sweep", 2)
    assert tel.derive_round("cc-e6-sweep-0") is None  # rounds start at 1
    assert tel.derive_round("vibe-quant-20261014") is None  # 8 digits are a date, not a round


def test_identity_from_name_reads_a_bare_round_suffix() -> None:
    assert tel.identity_from_name("e6-sweep-2") == ("e6-sweep-2", "e6-sweep", 2)
    # a model suffix is still a model: "-opus5" is not the round-5 retry
    assert tel.identity_from_name("impl-liq-t2a-opus5") == ("impl-liq-t2a-opus5", "liq-t2a", 5)


def _e6_db(path: Path) -> Path:
    """Round 1 and round 2 workers of one task, ids cc-e6-sweep-<n>, with no TELEMETRY line."""
    def row(cid: str, prompt: str) -> dict[str, Any]:
        return {"cid": cid, "provider": "commandcode_command", "status": "completed",
                "start": "2026-10-17T15:00:00.000Z", "end": "2026-10-17T15:10:00.000Z",
                "model": MODEL_B, "prompt": prompt, "result": "done"}  # fmt: skip

    return _t3_db(path, [
        row("cc-e6-sweep-1", f"ROLE: Command Code worker (agent id cc-e6-sweep-1), job {JOB}. Sweep."),
        row("cc-e6-sweep-2", f"ROLE: Command Code worker (agent id cc-e6-sweep-2), job {JOB}. Sweep again."),
        row("cc-e6-sweep-9", f"TELEMETRY: task=e6-sweep round=8\nROLE: worker, job {JOB}"),  # explicit line wins
    ])


def test_ingest_runs_records_cc_ids_as_rounds_of_one_task(_isolated_ledger: Path, tmp_path: Path) -> None:
    added = tel.ingest_runs(JOB, t3_db=_e6_db(tmp_path / "t3.sqlite"), cmd_dir=tmp_path / "nocmd", claude_dir=tmp_path / "nocl")
    assert sorted((r["task"], r["round"]) for r in added) == [("e6-sweep", 1), ("e6-sweep", 2), ("e6-sweep", 8)]
    # one task, round 1 fixed in round 2 -> 2.0 rounds to done (was 1.0 while each id was its own task)
    tel.main(["review", "--job", JOB, "--task", "e6-sweep", "--round", "1", "--review", "SHOULD-FIX"])
    tel.main(["review", "--job", JOB, "--task", "e6-sweep", "--round", "2", "--review", "PASS"])
    row = _row(tel.report_rows(_ledger_rows(_isolated_ledger), by="model"), MODEL_B)
    assert row is not None
    assert (row["tasks"], row["mean_rounds"], row["pass1_passes"], row["pass1_attempts"]) == (1, 2.0, 0, 1)


def test_ingest_cmd_log_auto_round_reads_a_bare_round_suffix(
    _isolated_ledger: Path, tmp_path: Path
) -> None:
    log = tmp_path / "20261017-150000-cc-e6-sweep-2-xiaomi_mimo-v2.6-pro.ndjson"
    _write_log(log, RESULT_SUCCESS + "\n")
    tel.main(
        ["ingest-cmd-log", str(log), "--job", JOB, "--agent", "cc-e6-sweep-2",
         "--task", "cc-e6-sweep-2", "--auto-round"]
    )  # fmt: skip
    (rec,) = _ledger_rows(_isolated_ledger)
    # the cmd path must land on the same task label as identity_from_name (T3), not keep "cc-"
    assert (rec["task"], rec["round"]) == ("e6-sweep", 2)
    assert rec["note"] == "round derived from name"


def test_ingest_cmd_log_auto_round_normalises_like_the_t3_path(
    _isolated_ledger: Path, tmp_path: Path
) -> None:
    # Rows for one task must join across runtimes: auto_round has to derive the SAME (task, round)
    # identity_from_name gives the T3 path — maker prefixes drop, checker roles map to review:/verify:.
    for agent, expected in (
        ("cc-e6-sweep-2", ("e6-sweep", 2)),
        ("impl-liq-t2a-3", ("liq-t2a", 3)),
        ("rev-x-3", ("review:x", 3)),
    ):
        log = tmp_path / f"20261017-150000-{agent}-xiaomi_mimo-v2.6-pro.ndjson"
        _write_log(log, RESULT_SUCCESS + "\n")
        tel.main(
            ["ingest-cmd-log", str(log), "--job", JOB, "--agent", agent,
             "--task", agent, "--auto-round"]
        )  # fmt: skip
        rec = _ledger_rows(_isolated_ledger)[-1]
        assert (rec["task"], rec["round"]) == expected
        assert (rec["task"], rec["round"]) == tel.identity_from_name(agent)[1:]
