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
    for extra in (["--runtime", "codex"], ["--outcome", "maybe"], ["--review", "MAYBE"]):
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
