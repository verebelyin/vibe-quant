import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/session_stats.py"
_spec = importlib.util.spec_from_file_location("session_stats", _PATH)
assert _spec and _spec.loader
stats = importlib.util.module_from_spec(_spec)
sys.modules["session_stats"] = stats
_spec.loader.exec_module(stats)


def _line(**entry: object) -> str:
    return json.dumps(entry)


def test_counts_prompts_tools_errors_big_results_and_peak_context(tmp_path: Path) -> None:
    path = tmp_path / "s1.jsonl"
    use = {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat huge.log"}}
    use2 = {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "rm -rf $D"}}
    path.write_text("\n".join([
        _line(type="user", timestamp="2026-10-18T08:00:00Z", message={"content": "fix it"}),
        _line(type="user", timestamp="2026-10-18T08:00:01Z", message={"content": "<system-reminder>x"}),
        _line(type="assistant", timestamp="2026-10-18T08:01:00Z", message={
            "content": [use, use2],
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 400_000, "cache_creation_input_tokens": 5},
        }),
        _line(type="user", timestamp="2026-10-18T08:02:00Z", message={"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 30_000},
            {"type": "tool_result", "tool_use_id": "t2", "is_error": True,
             "content": "Claude V2 adapter has no active turn for this tool request"},
        ]}),
        "not json",
    ]) + "\n")  # fmt: skip
    st = stats.session_stats(path, 20_000)
    assert st["prompts"] == 1  # system-reminder strings are not prompts
    assert st["tools"] == {"Bash": 2}
    assert st["errors"] == 1 and st["no_active_turn"] == 1
    assert st["peak_context"] == 400_015
    assert len(st["large"]) == 1 and st["large"][0][1] == "Bash: cat huge.log"
    assert st["span"] == "2026-10-18T08:00 -> 2026-10-18T08:02"
    lines = stats.report(path, 20_000)
    assert lines[0].startswith("== s1 2026-10-18T08:00 -> 2026-10-18T08:02 prompts=1 peak_context=400,015")
