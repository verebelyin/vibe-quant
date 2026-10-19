import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/ledger.py"
_spec = importlib.util.spec_from_file_location("ledger", _PATH)
assert _spec and _spec.loader
ledger = importlib.util.module_from_spec(_spec)
sys.modules["ledger"] = ledger
_spec.loader.exec_module(ledger)


def test_cli_appends_rows_under_one_header_and_escapes_pipes(tmp_path: Path) -> None:
    job = tmp_path / "20261020-x"
    assert ledger.main(["--job", str(job), "--id", "cc-a-1", "--status", "dispatched", "--runtime", "t3"]) == 0
    assert ledger.main(["--job", str(job), "--id", "cc-a-1", "--status", "GATE PASS | 3/3 mutants"]) == 0
    lines = (job / "ledger.md").read_text().splitlines()
    assert lines[0] == "# ledger 20261020-x"
    assert lines[1] == "| agent | runtime | model | task | bead | worktree | status |"
    assert len(lines) == 5
    assert lines[3].startswith("| cc-a-1 | t3 | - | - | - | - | ") and lines[3].endswith(" dispatched |")
    assert lines[4].endswith(" GATE PASS \\| 3/3 mutants |")
