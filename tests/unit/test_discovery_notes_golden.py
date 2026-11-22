"""Default discovery runs persist byte-identical notes across the ``--gate-pool`` flag.

``--gate-pool all`` (vibe-quant-clmmc.2) is opt-in: a run that omits the flag or passes
the default ``--gate-pool top_k`` must write exactly the notes it wrote before the
flag existed. The goldens were captured at the pre-flag base (0528a91e) with the same
argv; only ``compiler_version`` (it hashes the compiler source) and the scratch directory
are masked.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run

if TYPE_CHECKING:
    from vibe_quant.discovery.run import BacktestAdapter

GOLDENS = Path(__file__).parent / "goldens"
REPO = Path(__file__).resolve().parents[2]
BTC_4H = REPO / "data" / "catalog" / "data" / "bar" / "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"

MOCK_ARGV = [
    "--mock", "--seed", "7", "--population-size", "6", "--max-generations", "2",
    "--max-workers", "-1", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
    "--timeframe", "4h", "--symbols", "BTCUSDT",
]  # fmt: skip
MOCK_CASES = {
    "champion": ["--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate"],
    "empty": [],
    "multiseed": ["--num-seeds", "2"],
}

PORT_DOC = {
    "name": "golden_ema_cross",
    "params": {
        "fast": {"kind": "int", "low": 5, "high": 20, "step": 1, "default": 9},
        "slow": {"kind": "int", "low": 30, "high": 60, "step": 5, "default": 40},
    },
    "constraints": ["fast < slow"],
    "dsl": {
        "name": "golden_ema_cross",
        "timeframe": "4h",
        "indicators": {
            "fast_ema": {"type": "EMA", "period": "{{fast}}"},
            "slow_ema": {"type": "EMA", "period": "{{slow}}"},
        },
        "entry_conditions": {"long": ["fast_ema crosses_above slow_ema"], "short": []},
        "exit_conditions": {"long": ["fast_ema crosses_below slow_ema"], "short": []},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 6.0},
    },
}


def _masked(notes: str) -> str:
    return re.sub(r'"compiler_version": "[0-9a-f]+"', '"compiler_version": "<masked>"', notes)


def notes_bytes(tmp_path: Path, argv: list[str], backtests: BacktestAdapter) -> str:
    """Run ``run_discovery`` off ``argv`` on a scratch DB; return the persisted notes text."""
    from tests.unit.test_run_discovery import drive_run_discovery

    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2025-01-01", {}
    )
    state.close()
    outcome = drive_run_discovery(
        db, run_id, [*argv, "--db", str(db), "--run-id", str(run_id)], backtests
    )
    assert outcome.exit_code == 0, outcome.message
    state = StateManager(db)
    try:
        result = state.get_backtest_result(run_id)
    finally:
        state.close()
    assert result is not None
    return _masked(str(result["notes"])).replace(str(tmp_path), "<tmp>")


def port_argv(folder: Path) -> list[str]:
    return [
        "--seed", "3", "--population-size", "4", "--max-generations", "1",
        "--elite-count", "1", "--max-workers", "-1", "--symbols", "BTCUSDT",
        "--timeframe", "4h", "--start-date", "2024-01-01", "--end-date", "2024-04-01",
        "--port", str(folder),
    ]  # fmt: skip


def write_port(tmp_path: Path) -> Path:
    import yaml

    folder = tmp_path / "port"
    folder.mkdir()
    (folder / "port.yaml").write_text(yaml.safe_dump(PORT_DOC, sort_keys=False), encoding="utf-8")
    return folder


def _golden(name: str) -> str:
    return (GOLDENS / f"discovery_notes_{name}.json").read_text(encoding="utf-8").rstrip("\n")


@pytest.mark.parametrize("case", sorted(MOCK_CASES))
@pytest.mark.parametrize("flag", [[], ["--gate-pool", "top_k"]], ids=["omitted", "explicit"])
def test_default_bytes(tmp_path: Path, case: str, flag: list[str]) -> None:
    notes = notes_bytes(tmp_path, [*MOCK_ARGV, *MOCK_CASES[case], *flag], run.MockBacktests())

    assert notes == _golden(case)
    assert '"gate_pool"' not in notes and '"rejection_counts"' not in notes


@pytest.mark.slow(reason="a real NT port discovery on BTCUSDT 4h, twice")
@pytest.mark.skipif(not BTC_4H.is_dir(), reason="needs the BTCUSDT 4h catalog")
@pytest.mark.parametrize("flag", [[], ["--gate-pool", "top_k"]], ids=["omitted", "explicit"])
def test_default_bytes_nt_port(tmp_path: Path, flag: list[str]) -> None:
    notes = notes_bytes(tmp_path, [*port_argv(write_port(tmp_path)), *flag], run.NTBacktests())

    assert notes == _golden("nt_port")
