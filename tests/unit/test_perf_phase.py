"""Tests for ``scripts/agents/perf_phase.py``'s ``--random-genomes`` mode.

The perf-6n0n9-1 table needed a 90-line scratch harness (perf-6n0n9/phase.py) for every
number; the mode must reproduce its row shape in one command. No NT backtest runs here: the
phase wrappers, the genome list and the backtest callable are all injected.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "perf_phase.py"
_spec = importlib.util.spec_from_file_location("perf_phase", _SCRIPT)
assert _spec is not None and _spec.loader is not None
perf_phase = importlib.util.module_from_spec(_spec)
sys.modules["perf_phase"] = perf_phase
_spec.loader.exec_module(perf_phase)


class _Genome:
    """Just enough of a StrategyChromosome for the per-genome indicator line."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.entry_genes: list[Any] = []
        self.exit_genes: list[Any] = []


class TestRandomModeCli:
    def test_one_command_evaluates_n_genomes_and_prints_the_table(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[str] = []

        class FakeBacktestFn:
            def __call__(self, chromosome: _Genome) -> dict[str, object]:
                calls.append(chromosome.name)
                return {}

        genomes = [_Genome("g0"), _Genome("g1")]
        monkeypatch.setattr(perf_phase, "_install_random_wrappers", lambda: None)
        monkeypatch.setattr(
            perf_phase, "random_genomes", lambda n, pool, seed, tf: genomes[:n]
        )
        monkeypatch.setattr(
            perf_phase, "make_backtest_fn", lambda sym, tf, start, end: FakeBacktestFn()
        )
        argv = [
            "--random-genomes", "2", "--pool", "RSI", "--tf", "4h",
            "--start", "2024-01-01", "--end", "2024-01-05",
        ]
        assert perf_phase.main(argv) == 0
        out = capsys.readouterr().out
        # the warm-up plus exactly the N genomes, in order
        assert calls == ["g0", "g0", "g1"]
        assert "CFG tf=4h pool=RSI win=2024-01-01..2024-01-05 n=2 " in out
        assert "s/eval" in out and "%" in out
        assert "engine sort+run(+indicators)" in out
        per_genome = json.loads(out.split("PER-GENOME ", 1)[1].splitlines()[0])
        assert [types for _seconds, types in per_genome] == [[], []]
