"""Tests for ``scripts/agents/perf_phase.py``'s ``--random-genomes`` mode (bead 93q2v).

The perf-6n0n9-1 table needed a 90-line scratch harness (perf-6n0n9/phase.py) for every
number; the mode must reproduce its row shape in one command. No NT backtest runs here: the
phase wrappers, the genome list and the backtest callable are all injected.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "perf_phase.py"
_spec = importlib.util.spec_from_file_location("perf_phase", _SCRIPT)
assert _spec is not None and _spec.loader is not None
perf_phase = importlib.util.module_from_spec(_spec)
sys.modules["perf_phase"] = perf_phase
_spec.loader.exec_module(perf_phase)


class TestSplitWindows:
    """phase.py evaluated each genome on two sub-windows split at the midpoint."""

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            ("2024-01-01", "2024-01-05",
             [("2024-01-01", "2024-01-03"), ("2024-01-03", "2024-01-05")]),
            ("2025-03-01", "2025-03-29",
             [("2025-03-01", "2025-03-15"), ("2025-03-15", "2025-03-29")]),
        ],
    )
    def test_midpoint_is_literal(
        self, start: str, end: str, expected: list[tuple[str, str]]
    ) -> None:
        assert perf_phase.split_windows(start, end) == expected


class _CapturingPipeline:
    """Stands in for DiscoveryPipeline: records the config the pool filter received."""

    seen: list[Any] = []

    def __init__(self, cfg: Any, _fn: Any) -> None:
        self.cfg = cfg

    def _apply_indicator_pool_filter(self) -> None:
        _CapturingPipeline.seen.append(self.cfg)


class TestPool:
    def test_pool_is_forwarded_to_the_discovery_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two unrelated inputs: a comma list and ``full`` (None = no filter)."""
        import vibe_quant.discovery.pipeline as pipeline_mod

        _CapturingPipeline.seen = []
        monkeypatch.setattr(pipeline_mod, "DiscoveryPipeline", _CapturingPipeline)
        assert perf_phase.apply_pool("RSI,MFI", "4h") == ["RSI", "MFI"]
        assert perf_phase.apply_pool("full", "4h") is None
        assert [c.indicator_pool for c in _CapturingPipeline.seen] == [
            ["RSI", "MFI"], None
        ]
        assert [c.timeframe for c in _CapturingPipeline.seen] == ["4h", "4h"]

    def test_blank_items_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import vibe_quant.discovery.pipeline as pipeline_mod

        _CapturingPipeline.seen = []
        monkeypatch.setattr(pipeline_mod, "DiscoveryPipeline", _CapturingPipeline)
        assert perf_phase.apply_pool("RSI,,MFI,", "1d") == ["RSI", "MFI"]
        assert _CapturingPipeline.seen[0].indicator_pool == ["RSI", "MFI"]
        assert _CapturingPipeline.seen[0].timeframe == "1d"


class TestRandomGenomes:
    def test_same_seed_repeats_and_the_direction_is_long(self) -> None:
        from vibe_quant.discovery.genome import chromosome_to_serializable

        def serialized(seed: int) -> list[dict[str, object]]:
            return [
                chromosome_to_serializable(c)
                for c in perf_phase.random_genomes(4, "full", seed, "4h")
            ]

        first, second = serialized(11), serialized(11)
        assert first == second
        assert [c["direction"] for c in first] == ["long"] * 4
        assert serialized(12) != first  # the seed really drives the draw


class TestPhaseShareLines:
    """The table phase.py printed: per-phase s/eval + share, the engine row, the glue row."""

    def test_rows_are_literal(self) -> None:
        times = {
            "compile": 0.10,
            "node.build": 0.20,
            "data:catalog_query": 0.30,
            "data:add_data": 0.10,
            "run_oneshot(incl data)": 0.90,
            "extract_metrics": 0.10,
            "dispose": 0.10,
            "require_bars": 0.20,
        }
        assert perf_phase.phase_share_lines(times, 2.0, 2) == [
            "  compile                      0.050 s/eval    5.0%",
            "  node.build                   0.100 s/eval   10.0%",
            "  data:catalog_query           0.150 s/eval   15.0%",
            "  data:add_data                0.050 s/eval    5.0%",
            "  run_oneshot(incl data)       0.450 s/eval   45.0%",
            "  extract_metrics              0.050 s/eval    5.0%",
            "  dispose                      0.050 s/eval    5.0%",
            "  require_bars                 0.100 s/eval   10.0%",
            "  engine sort+run(+indicators)   0.250 s/eval   25.0%",
            "  other (glue,chromosome->dsl)   0.300 s/eval   30.0%",
        ]

    def test_engine_row_subtracts_the_data_phases_and_glue_is_the_remainder(self) -> None:
        times = {
            "compile": 0.0,
            "node.build": 0.0,
            "data:catalog_query": 0.4,
            "data:add_data": 0.2,
            "run_oneshot(incl data)": 1.0,
            "extract_metrics": 0.0,
            "dispose": 0.0,
            "require_bars": 0.0,
        }
        lines = perf_phase.phase_share_lines(times, 4.0, 4)
        assert lines[-2] == "  engine sort+run(+indicators)   0.100 s/eval   10.0%"
        assert lines[-1] == "  other (glue,chromosome->dsl)   0.750 s/eval   75.0%"


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

    def test_random_mode_without_tf_is_a_prerequisite_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(perf_phase, "_install_random_wrappers", lambda: None)
        assert perf_phase.main(["--random-genomes", "1"]) == 2
        assert "--tf" in capsys.readouterr().err

    def test_single_eval_mode_still_needs_a_strategy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(perf_phase, "_install_wrappers", lambda: None)
        with pytest.raises(SystemExit, match="either --sid or --dsl is required"):
            perf_phase.main([])
