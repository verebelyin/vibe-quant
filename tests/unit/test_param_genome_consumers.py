"""A ParamGenome (port params) works in discovery's gates, seeding, records and pooling.

Every consumer of genome fields goes through a type-dispatched accessor. These tests
drive the public entry points with two unrelated ports and assert literal values; the
legacy RNG golden (test_discovery_legacy_rng_golden.py) proves the legacy path is
bit-identical.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.unit.test_gate_chain_contract import (
    _bh_closes,
    _cfg,
    _chrom,
    _fit,
    _nt_fn,
    _pooled,
    _SeedOut,
    _series,
)
from vibe_quant.discovery.gate_chain import GateChain, GateEvidence
from vibe_quant.discovery.operators import Direction
from vibe_quant.discovery.param_genome import ParamGenome
from vibe_quant.discovery.port import Port, load_port

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Two unrelated ports (+ a third rendering both sides)
# ---------------------------------------------------------------------------


def _write_port(tmp_path: Path, doc: dict[str, object]) -> Port:
    folder = tmp_path / str(doc["name"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "port.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return load_port(path)


def _long_port(tmp_path: Path, *, sl: float = 2.0, tp: float = 6.0) -> Port:
    """SMA cross, long only; two free params."""
    return _write_port(
        tmp_path,
        {
            "name": "ma_long",
            "params": {
                "fast": {"kind": "int", "low": 5, "high": 20, "step": 1, "default": 10},
                "slow": {"kind": "int", "low": 30, "high": 60, "step": 5, "default": 40},
            },
            "constraints": ["fast < slow"],
            "dsl": {
                "name": "ma_long",
                "timeframe": "1h",
                "indicators": {
                    "fast_ma": {"type": "SMA", "period": "{{fast}}"},
                    "slow_ma": {"type": "SMA", "period": "{{slow}}"},
                },
                "entry_conditions": {"long": ["fast_ma crosses_above slow_ma"], "short": []},
                "exit_conditions": {"long": ["fast_ma crosses_below slow_ma"], "short": []},
                "stop_loss": {"type": "fixed_pct", "percent": sl},
                "take_profit": {"type": "fixed_pct", "percent": tp},
            },
        },
    )


def _short_port(tmp_path: Path, *, sl: float = 40.0, tp: float = 5.0) -> Port:
    """RSI fade, short only; three free params and one fixed one."""
    return _write_port(
        tmp_path,
        {
            "name": "rsi_short",
            "params": {
                "period": {"kind": "int", "low": 7, "high": 21, "step": 1, "default": 14},
                "upper": {"kind": "float", "low": 60.0, "high": 80.0, "step": 5.0, "default": 70.0},
                "lower": {"kind": "float", "low": 20.0, "high": 40.0, "step": 5.0, "default": 30.0},
                "pinned": {"kind": "int", "low": 3, "high": 3, "step": 1, "default": 3},
            },
            "constraints": [],
            "dsl": {
                "name": "rsi_short",
                "timeframe": "1h",
                "indicators": {
                    "rsi": {"type": "RSI", "period": "{{period}}"},
                    "base": {"type": "SMA", "period": "{{pinned}}"},
                },
                "entry_conditions": {"long": [], "short": ["rsi > {{upper}}"]},
                "exit_conditions": {"long": [], "short": ["rsi < {{lower}}"]},
                "stop_loss": {"type": "fixed_pct", "percent": sl},
                "take_profit": {"type": "fixed_pct", "percent": tp},
            },
        },
    )


def _both_port(tmp_path: Path) -> Port:
    """RSI both ways with a worse short side (SL 40 / TP 5 vs the generic 2 / 4)."""
    return _write_port(
        tmp_path,
        {
            "name": "rsi_both",
            "params": {
                "period": {"kind": "int", "low": 7, "high": 21, "step": 1, "default": 14},
            },
            "constraints": [],
            "dsl": {
                "name": "rsi_both",
                "timeframe": "1h",
                "indicators": {"rsi": {"type": "RSI", "period": "{{period}}"}},
                "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
                "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
                "stop_loss": {"type": "fixed_pct", "percent": 2.0},
                "take_profit": {"type": "fixed_pct", "percent": 4.0},
                "stop_loss_short": {"type": "fixed_pct", "percent": 40.0},
                "take_profit_short": {"type": "fixed_pct", "percent": 5.0},
            },
        },
    )


def _genome(port: Port, **changes: Any) -> ParamGenome:
    values = port.defaults()
    values.update(changes)
    return ParamGenome(port=port, values=values)


# ---------------------------------------------------------------------------
# R30: direction comes from the rendered DSL
# ---------------------------------------------------------------------------


def _run_buy_hold(chrom: Any, sharpe_ann: float) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    closes = _bh_closes()
    gates = GateChain(
        _cfg(),
        GateEvidence(_nt_fn(), True, buy_hold_closes_fn=lambda _symbols: closes),
    )
    fit = _fit(sharpe_ratio=2.0, mtm_daily_returns=_series(sharpe_ann, seed=7))
    kept = gates.apply_buy_hold([(chrom, fit)])
    return kept, gates.buy_hold_records[chrom.uid], list(gates.rejections)


def test_buy_hold_reads_rendered_direction(tmp_path: Path) -> None:
    """A ParamGenome champion passes or fails like a legacy one of the same direction."""
    long_g = _genome(_long_port(tmp_path))
    short_g = _genome(_short_port(tmp_path))

    # Market B&H sharpe_ann is 1.1155295831428362: a long bar of +1.1155 is not beaten
    # by a 0.5 candidate, a short bar of -1.1155 is.
    kept_long, rec_long, rej_long = _run_buy_hold(long_g, 0.5)
    assert kept_long == []
    assert rec_long["direction"] == "long"
    assert rec_long["bar_ann"] == 1.1155295831428362
    assert rej_long[0]["stage"] == "buy_hold"
    assert rej_long[0]["uid"] == long_g.uid

    kept_short, rec_short, rej_short = _run_buy_hold(short_g, 0.5)
    assert [g.uid for g, _ in kept_short] == [short_g.uid]
    assert rec_short["direction"] == "short"
    assert rec_short["bar_ann"] == -1.1155295831428362
    assert rej_short == []

    legacy_kept, legacy_rec, _ = _run_buy_hold(_chrom(direction=Direction.SHORT), 0.5)
    assert len(legacy_kept) == 1
    assert legacy_rec["direction"] == "short"
    assert legacy_rec["bar_ann"] == rec_short["bar_ann"]
    assert legacy_rec["sharpe_ann"] == rec_short["sharpe_ann"]


def test_buy_hold_both_sides_uses_absolute_bar(tmp_path: Path) -> None:
    both_g = _genome(_both_port(tmp_path))
    kept, record, _ = _run_buy_hold(both_g, 0.5)
    assert kept == []
    assert record["direction"] == "both"
    assert record["bar_ann"] == 1.1155295831428362


def test_rejected_param_champion_keeps_its_rendered_dsl(tmp_path: Path) -> None:
    long_g = _genome(_long_port(tmp_path), fast=8, slow=45)
    _, _, rejections = _run_buy_hold(long_g, 0.5)
    dsl = rejections[0]["dsl"]
    assert dsl["name"] == f"ma_long_{long_g.uid}"
    assert dsl["indicators"]["fast_ma"]["period"] == 8
    assert dsl["indicators"]["slow_ma"]["period"] == 45


def test_accessors_read_kind_direction_and_summary(tmp_path: Path) -> None:
    from vibe_quant.discovery.genome import genome_direction, genome_kind, genome_summary

    assert genome_kind(_genome(_long_port(tmp_path))) == "param"
    assert genome_kind(_chrom()) == "legacy"
    assert genome_direction(_genome(_long_port(tmp_path))) == "long"
    assert genome_direction(_genome(_short_port(tmp_path))) == "short"
    assert genome_direction(_genome(_both_port(tmp_path))) == "both"
    assert genome_direction(_chrom(direction=Direction.SHORT)) == "short"
    assert genome_direction(_chrom(direction=Direction.LONG)) == "long"
    assert genome_summary(_genome(_short_port(tmp_path), period=9, upper=75.0)) == (
        "port=rsi_short period=9 upper=75.0 lower=30.0 pinned=3"
    )
    assert genome_summary(_chrom(direction=Direction.LONG)) == (
        "dir=long entry=['RSI'] exit=['RSI'] sl=2.0% tp=4.0%"
    )
    with pytest.raises(TypeError):
        genome_kind("not a genome")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# R31 / R32: warm-start seeds
# ---------------------------------------------------------------------------


class _FakeState:
    def __init__(self, strategies: list[dict[str, object]]) -> None:
        self._notes = json.dumps({"top_strategies": [{"chromosome": s} for s in strategies]})

    def get_backtest_result(self, _run_id: int) -> dict[str, object]:
        return {"notes": self._notes}


def _legacy_both_seed() -> dict[str, object]:
    """A pre-inydc direction=both seed: no short_* keys at all."""
    from vibe_quant.discovery.genome import chromosome_to_serializable

    return chromosome_to_serializable(_chrom(direction=Direction.LONG)) | {"direction": "both"}


def test_param_seeds_load_untouched_and_legacy_both_still_repaired(tmp_path: Path) -> None:
    from vibe_quant.discovery.seed import load_seed_chromosomes

    port = _both_port(tmp_path)
    seeds = [_genome(port, period=9), _genome(port, period=12)]
    loaded = load_seed_chromosomes(
        _FakeState([s.to_serializable() for s in seeds]),  # type: ignore[arg-type]
        1,
        port=port,
    )
    assert loaded is not None
    assert all(isinstance(g, ParamGenome) for g in loaded)
    assert [(g.uid, g.values) for g in loaded] == [(s.uid, s.values) for s in seeds]  # type: ignore[union-attr]
    assert [genome.port.name for genome in loaded] == ["rsi_both", "rsi_both"]  # type: ignore[union-attr]

    legacy = load_seed_chromosomes(_FakeState([_legacy_both_seed()]), 1)  # type: ignore[arg-type]
    assert legacy is not None
    assert legacy[0].direction == Direction.LONG  # type: ignore[union-attr]


def test_warm_start_mixing_legacy_and_param_seeds_is_refused(tmp_path: Path) -> None:
    from vibe_quant.discovery.genome import SeedIncompatibleError, chromosome_to_serializable
    from vibe_quant.discovery.seed import load_seed_chromosomes

    port = _long_port(tmp_path)
    mixed = [_genome(port).to_serializable(), chromosome_to_serializable(_chrom())]
    with pytest.raises(SeedIncompatibleError, match="legacy"):
        load_seed_chromosomes(_FakeState(mixed), 1, port=port)  # type: ignore[arg-type]
    with pytest.raises(SeedIncompatibleError, match="legacy"):  # legacy first, then param
        load_seed_chromosomes(_FakeState(mixed[::-1]), 1, port=port)  # type: ignore[arg-type]


def test_warm_start_param_seeds_without_a_port_are_refused(tmp_path: Path) -> None:
    from vibe_quant.discovery.genome import SeedIncompatibleError
    from vibe_quant.discovery.seed import load_seed_chromosomes

    stored = [_genome(_short_port(tmp_path)).to_serializable()]
    with pytest.raises(SeedIncompatibleError, match="--port"):
        load_seed_chromosomes(_FakeState(stored), 1)  # type: ignore[arg-type]


def test_warm_start_from_another_port_is_refused(tmp_path: Path) -> None:
    from vibe_quant.discovery.genome import SeedIncompatibleError
    from vibe_quant.discovery.seed import load_seed_chromosomes

    stored = [_genome(_long_port(tmp_path)).to_serializable()]
    with pytest.raises(SeedIncompatibleError, match="ma_long"):
        load_seed_chromosomes(_FakeState(stored), 1, port=_short_port(tmp_path))  # type: ignore[arg-type]


def test_warm_start_after_the_port_file_changed_is_refused(tmp_path: Path) -> None:
    from vibe_quant.discovery.genome import SeedIncompatibleError
    from vibe_quant.discovery.seed import load_seed_chromosomes

    original = _long_port(tmp_path)
    stored = [_genome(original, fast=8).to_serializable()]
    edited = _long_port(tmp_path, tp=7.0)  # same name, different bytes
    assert edited.sha256 != original.sha256
    with pytest.raises(SeedIncompatibleError, match="revision"):
        load_seed_chromosomes(_FakeState(stored), 1, port=edited)  # type: ignore[arg-type]


def test_corrupt_legacy_seed_still_degrades_to_none() -> None:
    from vibe_quant.discovery.seed import load_seed_chromosomes

    bad = _legacy_both_seed() | {"condition": 1}
    bad["entry_genes"] = [{"indicator_type": "RSI", "condition": "nope", "threshold": 1.0}]
    assert load_seed_chromosomes(_FakeState([bad]), 1) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# R33: records and pooling
# ---------------------------------------------------------------------------


def test_champion_entry_carries_rendered_dsl_and_own_values(tmp_path: Path) -> None:
    from vibe_quant.discovery.record import champion_entry

    for port, changes in ((_long_port(tmp_path), {"fast": 7, "slow": 55}), (_short_port(tmp_path), {"upper": 65.0})):
        genome = _genome(port, **changes)
        entry = champion_entry(genome, _fit(), timeframe="4h", dsr=None, buy_hold=None)
        expected_dsl = genome.to_dsl()
        expected_dsl["timeframe"] = "4h"
        assert entry["dsl"] == expected_dsl
        assert entry["chromosome"] == genome.to_serializable()
        assert entry["chromosome"]["values"] == genome.values  # type: ignore[index]
        assert entry["chromosome"]["kind"] == "param"  # type: ignore[index]


def test_pooling_groups_param_champions_by_param_distance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port = _long_port(tmp_path)
    cfg = _cfg(top_k=3, min_diversity_distance=0.2)
    strong = _series(8.0, seed=7)
    near_a = _genome(port, fast=10, slow=40)
    near_b = _genome(port, fast=10, slow=45)  # slow differs by 5/30 of 2 params: 1/12 < 0.2
    far = _genome(port, fast=20, slow=60)  # (10/15 + 20/30)/2 = 2/3 > 0.2
    fits = {g.uid: _fit(sharpe_ratio=s, adjusted_score=s, mtm_daily_returns=strong)
            for g, s in ((near_a, 2.0), (near_b, 3.0), (far, 1.0))}
    own = {g.uid: {"enforced": True} for g in (near_a, near_b, far)}
    seeds = [
        _SeedOut((near_a, fits[near_a.uid]), cache={near_a.uid: fits[near_a.uid]}, buy_hold={near_a.uid: own[near_a.uid]}),
        _SeedOut((near_b, fits[near_b.uid]), cache={near_b.uid: fits[near_b.uid]}, buy_hold={near_b.uid: own[near_b.uid]}),
        _SeedOut((far, fits[far.uid]), cache={far.uid: fits[far.uid]}, buy_hold={far.uid: own[far.uid]}),
    ]

    def _no_bh(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("pooled re-check must not run buy-and-hold")

    monkeypatch.setattr("vibe_quant.discovery.buyhold.load_daily_closes", _no_bh)
    merged = _pooled(monkeypatch, cfg, _nt_fn(), seeds)

    kept = [g for g, _ in merged.top_strategies]
    assert all(isinstance(g, ParamGenome) for g in kept)
    # near_a and near_b form one group (best representative near_b); far is its own group.
    assert {g.uid for g in kept} == {near_b.uid, far.uid}
    assert [g.values for g in kept if g.uid == near_b.uid] == [{"fast": 10, "slow": 45}]


# ---------------------------------------------------------------------------
# R37: SL/TP penalty on the rendered DSL; complexity = free params
# ---------------------------------------------------------------------------


def test_sl_tp_penalty_reads_the_rendered_dsl(tmp_path: Path) -> None:
    from vibe_quant.discovery.fitness import chromosome_sl_tp_penalty

    assert chromosome_sl_tp_penalty(_genome(_long_port(tmp_path))) == 0.0  # 2 / 6
    assert chromosome_sl_tp_penalty(_genome(_long_port(tmp_path / "x", sl=30.0, tp=5.0))) == pytest.approx(0.02)
    # short only: SL 40 / TP 5 = 8, (8 - 5) * 0.02
    assert chromosome_sl_tp_penalty(_genome(_short_port(tmp_path))) == pytest.approx(0.06)
    # both sides: the worse (short, 8 -> 0.06) side counts; the long side falls back to 2 / 4
    assert chromosome_sl_tp_penalty(_genome(_both_port(tmp_path))) == pytest.approx(0.06)
    # legacy unchanged
    assert chromosome_sl_tp_penalty(_chrom(stop_loss_pct=8.0, take_profit_pct=1.0)) == pytest.approx(0.06)


def test_sl_tp_penalty_is_zero_without_two_fixed_pct_levels(tmp_path: Path) -> None:
    from vibe_quant.discovery.fitness import chromosome_sl_tp_penalty

    port = _write_port(
        tmp_path,
        {
            "name": "atr_stop",
            "params": {"period": {"kind": "int", "low": 7, "high": 21, "step": 1, "default": 14}},
            "constraints": [],
            "dsl": {
                "name": "atr_stop",
                "timeframe": "1h",
                "indicators": {"rsi": {"type": "RSI", "period": "{{period}}"}, "atr": {"type": "ATR", "period": 14}},
                "entry_conditions": {"long": ["rsi < 30"], "short": []},
                "exit_conditions": {"long": ["rsi > 50"], "short": []},
                "stop_loss": {"type": "atr_fixed", "atr_multiplier": 9.0, "indicator": "atr"},
                "take_profit": {"type": "fixed_pct", "percent": 1.0},
            },
        },
    )
    assert chromosome_sl_tp_penalty(_genome(port)) == 0.0


def test_complexity_counts_free_params_only(tmp_path: Path) -> None:
    from vibe_quant.discovery.operators import scalar_gene_count

    assert scalar_gene_count(_genome(_long_port(tmp_path))) == 2
    assert scalar_gene_count(_genome(_short_port(tmp_path))) == 3  # `pinned` is fixed
    assert scalar_gene_count(_chrom()) == 2


def test_param_genome_scores_through_evaluate_population(tmp_path: Path) -> None:
    """A backtest of a ParamGenome gets a real score, not a zero-fitness error."""
    from vibe_quant.discovery.fitness import evaluate_population

    def backtest(_chrom: Any) -> dict[str, float | int]:
        return {"sharpe_ratio": 2.0, "max_drawdown": 0.1, "profit_factor": 1.5,
                "total_trades": 200, "total_return": 0.3}

    genome = _genome(_short_port(tmp_path))
    [result] = evaluate_population([genome], backtest, None, max_workers=None, min_trades=50, timeframe="1h")  # type: ignore[arg-type]
    assert result.error is None
    assert result.adjusted_score > 0
    assert result.sl_tp_penalty == pytest.approx(0.06)
    assert result.complexity_penalty == pytest.approx(0.02)  # 3 free params, 2 free


# ---------------------------------------------------------------------------
# Serialization and distance dispatch
# ---------------------------------------------------------------------------


def test_serializable_dispatches_on_kind(tmp_path: Path) -> None:
    from vibe_quant.discovery.genome import (
        chromosome_to_dsl,
        chromosome_to_serializable,
        serializable_to_chromosome,
        serializable_to_genome,
    )

    port = _short_port(tmp_path)
    genome = _genome(port, period=11)
    stored = chromosome_to_serializable(genome)
    assert stored == genome.to_serializable()
    assert chromosome_to_dsl(genome) == genome.to_dsl()

    back = serializable_to_genome(stored, port)
    assert isinstance(back, ParamGenome)
    assert (back.uid, back.values) == (genome.uid, genome.values)

    legacy = _chrom(direction=Direction.SHORT)
    legacy_back = serializable_to_genome(chromosome_to_serializable(legacy), None)
    assert not isinstance(legacy_back, ParamGenome)
    assert chromosome_to_serializable(legacy_back) == chromosome_to_serializable(legacy)

    with pytest.raises(ValueError, match="Port"):
        serializable_to_genome(stored, None)
    with pytest.raises(ValueError, match="param"):
        serializable_to_chromosome(stored)


def test_distance_between_param_genomes_and_never_across_kinds(tmp_path: Path) -> None:
    from vibe_quant.discovery.distance import chromosome_distance

    port = _long_port(tmp_path)
    a = _genome(port, fast=10, slow=40)
    b = _genome(port, fast=20, slow=60)
    assert chromosome_distance(a, a.clone()) == 0.0
    assert chromosome_distance(a, b) == pytest.approx((10 / 15 + 20 / 30) / 2)
    with pytest.raises(TypeError):
        chromosome_distance(a, _chrom())
    with pytest.raises(TypeError):
        chromosome_distance(_chrom(), a)
    with pytest.raises(ValueError, match="different ports"):
        chromosome_distance(a, _genome(_short_port(tmp_path)))
