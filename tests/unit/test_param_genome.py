"""Tests for parameter genomes and the port GA operators."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
import yaml

from vibe_quant.discovery.param_genome import (
    ParamGenome,
    ParamOps,
    from_serializable,
    param_distance,
)
from vibe_quant.discovery.port import ParamValue, Port, PortError, load_port

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


_SIMPLE_DSL: dict[str, object] = {
    "name": "simple",
    "timeframe": "1h",
    "description": "a={{a}} b={{b}} c={{c}}",
    "indicators": {"base": {"type": "SMA", "period": "{{d}}"}},
    "entry_conditions": {"long": ["close > base"], "short": []},
    "exit_conditions": {"long": ["close < base"], "short": []},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 5.0},
}


def _simple_doc() -> dict[str, object]:
    return {
        "name": "simple_port",
        "params": {
            "a": {"kind": "int", "low": 0, "high": 10, "step": 1, "default": 5},
            "b": {"kind": "float", "low": 0.0, "high": 1.0, "step": 0.1, "default": 0.5},
            "c": {"kind": "choice", "choices": ["x", "y"], "default": "x"},
            "d": {"kind": "int", "low": 7, "high": 7, "step": 1, "default": 7},
        },
        "constraints": [],
        "dsl": yaml.safe_load(yaml.safe_dump(_SIMPLE_DSL)),
    }


def _repair_doc() -> dict[str, object]:
    return {
        "name": "repair_port",
        "params": {
            "a": {"kind": "int", "low": 1, "high": 2, "step": 1, "default": 2},
            "b": {"kind": "int", "low": 1, "high": 2, "step": 1, "default": 1},
        },
        "constraints": ["a > b"],
        "dsl": {
            "name": "repair",
            "timeframe": "1h",
            "description": "a={{a}} b={{b}}",
            "indicators": {"base": {"type": "SMA", "period": 10}},
            "entry_conditions": {"long": ["close > base"], "short": []},
            "exit_conditions": {"long": ["close < base"], "short": []},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 5.0},
        },
    }


def _load(tmp_path: Path, doc: dict[str, object]) -> Port:
    folder = tmp_path / str(doc["name"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "port.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return load_port(folder)


def _simple(tmp_path: Path) -> Port:
    return _load(tmp_path, _simple_doc())


def _repair(tmp_path: Path) -> Port:
    return _load(tmp_path, _repair_doc())


def test_param_ops_rejects_zero_free_port(tmp_path: Path) -> None:
    doc = _simple_doc()
    params = doc["params"]
    assert isinstance(params, dict)
    params["a"] = {"kind": "int", "low": 5, "high": 5, "step": 1, "default": 5}
    params["b"] = {"kind": "float", "low": 0.5, "high": 0.5, "step": 0.1, "default": 0.5}
    params["c"] = {"kind": "choice", "choices": ["x"], "default": "x"}
    port = _load(tmp_path, doc)
    assert port.free_params() == ()
    with pytest.raises(PortError, match="no free parameters"):
        ParamOps(port)


def test_random_stays_in_domain_and_pins_fixed_params(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    random.seed(1)
    genome = ops.random()
    port.check_values(genome.values)
    assert genome.values["d"] == 7
    assert ops.is_valid(genome)


def test_clone_keeps_uid_and_copies_values(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    genome = ParamGenome(port=port, values=port.defaults())
    clone = genome.clone()
    assert clone.uid == genome.uid
    assert clone.values == genome.values
    assert clone.values is not genome.values


def test_mutate_returns_fresh_uid(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    genome = ParamGenome(port=port, values=port.defaults())
    random.seed(2)
    mutated = ops.mutate(genome, 1.0)
    assert mutated.uid != genome.uid
    assert ops.is_valid(mutated)


def test_crossover_children_take_each_param_from_one_parent(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    parent_a = ParamGenome(port=port, values={"a": 0, "b": 0.0, "c": "x", "d": 7})
    parent_b = ParamGenome(port=port, values={"a": 10, "b": 1.0, "c": "y", "d": 7})
    random.seed(3)
    child_a, child_b = ops.crossover(parent_a, parent_b)
    assert child_a.uid not in {parent_a.uid, parent_b.uid}
    assert child_b.uid not in {parent_a.uid, parent_b.uid}
    assert child_a.uid != child_b.uid
    for name in ("a", "b", "c"):
        assert {child_a.values[name], child_b.values[name]} == {
            parent_a.values[name],
            parent_b.values[name],
        }
    assert child_a.values["d"] == child_b.values["d"] == 7


def test_crossover_rejects_different_ports(tmp_path: Path) -> None:
    simple = _simple(tmp_path)
    repair = _repair(tmp_path)
    ops = ParamOps(simple)
    parent_a = ParamGenome(port=simple, values=simple.defaults())
    parent_b = ParamGenome(port=repair, values=repair.defaults())
    with pytest.raises(PortError, match="different ports"):
        ops.crossover(parent_a, parent_b)


def test_is_valid_checks_domain_and_constraints(tmp_path: Path) -> None:
    port = _repair(tmp_path)
    ops = ParamOps(port)
    assert ops.is_valid(ParamGenome(port=port, values={"a": 2, "b": 1}))
    assert not ops.is_valid(ParamGenome(port=port, values={"a": 1, "b": 2}))
    assert not ops.is_valid(ParamGenome(port=port, values={"a": 3, "b": 1}))
    assert not ops.is_valid(ParamGenome(port=port, values={"a": 2}))


def test_repair_returns_input_when_valid(tmp_path: Path) -> None:
    port = _repair(tmp_path)
    ops = ParamOps(port)
    genome = ParamGenome(port=port, values={"a": 2, "b": 1})
    assert ops.repair(genome) is genome


def test_repair_mutation_phase_finds_the_only_valid_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _repair(tmp_path)
    ops = ParamOps(port)
    genome = ParamGenome(port=port, values={"a": 1, "b": 2})
    signs = iter([1, -1])
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda low, high: 1)
    monkeypatch.setattr(random, "choice", lambda seq: next(signs))
    repaired = ops.repair(genome)
    assert repaired.values == {"a": 2, "b": 1}
    assert repaired.uid != genome.uid


def test_repair_fails_after_bounded_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _repair(tmp_path)
    ops = ParamOps(port)
    genome = ParamGenome(port=port, values={"a": 1, "b": 2})
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda low, high: 1)
    monkeypatch.setattr(random, "choice", lambda seq: 1)
    monkeypatch.setattr(random, "randrange", lambda *args: 0)
    with pytest.raises(PortError, match="repair failed"):
        ops.repair(genome)


def test_serialization_round_trip(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    genome = ParamGenome(port=port, values={"a": 3, "b": 0.7, "c": "y", "d": 7})
    payload = genome.to_serializable()
    assert payload["kind"] == "param"
    restored = from_serializable(payload, port)
    assert restored.uid == genome.uid
    assert restored.values == genome.values
    assert restored.to_dsl() == genome.to_dsl()


def _corrupt_kind(payload: dict[str, object]) -> None:
    payload["kind"] = "legacy"


def _corrupt_name(payload: dict[str, object]) -> None:
    meta = payload["port"]
    assert isinstance(meta, dict)
    meta["name"] = "other_port"


def _corrupt_sha(payload: dict[str, object]) -> None:
    meta = payload["port"]
    assert isinstance(meta, dict)
    meta["sha256"] = "0" * 64


def _corrupt_missing_value(payload: dict[str, object]) -> None:
    values = payload["values"]
    assert isinstance(values, dict)
    del values["a"]


def _corrupt_extra_value(payload: dict[str, object]) -> None:
    values = payload["values"]
    assert isinstance(values, dict)
    values["bogus"] = 1


def _corrupt_wrong_type(payload: dict[str, object]) -> None:
    values = payload["values"]
    assert isinstance(values, dict)
    values["a"] = 1.5


def _corrupt_out_of_domain(payload: dict[str, object]) -> None:
    values = payload["values"]
    assert isinstance(values, dict)
    values["a"] = 99


def _corrupt_uid(payload: dict[str, object]) -> None:
    del payload["uid"]


@pytest.mark.parametrize(
    "corrupt",
    [
        _corrupt_kind,
        _corrupt_name,
        _corrupt_sha,
        _corrupt_missing_value,
        _corrupt_extra_value,
        _corrupt_wrong_type,
        _corrupt_out_of_domain,
        _corrupt_uid,
    ],
)
def test_from_serializable_rejects_bad_payloads(
    tmp_path: Path, corrupt: Callable[[dict[str, object]], None]
) -> None:
    port = _simple(tmp_path)
    genome = ParamGenome(port=port, values={"a": 3, "b": 0.7, "c": "y", "d": 7})
    payload = genome.to_serializable()
    corrupt(payload)
    with pytest.raises(PortError):
        from_serializable(payload, port)


def test_param_distance_zero_when_values_equal(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    one = ParamGenome(port=port, values=port.defaults(), uid="one")
    other = ParamGenome(port=port, values=port.defaults(), uid="other")
    assert param_distance(one, other) == 0.0


def test_param_distance_is_mean_over_free_params(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    base = ParamGenome(port=port, values={"a": 0, "b": 0.0, "c": "x", "d": 7})
    half = ParamGenome(port=port, values={"a": 5, "b": 0.5, "c": "y", "d": 7})
    far = ParamGenome(port=port, values={"a": 10, "b": 1.0, "c": "y", "d": 7})
    assert param_distance(base, half) == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert param_distance(base, far) == pytest.approx(1.0)


def test_read_only_operations_draw_no_rng(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    genome = ParamGenome(port=port, values=port.defaults())
    other = ParamGenome(port=port, values={"a": 0, "b": 0.0, "c": "y", "d": 7})
    payload = genome.to_serializable()
    before = random.getstate()
    genome.clone()
    genome.to_serializable()
    from_serializable(payload, port)
    ops.is_valid(genome)
    param_distance(genome, other)
    genome.to_dsl()
    assert random.getstate() == before


def test_initialize_seeds_defaults_and_random(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    seed = ParamGenome(port=port, values={"a": 1, "b": 0.1, "c": "y", "d": 7})
    random.seed(11)
    population = ops.initialize(8, [seed, {"a": 2, "b": 0.2, "c": "x", "d": 7}])
    assert len(population) == 8
    assert population[0].values == seed.values
    assert population[0].uid == seed.uid
    assert population[1].values == {"a": 2, "b": 0.2, "c": "x", "d": 7}
    assert any(genome.values == port.defaults() for genome in population)
    for genome in population:
        port.check_values(genome.values)


def test_initialize_dedupes_defaults_genome(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    population = ops.initialize(3, [port.defaults()])
    assert sum(1 for genome in population if genome.values == port.defaults()) == 1


def test_initialize_rejects_foreign_seed(tmp_path: Path) -> None:
    simple = _simple(tmp_path)
    repair = _repair(tmp_path)
    ops = ParamOps(simple)
    foreign = ParamGenome(port=repair, values=repair.defaults())
    with pytest.raises(PortError):
        ops.initialize(3, [foreign])


def test_initialize_truncates_to_size(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    seed = ParamGenome(port=port, values={"a": 1, "b": 0.1, "c": "y", "d": 7})
    population = ops.initialize(1, [seed])
    assert len(population) == 1
    assert population[0].values == seed.values


def test_seeded_populations_are_deterministic(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    ops = ParamOps(port)
    random.seed(7)
    first = ops.initialize(6)
    random.seed(7)
    second = ops.initialize(6)
    assert [genome.values for genome in first] == [genome.values for genome in second]
    random.seed(8)
    third = ops.initialize(6)
    assert [genome.values for genome in third] != [genome.values for genome in first]


def test_to_dsl_overrides_name(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    genome = ParamGenome(port=port, values=port.defaults())
    dsl = genome.to_dsl()
    assert dsl["name"] == f"{port.name}_{genome.uid}"
    assert dsl["description"] == "a=5 b=0.5 c=x"


def test_to_dsl_forwards_timeframe(tmp_path: Path) -> None:
    doc = _simple_doc()
    doc["allowed_timeframes"] = ["1h", "4h"]
    port = _load(tmp_path, doc)
    genome = ParamGenome(port=port, values=port.defaults())
    assert genome.to_dsl(timeframe="4h")["timeframe"] == "4h"
    with pytest.raises(PortError, match="does not support timeframe 15m"):
        genome.to_dsl(timeframe="15m")


_SNAP_TARGET = 0.5


def _off_grid_genome(port: Port) -> tuple[dict[str, ParamValue], ParamGenome]:
    values = port.defaults()
    values["b"] = 0.5000000001
    return values, ParamGenome(port, values, uid="fixed")


@pytest.mark.parametrize("operation", ["restore", "mutate", "crossover", "initialize", "repair"])
def test_operations_store_snapped_floats(tmp_path: Path, operation: str) -> None:
    port = _simple(tmp_path)
    values, genome = _off_grid_genome(port)
    ops = ParamOps(port)
    if operation == "restore":
        result = from_serializable(genome.to_serializable(), port)
    elif operation == "mutate":
        result = ops.mutate(genome, 0)
    elif operation == "crossover":
        result = ops.crossover(genome, genome)[0]
    elif operation == "repair":
        result = ops.repair(genome)
        assert result.uid == "fixed"
    else:
        result = ops.initialize(1, [values])[0]
    assert result.values["b"] == _SNAP_TARGET
    assert type(result.values["b"]) is float


def test_genome_seed_is_stored_snapped(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    _values, genome = _off_grid_genome(port)
    seeded = ParamOps(port).initialize(1, [genome])[0]
    assert seeded.values["b"] == _SNAP_TARGET
    assert seeded.uid == "fixed"
    assert genome.values["b"] == 0.5000000001


def test_snapped_seed_counts_as_defaults_genome(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    values, _genome = _off_grid_genome(port)
    population = ParamOps(port).initialize(1, [values])
    assert [genome.values for genome in population] == [port.defaults()]


def test_valid_canonical_repair_keeps_identity(tmp_path: Path) -> None:
    port = _simple(tmp_path)
    genome = ParamGenome(port, port.defaults())
    assert ParamOps(port).repair(genome) is genome


def _near_zero_constraint_port(tmp_path: Path) -> Port:
    doc = _simple_doc()
    params = doc["params"]
    assert isinstance(params, dict)
    params["a"] = {"kind": "float", "low": 0.0, "high": 1.0, "step": 0.1, "default": 0.0}
    doc["constraints"] = ["b <= a"]
    params["b"] = {"kind": "float", "low": 0.0, "high": 1.0, "step": 0.1, "default": 0.0}
    return _load(tmp_path, doc)


def test_is_valid_agrees_with_render_on_snapped_constraint_operands(tmp_path: Path) -> None:
    port = _near_zero_constraint_port(tmp_path)
    values = port.defaults()
    values["a"] = 0.0
    values["b"] = 1e-10
    genome = ParamGenome(port, values, uid="fixed")
    restored = from_serializable(genome.to_serializable(), port)
    assert restored.values["b"] == 0.0
    assert port.render(values)["description"] == "a=0.0 b=0.0 c=x"
    assert ParamOps(port).is_valid(genome) is True


def test_is_valid_false_where_render_raises_on_constraint(tmp_path: Path) -> None:
    port = _near_zero_constraint_port(tmp_path)
    values = port.defaults()
    values["a"] = 0.0
    values["b"] = 0.1
    with pytest.raises(PortError):
        port.render(values)
    assert ParamOps(port).is_valid(ParamGenome(port, values)) is False


@pytest.mark.parametrize("entrypoint", ["render", "restore", "is_valid", "repair"])
def test_huge_external_float_value_is_domain_error(tmp_path: Path, entrypoint: str) -> None:
    port = _simple(tmp_path)
    values = port.defaults()
    values["b"] = 10**400
    genome = ParamGenome(port, values, uid="fixed")
    ops = ParamOps(port)
    if entrypoint == "is_valid":
        assert ops.is_valid(genome) is False
    elif entrypoint == "repair":
        repaired = ops.repair(genome)
        assert ops.is_valid(repaired)
        assert repaired.values["b"] != 10**400
    else:
        with pytest.raises(PortError):
            if entrypoint == "render":
                port.render(values)
            else:
                from_serializable(genome.to_serializable(), port)
