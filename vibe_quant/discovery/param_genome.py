"""Parameter genome: evolve a port's own inputs instead of a chromosome.

A ``ParamGenome`` carries one value per port param, renders that into the port's
Strategy DSL, and is evolved by ``ParamOps`` on the port's own grid. Read-only
operations never draw from the global ``random`` state; the GA operators do, the
way ``vibe_quant.discovery.operators`` does.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from vibe_quant.discovery.operators import _new_uid
from vibe_quant.discovery.port import ParamValue, Port, PortError

REPAIR_MUTATIONS = 32
REPAIR_DRAWS = 32


def _same_values(canonical: Mapping[str, ParamValue], values: Mapping[str, object]) -> bool:
    """Whether ``values`` already equals ``canonical`` in value and type."""
    return canonical.keys() == values.keys() and all(
        type(values[name]) is type(value) and values[name] == value
        for name, value in canonical.items()
    )


@dataclass(slots=True)
class ParamGenome:
    """One set of values for a port's params."""

    port: Port
    values: dict[str, ParamValue]
    uid: str = field(default_factory=_new_uid)

    def clone(self) -> ParamGenome:
        """Deep-copy the values, KEEPING the uid (an unchanged copy)."""
        return ParamGenome(port=self.port, values=dict(self.values), uid=self.uid)

    def to_dsl(self, timeframe: str | None = None) -> dict[str, object]:
        """Render the genome to a validated DSL dict under a unique name."""
        dsl = self.port.render(self.values, timeframe=timeframe)
        dsl["name"] = f"{self.port.name}_{self.uid}"
        return dsl

    def to_serializable(self) -> dict[str, object]:
        """Portable form: port identity, values and uid."""
        return {
            "kind": "param",
            "port": {
                "name": self.port.name,
                "path": str(self.port.path),
                "sha256": self.port.sha256,
            },
            "values": dict(self.values),
            "uid": self.uid,
        }


def from_serializable(data: Mapping[str, object], port: Port) -> ParamGenome:
    """Rebuild a genome from ``to_serializable`` output for the given port."""
    if not isinstance(data, Mapping):
        msg = "serialized genome must be a mapping"
        raise PortError(msg)
    if data.get("kind") != "param":
        msg = f"serialized genome kind is not 'param': {data.get('kind')!r}"
        raise PortError(msg)
    meta = data.get("port")
    if not isinstance(meta, Mapping):
        msg = "serialized genome has no port identity"
        raise PortError(msg)
    if meta.get("name") != port.name:
        msg = f"serialized genome belongs to port {meta.get('name')!r}, not {port.name!r}"
        raise PortError(msg)
    if meta.get("sha256") != port.sha256:
        msg = "serialized genome was produced by a different port.yaml revision"
        raise PortError(msg)
    raw_values = data.get("values")
    if not isinstance(raw_values, Mapping):
        msg = "serialized genome has no values mapping"
        raise PortError(msg)
    values = port.canonical_values(raw_values)
    uid = data.get("uid")
    if not isinstance(uid, str) or not uid:
        msg = "serialized genome has no uid"
        raise PortError(msg)
    return ParamGenome(port=port, values=values, uid=uid)


class GenomeOps(Protocol):
    """The operator surface the discovery pipeline drives by type."""

    def initialize(
        self, size: int, seeds: Sequence[ParamGenome | Mapping[str, ParamValue]] | None = None
    ) -> list[ParamGenome]: ...

    def random(self) -> ParamGenome: ...

    def mutate(self, genome: ParamGenome, rate: float) -> ParamGenome: ...

    def crossover(
        self, parent_a: ParamGenome, parent_b: ParamGenome
    ) -> tuple[ParamGenome, ParamGenome]: ...

    def is_valid(self, genome: ParamGenome) -> bool: ...

    def repair(self, genome: ParamGenome) -> ParamGenome: ...


class ParamOps:
    """GA operators over one port's parameter grid."""

    def __init__(self, port: Port) -> None:
        if not port.free_params():
            msg = f"port '{port.name}' has no free parameters; discovery needs at least one"
            raise PortError(msg)
        self.port = port

    def random(self) -> ParamGenome:
        """Uniform draw over every free param, then repair."""
        return self.repair(ParamGenome(port=self.port, values=self._random_values()))

    def mutate(self, genome: ParamGenome, rate: float) -> ParamGenome:
        """Step-mutate each free param with probability ``rate``; fresh uid."""
        values = self._mutate_values(genome.values, rate)
        return self.repair(ParamGenome(port=self.port, values=values))

    def crossover(
        self, parent_a: ParamGenome, parent_b: ParamGenome
    ) -> tuple[ParamGenome, ParamGenome]:
        """Uniform crossover per free param; two children with fresh uids."""
        if parent_a.port.sha256 != parent_b.port.sha256:
            msg = "cannot crossover genomes from different ports"
            raise PortError(msg)
        values_a = dict(parent_a.values)
        values_b = dict(parent_b.values)
        for param in self.port.params:
            if not param.is_free:
                continue
            if random.random() < 0.5:
                values_a[param.name], values_b[param.name] = (
                    values_b[param.name],
                    values_a[param.name],
                )
        child_a = self.repair(ParamGenome(port=self.port, values=values_a))
        child_b = self.repair(ParamGenome(port=self.port, values=values_b))
        return child_a, child_b

    def is_valid(self, genome: ParamGenome) -> bool:
        """Whether the genome's values are in domain and satisfy constraints."""
        return self._values_valid(genome.values)

    def repair(self, genome: ParamGenome) -> ParamGenome:
        """Return ``genome`` if valid, else mutate then resample, else raise.

        A valid genome whose values are already canonical comes back as the same
        object; one with off-grid noise comes back with its snapped values and uid.
        """
        canonical = self._canonical_or_none(genome.values)
        if canonical is not None:
            if _same_values(canonical, genome.values):
                return genome
            return ParamGenome(port=self.port, values=canonical, uid=genome.uid)
        values = dict(genome.values)
        for _ in range(REPAIR_MUTATIONS):
            values = self._mutate_values(values, 1.0)
            canonical = self._canonical_or_none(values)
            if canonical is not None:
                return ParamGenome(port=self.port, values=canonical)
        for _ in range(REPAIR_DRAWS):
            canonical = self._canonical_or_none(self._random_values())
            if canonical is not None:
                return ParamGenome(port=self.port, values=canonical)
        raise PortError("repair failed")

    def initialize(
        self, size: int, seeds: Sequence[ParamGenome | Mapping[str, ParamValue]] | None = None
    ) -> list[ParamGenome]:
        """Seeds + the defaults genome + random draws, truncated to ``size``."""
        population: list[ParamGenome] = [self._seed_genome(seed) for seed in seeds or ()]
        defaults = self.port.defaults()
        if not any(genome.values == defaults for genome in population):
            population.append(ParamGenome(port=self.port, values=dict(defaults)))
        while len(population) < size:
            population.append(self.random())
        return population[:size]

    def _seed_genome(self, seed: object) -> ParamGenome:
        if isinstance(seed, ParamGenome):
            if seed.port.sha256 != self.port.sha256:
                msg = "seed genome belongs to a different port"
                raise PortError(msg)
            canonical = self._canonical_or_none(seed.values)
            if canonical is None:
                return seed.clone()
            return ParamGenome(port=self.port, values=canonical, uid=seed.uid)
        if isinstance(seed, Mapping):
            return ParamGenome(port=self.port, values=self.port.canonical_values(seed))
        msg = f"unsupported seed of type {type(seed).__name__}"
        raise PortError(msg)

    def _values_valid(self, values: Mapping[str, object]) -> bool:
        return self._canonical_or_none(values) is not None

    def _canonical_or_none(self, values: Mapping[str, object]) -> dict[str, ParamValue] | None:
        """Canonical values when ``port.render`` would accept them, else ``None``."""
        return self.port.try_canonical_values(values)

    def _random_values(self) -> dict[str, ParamValue]:
        values: dict[str, ParamValue] = {}
        for param in self.port.params:
            if param.kind == "choice":
                values[param.name] = (
                    random.choice(param.choices) if param.is_free else param.default
                )
            elif param.is_free:
                values[param.name] = param.value_at(random.randrange(param.n_steps + 1))
            else:
                values[param.name] = param.default
        return values

    def _mutate_values(
        self, values: Mapping[str, ParamValue], rate: float
    ) -> dict[str, ParamValue]:
        mutated = dict(values)
        for param in self.port.params:
            if not param.is_free:
                continue
            if random.random() >= rate:
                continue
            if param.kind == "choice":
                others = [choice for choice in param.choices if choice != mutated[param.name]]
                mutated[param.name] = random.choice(others)
                continue
            steps = max(1, param.n_steps // 10)
            shift = random.randint(1, steps) * random.choice((-1, 1))
            try:
                index = param.index_of(mutated[param.name]) + shift
            except (ArithmeticError, TypeError, ValueError):
                index = random.randrange(param.n_steps + 1)
            index = max(0, min(param.n_steps, index))
            mutated[param.name] = param.value_at(index)
        return mutated


def param_distance(a: ParamGenome, b: ParamGenome) -> float:
    """Mean normalised distance over free params; fixed params are ignored."""
    free = a.port.free_params()
    if not free:
        return 0.0
    total = 0.0
    for param in free:
        value_a = a.values[param.name]
        value_b = b.values[param.name]
        if param.kind == "choice":
            if value_a != value_b:
                total += 1.0
            continue
        if param.low is None or param.high is None:
            continue
        span = float(param.high) - float(param.low)
        if span <= 0:
            continue
        total += abs(float(value_a) - float(value_b)) / span
    return total / len(free)
