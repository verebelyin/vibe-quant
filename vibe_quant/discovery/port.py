"""TradingView port loader: a ``port.yaml`` file into a validated Strategy DSL dict.

A "port" is one TradingView script expressed as data. The loader validates the
parameter domain once, so a broken port fails before any discovery work starts,
and renders a plain DSL dict for the compiler without ever importing
``scripts/research``.

Nothing in this module draws from the global ``random`` state: the feasibility
sample uses a private ``random.Random(0)``.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import yaml

from vibe_quant.dsl.parser import DSLValidationError, validate_strategy_dict
from vibe_quant.dsl.schema import VALID_TIMEFRAMES

ParamValue = int | float | str
ParamKind = Literal["int", "float", "choice"]
Constraint = tuple[str, str, str]

MAX_FREE_PARAMS = 10
FEASIBILITY_SAMPLES = 1000
_INT_LIMIT = 2**53

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_PLACEHOLDER_FULL_RE = re.compile(r"\A\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}\Z")
_CONSTRAINT_RE = re.compile(
    r"\A\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\Z"
)

_PARAM_KEYS = frozenset({"kind", "low", "high", "step", "default", "choices"})
_NUMERIC_KEYS = ("low", "high", "step", "default")
_KINDS = frozenset({"int", "float", "choice"})


class PortError(ValueError):
    """Raised when a port file is malformed or a value is outside its domain."""


def _tolerance(value: float) -> float:
    """Absolute grid tolerance: 1e-9 scaled by ``max(1, |value|)``."""
    return 1e-9 * max(1.0, abs(value))


def _compare(left: int | float, op: str, right: int | float) -> bool:
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    return left >= right


def _as_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"constraint operand must be numeric, got {type(value).__name__}"
        raise PortError(msg)
    return value


@dataclass(frozen=True)
class PortParam:
    """One evolvable input of a port.

    A numeric param is fixed (not free) when ``low == high``; a choice param is
    fixed when it has a single choice.
    """

    name: str
    kind: ParamKind
    low: int | float | None
    high: int | float | None
    step: int | float | None
    choices: tuple[ParamValue, ...]
    default: ParamValue

    @property
    def is_free(self) -> bool:
        """Whether the GA may vary this param."""
        if self.kind == "choice":
            return len(self.choices) > 1
        return self.low != self.high

    @property
    def n_steps(self) -> int:
        """Number of steps from ``low`` to the last grid value."""
        low, high, step = self._bounds()
        if self.kind == "int":
            return (int(high) - int(low)) // int(step)
        raw = (high - low) / step
        return max(0, math.floor(raw + _tolerance(raw)))

    def value_at(self, index: int) -> int | float:
        """Grid value at ``index`` steps above ``low``."""
        low, _high, step = self._bounds()
        if self.kind == "int":
            return int(low) + index * int(step)
        return round(low + index * step, 10)

    def index_of(self, value: ParamValue) -> int:
        """Grid index of a canonical numeric value."""
        low, _high, step = self._bounds()
        if self.kind == "int":
            return (int(value) - int(low)) // int(step)
        return round((float(value) - low) / step)

    def sample(self, rng: random.Random) -> ParamValue:
        """Draw a uniform value for a free param."""
        if self.kind == "choice":
            return rng.choice(self.choices)
        return self.value_at(rng.randrange(self.n_steps + 1))

    def canonical(self, value: object) -> ParamValue:
        """Return ``value`` in domain form or raise ``PortError``."""
        canonical, error = self._evaluate(value)
        if error is not None or canonical is None:
            raise PortError(error or f"param '{self.name}': invalid value")
        return canonical

    def in_domain(self, value: object) -> bool:
        """Whether ``value`` is a canonical value for this param (never raises)."""
        return self._evaluate(value)[1] is None

    def _evaluate(self, value: object) -> tuple[ParamValue | None, str | None]:
        if self.kind == "choice":
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                return None, self._type_error(value)
            for choice in self.choices:
                if type(value) is type(choice) and value == choice:
                    return choice, None
            return None, f"param '{self.name}': {value!r} is not one of {list(self.choices)!r}"
        if isinstance(value, bool):
            return None, self._type_error(value)
        low, high, step = self._bounds()
        if self.kind == "int":
            if not isinstance(value, int):
                return None, self._type_error(value)
            if value < low or value > high:
                return None, self._range_error(value, low, high)
            if (value - int(low)) % int(step) != 0:
                return None, self._grid_error(value)
            return value, None
        if not isinstance(value, (int, float)):
            return None, self._type_error(value)
        if isinstance(value, int) and abs(value) > _INT_LIMIT:
            return None, f"param '{self.name}': {value} exceeds 2**53"
        number = float(value)
        if not math.isfinite(number):
            return None, f"param '{self.name}': value must be finite"
        raw = (number - low) / step
        if not math.isfinite(raw):
            return None, self._range_error(value, low, high)
        index = round(raw)
        offset = index * step
        if not math.isfinite(offset) or not math.isfinite(low + offset):
            return None, self._range_error(value, low, high)
        snapped = round(low + offset, 10)
        if abs(number - snapped) > _tolerance(number):
            return None, self._grid_error(value)
        if snapped < low - _tolerance(low) or snapped > high + _tolerance(high):
            return None, self._range_error(value, low, high)
        return snapped, None

    def _bounds(self) -> tuple[float, float, float]:
        if self.low is None or self.high is None or self.step is None:
            msg = f"param '{self.name}' has no numeric bounds"
            raise PortError(msg)
        return float(self.low), float(self.high), float(self.step)

    def _type_error(self, value: object) -> str:
        return f"param '{self.name}': wrong type {type(value).__name__}"

    def _range_error(self, value: object, low: float, high: float) -> str:
        return f"param '{self.name}': {value!r} is outside [{low}, {high}]"

    def _grid_error(self, value: object) -> str:
        return f"param '{self.name}': {value!r} is off the step grid"


def _parse_param(name: str, spec: object) -> PortParam:
    if not isinstance(spec, Mapping):
        msg = f"params.{name}: expected a mapping"
        raise PortError(msg)
    unknown = set(spec) - _PARAM_KEYS
    if unknown:
        msg = f"params.{name}: unknown key(s) {sorted(map(str, unknown))}"
        raise PortError(msg)
    kind_raw = spec.get("kind")
    if not isinstance(kind_raw, str) or kind_raw not in _KINDS:
        msg = f"params.{name}: unknown kind {kind_raw!r}"
        raise PortError(msg)
    kind = cast("ParamKind", kind_raw)
    param = _parse_choice_param(name, spec) if kind == "choice" else _parse_numeric_param(
        name, spec, kind
    )
    if "default" not in spec:
        msg = f"params.{name}: missing 'default'"
        raise PortError(msg)
    return replace(param, default=param.canonical(spec["default"]))


def _parse_numeric_param(name: str, spec: Mapping[object, object], kind: ParamKind) -> PortParam:
    if "choices" in spec:
        msg = f"params.{name}: 'choices' is not valid for a {kind} param"
        raise PortError(msg)
    literals: dict[str, int | float] = {}
    for key in _NUMERIC_KEYS:
        if key not in spec:
            msg = f"params.{name}: missing '{key}'"
            raise PortError(msg)
        literals[key] = _numeric_literal(name, key, spec[key], kind)
    low = literals["low"]
    high = literals["high"]
    step = literals["step"]
    if low > high:
        msg = f"params.{name}: low {low} > high {high}"
        raise PortError(msg)
    if step <= 0:
        msg = f"params.{name}: step must be positive, got {step}"
        raise PortError(msg)
    if kind == "float" and not math.isfinite((float(high) - float(low)) / float(step)):
        msg = f"params.{name}: grid [{low}, {high}] step {step} overflows float arithmetic"
        raise PortError(msg)
    return PortParam(
        name=name,
        kind=kind,
        low=low,
        high=high,
        step=step,
        choices=(),
        default=literals["default"],
    )


def _parse_choice_param(name: str, spec: Mapping[object, object]) -> PortParam:
    for key in ("low", "high", "step"):
        if key in spec:
            msg = f"params.{name}: '{key}' is not valid for a choice param"
            raise PortError(msg)
    raw = spec.get("choices")
    if not isinstance(raw, (list, tuple)) or not raw:
        msg = f"params.{name}.choices: expected a non-empty list"
        raise PortError(msg)
    choices: list[ParamValue] = []
    kinds: set[type[object]] = set()
    for choice in raw:
        if isinstance(choice, bool) or not isinstance(choice, (str, int, float)):
            msg = f"params.{name}.choices: unsupported choice {choice!r}"
            raise PortError(msg)
        if isinstance(choice, float) and not math.isfinite(choice):
            msg = f"params.{name}.choices: {choice!r} must be finite"
            raise PortError(msg)
        if isinstance(choice, int) and abs(choice) > _INT_LIMIT:
            msg = f"params.{name}.choices: {choice} exceeds 2**53"
            raise PortError(msg)
        kinds.add(type(choice))
        choices.append(choice)
    if len(kinds) > 1:
        msg = f"params.{name}.choices: mixed scalar types"
        raise PortError(msg)
    if len(set(choices)) != len(choices):
        msg = f"params.{name}.choices: duplicate values"
        raise PortError(msg)
    return PortParam(
        name=name,
        kind="choice",
        low=None,
        high=None,
        step=None,
        choices=tuple(choices),
        default=choices[0],
    )


def _numeric_literal(name: str, key: str, value: object, kind: ParamKind) -> int | float:
    if isinstance(value, bool):
        msg = f"params.{name}.{key}: boolean is not a number"
        raise PortError(msg)
    if kind == "int":
        if not isinstance(value, int):
            msg = f"params.{name}.{key}: expected int, got {type(value).__name__}"
            raise PortError(msg)
        if abs(value) > _INT_LIMIT:
            msg = f"params.{name}.{key}: {value} exceeds 2**53"
            raise PortError(msg)
        return value
    if not isinstance(value, (int, float)):
        msg = f"params.{name}.{key}: expected a number, got {type(value).__name__}"
        raise PortError(msg)
    if isinstance(value, int) and abs(value) > _INT_LIMIT:
        msg = f"params.{name}.{key}: {value} exceeds 2**53"
        raise PortError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"params.{name}.{key}: value must be finite"
        raise PortError(msg)
    return number


def _parse_constraint(raw: object, numeric: set[str]) -> Constraint:
    if not isinstance(raw, str):
        msg = f"constraint {raw!r} is not a string"
        raise PortError(msg)
    match = _CONSTRAINT_RE.match(raw)
    if match is None:
        msg = f"constraint {raw!r} is not 'a OP b' with OP in < <= > >="
        raise PortError(msg)
    left, op, right = match.group(1), match.group(2), match.group(3)
    for name in (left, right):
        if name not in numeric:
            msg = f"constraint {raw!r} references non-numeric param '{name}'"
            raise PortError(msg)
    return (left, op, right)


def _collect_placeholders(node: object, used: set[str]) -> None:
    if isinstance(node, Mapping):
        for key, child in node.items():
            if isinstance(key, str) and _PLACEHOLDER_RE.search(key):
                msg = f"placeholder not allowed in mapping key: {key!r}"
                raise PortError(msg)
            _collect_placeholders(child, used)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_placeholders(item, used)
        return
    if isinstance(node, str):
        used.update(match.group(1) for match in _PLACEHOLDER_RE.finditer(node))


def _substitute_node(node: object, values: Mapping[str, ParamValue]) -> object:
    if isinstance(node, Mapping):
        rendered: dict[str, object] = {}
        for key, child in node.items():
            if not isinstance(key, str):
                msg = f"mapping key must be a string, got {key!r}"
                raise PortError(msg)
            if _PLACEHOLDER_RE.search(key):
                msg = f"placeholder not allowed in mapping key: {key!r}"
                raise PortError(msg)
            rendered[key] = _substitute_node(child, values)
        return rendered
    if isinstance(node, (list, tuple)):
        return [_substitute_node(item, values) for item in node]
    if isinstance(node, str):
        full = _PLACEHOLDER_FULL_RE.match(node)
        if full is not None:
            return _lookup(full.group(1), values)
        return _PLACEHOLDER_RE.sub(lambda match: str(_lookup(match.group(1), values)), node)
    return node


def _lookup(name: str, values: Mapping[str, ParamValue]) -> ParamValue:
    if name not in values:
        msg = f"unknown placeholder {name!r}"
        raise PortError(msg)
    return values[name]


def _parse_allowed_timeframes(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        msg = "allowed_timeframes must be a non-empty list of timeframe strings"
        raise PortError(msg)
    allowed: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or item not in VALID_TIMEFRAMES:
            msg = f"allowed_timeframes: invalid timeframe {item!r}"
            raise PortError(msg)
        if item in seen:
            msg = f"allowed_timeframes: duplicate timeframe {item!r}"
            raise PortError(msg)
        seen.add(item)
        allowed.append(item)
    return tuple(allowed)


def _feasible(port: Port, rng: random.Random) -> bool:
    for _ in range(FEASIBILITY_SAMPLES):
        values = {
            param.name: param.sample(rng) if param.is_free else param.default
            for param in port.params
        }
        if port.satisfies_constraints(values):
            return True
    return False


@dataclass(frozen=True)
class Port:
    """One TradingView script as data, validated at load time."""

    name: str
    path: Path
    sha256: str
    params: tuple[PortParam, ...]
    constraints: tuple[Constraint, ...]
    dsl_tree: Mapping[str, object]
    tv: Mapping[str, object]
    allowed_timeframes: tuple[str, ...] | None = None

    def free_params(self) -> tuple[PortParam, ...]:
        """Params the GA may vary."""
        return tuple(param for param in self.params if param.is_free)

    def defaults(self) -> dict[str, ParamValue]:
        """Declared default for every param."""
        return {param.name: param.default for param in self.params}

    def check_values(self, values: Mapping[str, object]) -> None:
        """Raise ``PortError`` unless ``values`` is in domain and satisfies constraints."""
        self.canonical_values(values)

    def satisfies_constraints(self, values: Mapping[str, object]) -> bool:
        """Whether ``values`` satisfies every declared constraint.

        In-domain operands are compared in their canonical (snapped) form, the
        same form ``render`` and ``check_values`` use.
        """
        by_name = {param.name: param for param in self.params}

        def operand(name: str) -> int | float:
            value = values[name]
            param = by_name[name]
            return _as_number(param.canonical(value) if param.in_domain(value) else value)

        return all(
            _compare(operand(left), op, operand(right)) for left, op, right in self.constraints
        )

    def render(
        self, values: Mapping[str, object], timeframe: str | None = None
    ) -> dict[str, object]:
        """Return the substituted, validated DSL dict (a deep copy)."""
        canonical = self.canonical_values(values)
        substituted = _substitute_node(self.dsl_tree, canonical)
        if not isinstance(substituted, dict):
            msg = f"port '{self.name}': dsl must be a mapping"
            raise PortError(msg)
        if timeframe is not None:
            substituted["timeframe"] = timeframe
        effective = substituted.get("timeframe")
        if self.allowed_timeframes is not None and effective not in self.allowed_timeframes:
            msg = f"port {self.name} does not support timeframe {effective}"
            raise PortError(msg)
        try:
            validate_strategy_dict(substituted)
        except DSLValidationError as exc:
            raise PortError(str(exc)) from exc
        return substituted

    def canonical_values(self, values: Mapping[str, object]) -> dict[str, ParamValue]:
        """Every param's value in canonical form, or ``PortError`` (domain + constraints)."""
        canonical, error = self._evaluate_values(values)
        if canonical is None:
            raise PortError(error or f"port '{self.name}': invalid values")
        return canonical

    def try_canonical_values(self, values: Mapping[str, object]) -> dict[str, ParamValue] | None:
        """``canonical_values`` without raising: ``None`` when ``values`` are rejected."""
        return self._evaluate_values(values)[0]

    def _evaluate_values(
        self, values: Mapping[str, object]
    ) -> tuple[dict[str, ParamValue] | None, str | None]:
        known = {param.name for param in self.params}
        provided = set(values)
        missing = known - provided
        if missing:
            return None, f"port '{self.name}': missing value(s) {sorted(missing)}"
        extra = provided - known
        if extra:
            return None, f"port '{self.name}': unknown value(s) {sorted(map(str, extra))}"
        canonical: dict[str, ParamValue] = {}
        for param in self.params:
            value, error = param._evaluate(values[param.name])
            if value is None:
                return None, error or f"param '{param.name}': invalid value"
            canonical[param.name] = value
        for left, op, right in self.constraints:
            if not _compare(_as_number(canonical[left]), op, _as_number(canonical[right])):
                return None, f"port '{self.name}': values violate constraint '{left} {op} {right}'"
        return canonical, None


def load_port(path: str | Path) -> Port:
    """Load and validate a port from its folder or ``port.yaml`` file."""
    port_path = Path(path)
    if port_path.is_dir():
        port_path = port_path / "port.yaml"
    try:
        raw = port_path.read_bytes()
    except OSError as exc:
        msg = f"cannot read port file {port_path}: {exc}"
        raise PortError(msg) from exc
    try:
        data: object = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"invalid YAML in {port_path}: {exc}"
        raise PortError(msg) from exc
    if not isinstance(data, Mapping):
        msg = f"port file {port_path} is not a mapping"
        raise PortError(msg)

    name_raw = data.get("name")
    if not isinstance(name_raw, str) or not name_raw:
        msg = f"port file {port_path}: 'name' must be a non-empty string"
        raise PortError(msg)

    params_raw = data.get("params")
    if not isinstance(params_raw, Mapping):
        msg = f"port '{name_raw}': 'params' must be a mapping"
        raise PortError(msg)
    params = tuple(_parse_param(str(key), spec) for key, spec in params_raw.items())

    constraints_raw = data.get("constraints", [])
    if constraints_raw is None:
        constraints_raw = []
    if not isinstance(constraints_raw, (list, tuple)):
        msg = f"port '{name_raw}': 'constraints' must be a list"
        raise PortError(msg)
    numeric = {param.name for param in params if param.kind != "choice"}
    constraints = tuple(_parse_constraint(item, numeric) for item in constraints_raw)

    dsl_raw = data.get("dsl")
    if not isinstance(dsl_raw, Mapping):
        msg = f"port '{name_raw}': 'dsl' must be a mapping"
        raise PortError(msg)
    used: set[str] = set()
    _collect_placeholders(dsl_raw, used)
    declared = {param.name for param in params}
    unknown = used - declared
    if unknown:
        msg = f"port '{name_raw}': unknown placeholder(s) {sorted(unknown)}"
        raise PortError(msg)
    unused = declared - used
    if unused:
        msg = f"port '{name_raw}': param(s) never used by a placeholder {sorted(unused)}"
        raise PortError(msg)

    tv_raw = data.get("tv", {})
    if tv_raw is None:
        tv_raw = {}
    if not isinstance(tv_raw, Mapping):
        msg = f"port '{name_raw}': 'tv' must be a mapping"
        raise PortError(msg)

    port = Port(
        name=name_raw,
        path=port_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        params=params,
        constraints=constraints,
        dsl_tree=dict(dsl_raw),
        tv=dict(tv_raw),
        allowed_timeframes=(
            _parse_allowed_timeframes(data["allowed_timeframes"])
            if "allowed_timeframes" in data
            else None
        ),
    )
    if len(port.free_params()) > MAX_FREE_PARAMS:
        msg = f"port '{port.name}' has more than {MAX_FREE_PARAMS} free params"
        raise PortError(msg)
    if port.constraints:
        defaults = port.defaults()
        if not port.satisfies_constraints(defaults):
            if not _feasible(port, random.Random(0)):
                raise PortError("constraints unsatisfiable")
            msg = f"port '{port.name}': default values violate a constraint"
            raise PortError(msg)
    port.render(port.defaults())
    return port
