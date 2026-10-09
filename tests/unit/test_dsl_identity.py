"""Tests for vibe_quant.dsl.identity.dsl_body_key — the shared DSL identity key.

The canonical-JSON-minus-name formula used to be duplicated byte-for-byte in
vibe_quant/validation/consistency.py (``_dsl_body``) and
vibe_quant/api/routers/discovery.py (``_dsl_body_key``). If the two ever
diverge, dedupe (promote) and validation consistency reference matching
silently disagree about strategy identity. These tests pin the byte-exact
output against the old formula and assert both modules now share one function
object from vibe_quant.dsl.identity.
"""

from __future__ import annotations

import json
from datetime import datetime

import vibe_quant.api.routers.discovery as discovery
import vibe_quant.validation.consistency as consistency
from vibe_quant.dsl import identity


def _legacy_formula(dsl: dict[str, object]) -> str:
    """The pre-refactor formula (consistency._dsl_body / discovery._dsl_body_key)."""
    body = {k: v for k, v in dsl.items() if k != "name"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


SAMPLE_DSLS: tuple[dict[str, object], ...] = (
    {"name": "genome_flat", "direction": "long", "size": 0.25, "leverage": 2.0},
    {
        "name": "genome_nested",
        "entry": {"all": [{"lhs": {"ind": "RSI", "ref": 0}, "op": "<", "rhs": 30.5}]},
        "indicators": [{"ind": "EMA", "params": {"period": 21, "alpha": 0.07142857142857142}}],
        "risk": {"stop": {"atr_mult": 1.5}, "target": 2.25, "flags": [True, None]},
    },
    {"name": "genome_dt", "as_of": datetime(2026, 10, 8, 12, 30, 45), "threshold": 1.5},
)


def test_matches_legacy_formula_byte_for_byte() -> None:
    for dsl in SAMPLE_DSLS:
        assert identity.dsl_body_key(dsl) == _legacy_formula(dsl)


def test_name_is_ignored() -> None:
    for dsl in SAMPLE_DSLS:
        renamed = {**dsl, "name": "totally_different_genome"}
        nameless = {k: v for k, v in dsl.items() if k != "name"}
        assert identity.dsl_body_key(dsl) == identity.dsl_body_key(renamed)
        assert identity.dsl_body_key(dsl) == identity.dsl_body_key(nameless)


def test_key_order_is_irrelevant() -> None:
    for dsl in SAMPLE_DSLS:
        reordered = dict(reversed(list(dsl.items())))
        assert identity.dsl_body_key(dsl) == identity.dsl_body_key(reordered)


def test_both_modules_use_the_shared_function_object(monkeypatch) -> None:
    # (a) both modules reference the same function object as identity.dsl_body_key
    assert consistency.dsl_body_key is identity.dsl_body_key
    assert discovery.dsl_body_key is identity.dsl_body_key
    # (b) the old private duplicates are gone
    assert not hasattr(consistency, "_dsl_body")
    assert not hasattr(discovery, "_dsl_body_key")
    # (c) the real call paths route through each module's dsl_body_key name
    calls: list[dict[str, object]] = []

    def record(dsl: dict[str, object]) -> str:
        calls.append(dsl)
        return "sentinel-key"

    class _StubConn:
        def execute(self, *_args: object) -> _StubConn:
            return self

        def fetchall(self) -> list[object]:
            return []

        def __iter__(self):
            return iter(())

    class _StubState:
        conn = _StubConn()

    monkeypatch.setattr(consistency, "dsl_body_key", record)
    monkeypatch.setattr(discovery, "dsl_body_key", record)

    dsl: dict[str, object] = {"name": "genome_x", "size": 0.25}
    consistency._reference_from_discovery_notes(_StubState(), "genome_x", strategy_dsl=dsl)
    discovery._find_strategy_by_dsl(_StubState(), dsl)

    assert len(calls) == 2
