"""30vki: the start_fresh texts match what the code does.

``start_fresh`` skips only ANOTHER trader id's risk state on the account; this
trader id's own saved risk state and halt always carry. These rows pin that
behaviour against the CLI/API/UI wording. R21 (CLI refusal text) lives in
test_paper_cli.py, R24 (API 400 detail) in test_audit_paper_api.py.
"""

from __future__ import annotations

from typing import Any

from tests.unit.test_paper_halt_restore import seeded  # noqa: F401  (fixture)
from tests.unit.test_paper_rotation_consent import (  # noqa: F401  (helpers)
    K_NEW,
    TRADER,
    Row,
    _restore,
    _seed,
)
from vibe_quant.paper.guard import HaltReason

OTHER = "PAPER-041"


def _sources(events: list[dict[str, object]]) -> list[object]:
    return [e["risk_source_trader_id"] for e in events if e.get("action") == "restore"]


def test_live_start_fresh_keeps_own_state(seeded: dict[str, Any]) -> None:  # noqa: F811
    """R22: the own row (risk and halt) is kept, as the CLI's consent text says."""
    _seed(seeded["db"], [Row(TRADER, K_NEW, halt="manual", hwm="10600")])

    node, events, persistence = _restore(seeded, start_fresh=True)
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert node._restored_halt is not None
        assert node._restored_halt.reason is HaltReason.MANUAL
        assert _sources(events) == [TRADER]
    finally:
        persistence.close()
        node._state_manager.close()


def test_start_fresh_skips_other_trader(seeded: dict[str, Any]) -> None:  # noqa: F811
    """R23: only the other trader id's newer risk is skipped, not the own row."""
    _seed(seeded["db"], [Row(TRADER, K_NEW, hwm="10000"), Row(OTHER, K_NEW, hwm="12000")])

    for start_fresh, hwm, source in (
        (True, "10000", TRADER),
        (False, "12000", OTHER),
    ):
        node, events, persistence = _restore(seeded, start_fresh=start_fresh)
        try:
            assert node._restored_risk is not None, start_fresh
            assert node._restored_risk.to_dict()["high_water_mark"] == hwm, start_fresh
            assert _sources(events) == [source], start_fresh
        finally:
            persistence.close()
            node._state_manager.close()
