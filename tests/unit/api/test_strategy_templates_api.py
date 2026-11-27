"""GET /api/strategies/templates feeds the Create Strategy dialog."""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003

import pytest
from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.strategies.templates._metadata import TEMPLATES


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app()
    mgr = StateManager(db_path=tmp_path / "test.db")
    _ = mgr.conn
    app.state.state_manager = mgr
    yield TestClient(app)
    mgr.close()


def test_templates_endpoint_lists_every_valid_template(client: TestClient) -> None:
    r = client.get("/api/strategies/templates")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == len(TEMPLATES) == 12
    names = [i["name"] for i in items]
    assert len(set(names)) == len(names)
    assert "dual_ema_crossover" in names
    for item in items:  # the fields StrategyCreateDialog reads
        assert item["description"]
        assert item["strategy_type"]
        assert isinstance(item["dsl_config"], dict) and item["dsl_config"]


def test_strategy_created_from_template_payload_is_stored_and_valid(client: TestClient) -> None:
    item = next(
        i for i in client.get("/api/strategies/templates").json() if i["name"] == "ema_ribbon"
    )

    created = client.post(
        "/api/strategies",
        json={
            "name": f"{item['name']}_x1",
            "description": item["description"],
            "strategy_type": item["strategy_type"],
            "dsl_config": item["dsl_config"],
        },
    )
    assert created.status_code == 201
    row = client.get(f"/api/strategies/{created.json()['id']}").json()
    assert row["name"] == "ema_ribbon_x1"
    assert row["dsl_config"] == item["dsl_config"]
    assert set(row["dsl_config"]["indicators"]) == {"ema_fast", "ema_mid", "ema_slow", "rsi", "atr"}
    assert client.post(f"/api/strategies/{row['id']}/validate").json() == {
        "valid": True,
        "errors": [],
    }
    validate_strategy_dict(row["dsl_config"])


def test_templates_endpoint_runs_off_the_event_loop() -> None:
    import inspect

    from vibe_quant.api.routers.strategies import list_templates

    assert not inspect.iscoroutinefunction(list_templates)
