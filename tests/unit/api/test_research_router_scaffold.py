"""Endpoint tests for POST /api/research/extractions/{id}/indicators/{idx}/scaffold.

Slice 2 of bd-3p1k.1: the LLM codegen step is wired up — tests mock
``_run_claude_codegen`` and the mypy/ruff gates so the suite never
shells out for real. The slice-1 validation paths (404 / invalid_input /
name_collision / already_scaffolded / force / failed-cache-no-short-
circuit) still apply and are kept intact.
"""

from __future__ import annotations

import json
import re as _re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.api.deps import get_state_manager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.research import extraction_log, indicator_scaffold
from vibe_quant.research.sources import _reset_for_tests, register_source

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def sm(tmp_path: Path) -> Generator[StateManager]:
    s = StateManager(tmp_path / "scaffold.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _isolate_log_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path]:
    root = tmp_path / "research-logs"
    monkeypatch.setattr(extraction_log, "DEFAULT_LOG_ROOT", root)
    yield root


@pytest.fixture(autouse=True)
def _isolate_registry() -> Generator[None]:
    _reset_for_tests()

    @register_source("reddit")
    class FakeReddit:
        name = "reddit"

        def fetch(self, since, limit):  # noqa: ARG002
            yield from ()

    yield
    _reset_for_tests()


def _safe_body_for(prompt: str) -> str:
    """Generate a hermetic SAFE compute_fn body that matches the prompt's name.

    The prompt always contains the locked signature line; we grep the
    function name out so a single stub serves every test regardless of
    the proposal's name.
    """
    m = _re.search(r"def (compute_[a-z0-9_]+)\(", prompt)
    fn_name = m.group(1) if m else "compute_x"
    return (
        f"def {fn_name}(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        f'    import pandas as pd\n'
        f'    period = int(params.get("period", 14) or 14)\n'
        f'    return df["close"].rolling(period).mean()\n'
    )


@pytest.fixture
def _stub_codegen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path]:
    """Pin claude-p + mypy + ruff + pytest + git + dirs for hermetic tests.

    The slice-3 endpoint flow is:
        synthesize_and_write → write contract test → pytest → git commit
        → reload_plugins → upsert row.

    With these stubs the LLM never runs, the toolchain never runs,
    pytest never runs, git never runs, and writes land in tmp_path
    so we don't pollute the real plugins dir or commit log.
    """
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    tests_out = tmp_path / "tests_out"
    tests_out.mkdir()
    monkeypatch.setattr(indicator_scaffold, "PLUGINS_DIR", plugins)
    monkeypatch.setattr(indicator_scaffold, "TESTS_DIR", tests_out)
    monkeypatch.setattr(indicator_scaffold, "_run_claude_codegen", _safe_body_for)
    monkeypatch.setattr(indicator_scaffold, "run_mypy", lambda _p: (True, ""))
    monkeypatch.setattr(indicator_scaffold, "run_ruff", lambda _p: (True, ""))
    monkeypatch.setattr(
        indicator_scaffold,
        "run_contract_test",
        lambda _p, **_kw: (True, "1 passed"),
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "git_commit_scaffold",
        lambda _pp, _tp, *, name, repo_root=None: "a" * 40,
    )
    # reload_plugins() walks the real plugin dir — make it a no-op so
    # the tmp_path scaffolds aren't expected to be importable.
    monkeypatch.setattr(
        "vibe_quant.dsl.plugin_loader.reload_plugins", lambda: []
    )
    yield plugins


@pytest.fixture
def client(sm: StateManager) -> Generator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_state_manager] = lambda: sm
    with TestClient(app) as c:
        yield c


def _seed_extraction_with_proposals(
    sm: StateManager, proposals: list[dict] | None
) -> int:
    item_id = sm.create_research_item(
        source="reddit",
        external_id="abc",
        url="https://reddit.com/abc",
        title="t",
        body="b",
        author="u/x",
        posted_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        score=10,
        extras={"comments": []},
    )
    return sm.create_extraction(
        research_item_id=item_id,
        status="parsed",
        llm_model="t",
        confidence=0.8,
        rationale="ok",
        raw_response="{}",
        dsl_yaml="name: x\n",
        parsed_dsl_json=None,
        parse_error=None,
        proposed_indicators_json=(
            json.dumps(proposals) if proposals is not None else None
        ),
    )


def _scaffold(client: TestClient, ext_id: int, idx: int, force: bool = False) -> dict:
    params = {"force": "true"} if force else None
    resp = client.post(
        f"/api/research/extractions/{ext_id}/indicators/{idx}/scaffold",
        params=params,
    )
    return {"status_code": resp.status_code, "body": resp.json()}


# ---------- 404 paths ----------


def test_scaffold_unknown_extraction_404(client: TestClient) -> None:
    out = _scaffold(client, 9999, 0)
    assert out["status_code"] == 404


def test_scaffold_extraction_without_proposals_404(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(sm, proposals=None)
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 404


def test_scaffold_idx_out_of_range_404(client: TestClient, sm: StateManager) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "x", "formula": "f"}]
    )
    out = _scaffold(client, ext_id, 5)
    assert out["status_code"] == 404


# ---------- status=invalid_input ----------


def test_scaffold_missing_formula_invalid_input(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "x"}]  # no formula
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 400
    detail = out["body"]["detail"]
    assert detail["status"] == "invalid_input"
    assert "formula" in (detail.get("error") or "").lower()


# ---------- status=name_collision ----------


def test_scaffold_name_collision_returns_suggested(
    client: TestClient, sm: StateManager
) -> None:
    # RSI is a built-in spec.
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "rsi", "formula": "100 - 100/(1+RS)"}]
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 409
    detail = out["body"]["detail"]
    assert detail["status"] == "name_collision"
    assert detail["name"] == "RSI"
    # suggested_name survives in the error body — the collision path persists
    # no scaffold row, so this is the ONLY place the UI can read it.
    assert detail["suggested_name"] == "RSI_V2"


# ---------- status=ok (happy path, slice 2) ----------


def test_scaffold_happy_path_writes_plugin_and_caches_ok(
    client: TestClient, sm: StateManager, _stub_codegen: Path
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[
            {
                "name": "my_novel",
                "formula": "ema(close, period) / atr(period)",
                "parameters": {"period": {"default": 14, "range": [5, 30]}},
                "output_range": "unbounded",
            }
        ],
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 200
    assert out["body"]["status"] == "ok"
    assert out["body"]["name"] == "MY_NOVEL"
    assert out["body"]["commit_sha"] == "a" * 40
    assert out["body"]["test_path"] and out["body"]["test_path"].endswith(
        "test_proposed_my_novel.py"
    )
    assert (_stub_codegen / "proposed_my_novel.py").exists()

    cached = sm.get_indicator_scaffold(ext_id, 0)
    assert cached is not None
    assert cached["status"] == "ok"
    assert cached["plugin_path"] and cached["plugin_path"].endswith(
        "proposed_my_novel.py"
    )
    assert cached["test_path"] and cached["test_path"].endswith(
        "test_proposed_my_novel.py"
    )
    assert cached["commit_sha"] == "a" * 40


def test_scaffold_test_failure_returns_test_failed_and_cleans_up(
    client: TestClient,
    sm: StateManager,
    _stub_codegen: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        indicator_scaffold,
        "run_contract_test",
        lambda _p, **_kw: (False, "FAILED test_my_thing_not_all_nan"),
    )
    tests_out = _stub_codegen.parent / "tests_out"
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "my_thing", "formula": "ema(close, period)"}],
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 422
    detail = out["body"]["detail"]
    assert detail["status"] == "test_failed"
    assert "FAILED" in (detail["test_output"] or "")
    # Both files cleaned up.
    assert not (_stub_codegen / "proposed_my_thing.py").exists()
    assert not (tests_out / "test_proposed_my_thing.py").exists()
    # Cache row records the failure with test_output for the UI.
    cached = sm.get_indicator_scaffold(ext_id, 0)
    assert cached is not None
    assert cached["status"] == "test_failed"
    assert "FAILED" in (cached["test_output"] or "")


def test_scaffold_commit_failure_surfaces_as_test_failed(
    client: TestClient,
    sm: StateManager,
    _stub_codegen: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-commit hook reject → status=test_failed with the hook output."""
    from vibe_quant.research.indicator_scaffold import ScaffoldError

    def fail_commit(*_a, **_kw):
        raise ScaffoldError("commit_failed", "pre-commit hook rejected")

    monkeypatch.setattr(indicator_scaffold, "git_commit_scaffold", fail_commit)
    tests_out = _stub_codegen.parent / "tests_out"
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "my_hooked", "formula": "ema(close, period)"}],
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 422
    detail = out["body"]["detail"]
    assert detail["status"] == "test_failed"
    assert "pre-commit" in (detail["test_output"] or "")
    assert not (_stub_codegen / "proposed_my_hooked.py").exists()
    assert not (tests_out / "test_proposed_my_hooked.py").exists()


# ---------- status=codegen_failed (slice 2) ----------


def test_scaffold_banned_import_returns_codegen_failed(
    client: TestClient,
    sm: StateManager,
    _stub_codegen: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_body = (
        "def compute_bad_a(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    import os\n"
        "    return df['close']\n"
    )
    monkeypatch.setattr(indicator_scaffold, "_run_claude_codegen", lambda _p: bad_body)

    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "bad_a", "formula": "ema(close, period)"}],
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 422
    detail = out["body"]["detail"]
    assert detail["status"] == "codegen_failed"
    assert "banned_import" in (detail["error"] or "")
    assert not (_stub_codegen / "proposed_bad_a.py").exists()

    # Cache row records the failure so the UI can show the reason.
    cached = sm.get_indicator_scaffold(ext_id, 0)
    assert cached is not None
    assert cached["status"] == "codegen_failed"
    assert "banned_import" in (cached["error"] or "")


def test_scaffold_mypy_failure_returns_codegen_failed(
    client: TestClient,
    sm: StateManager,
    _stub_codegen: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        indicator_scaffold, "run_mypy", lambda _p: (False, "incompatible types")
    )
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "bad_b", "formula": "ema(close, period)"}]
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 422
    detail = out["body"]["detail"]
    assert detail["status"] == "codegen_failed"
    assert detail["error"].startswith("mypy_fail")
    assert not (_stub_codegen / "proposed_bad_b.py").exists()


def test_scaffold_syntax_error_returns_codegen_failed(
    client: TestClient,
    sm: StateManager,
    _stub_codegen: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", lambda _p: "def x(:\n"
    )
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "bad_c", "formula": "f"}]
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 422
    detail = out["body"]["detail"]
    assert detail["status"] == "codegen_failed"
    assert detail["error"].startswith("syntax_error")


# ---------- status=already_scaffolded + force ----------


def test_scaffold_already_scaffolded_returns_cached_row(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "novel_a", "formula": "ema(close, period)"}],
    )
    # Simulate slices 2+3 having already succeeded by seeding the scaffold row.
    sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="ok",
        plugin_path="vibe_quant/dsl/plugins/proposed_novel_a.py",
        test_path="tests/unit/test_plugins/test_proposed_novel_a.py",
        commit_sha="abc123",
    )
    out = _scaffold(client, ext_id, 0)
    # Idempotent re-scaffold without force stays a 200 success.
    assert out["status_code"] == 200
    assert out["body"]["status"] == "already_scaffolded"
    assert out["body"]["plugin_path"] == "vibe_quant/dsl/plugins/proposed_novel_a.py"
    assert out["body"]["commit_sha"] == "abc123"


def test_scaffold_force_clears_cache_and_reruns_codegen(
    client: TestClient, sm: StateManager, _stub_codegen: Path
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "novel_b", "formula": "ema(close, period)"}],
    )
    sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="ok",
        plugin_path="vibe_quant/dsl/plugins/proposed_novel_b.py",
        commit_sha="def456",
    )
    # Re-stub the body so the rendered file matches the new name.
    out = _scaffold(client, ext_id, 0, force=True)
    # Force re-runs codegen + commit; cache row reflects the NEW SHA, not
    # the seeded one — slice 3 always re-stamps on force.
    assert out["body"]["status"] == "ok"
    cached = sm.get_indicator_scaffold(ext_id, 0)
    assert cached is not None
    assert cached["status"] == "ok"
    assert cached["commit_sha"] == "a" * 40  # from _stub_codegen
    assert cached["commit_sha"] != "def456"


def test_scaffold_failed_cache_does_not_short_circuit(
    client: TestClient, sm: StateManager, _stub_codegen: Path
) -> None:
    # A prior codegen_failed row should NOT count as "already scaffolded";
    # the user clicking again is a retry intent. With slice 2 wired up,
    # the retry now actually runs codegen and lands as ok.
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "novel_c", "formula": "f"}],
    )
    sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="codegen_failed",
        error="banned_import:os",
    )
    out = _scaffold(client, ext_id, 0)
    assert out["body"]["status"] == "ok"


# ---------- state manager round-trip ----------


def test_state_manager_scaffold_round_trip(sm: StateManager) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "x", "formula": "f"}]
    )
    assert sm.get_indicator_scaffold(ext_id, 0) is None

    row = sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="ok",
        plugin_path="p",
        test_path="t",
        commit_sha="abc",
    )
    assert row["status"] == "ok"
    assert row["plugin_path"] == "p"

    # Update flips status without losing key identity.
    row2 = sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="codegen_failed",
        error="boom",
    )
    assert row2["status"] == "codegen_failed"
    assert row2["error"] == "boom"
    assert row2["plugin_path"] is None  # cleared by upsert

    assert sm.delete_indicator_scaffold(ext_id, 0) is True
    assert sm.get_indicator_scaffold(ext_id, 0) is None
    assert sm.delete_indicator_scaffold(ext_id, 0) is False
