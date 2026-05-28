"""Endpoint tests for POST /api/research/indicators/{name}/promote.

bd-3p1k.3 — promote a scaffolded ``proposed_<name>.py`` plugin to a
permanent ``<name>.py``. The git commit and ``bd remember`` steps are
monkeypatched so the suite never mutates the real repo. The file
move is real (it runs against a tmp ``PLUGINS_DIR``).
"""

from __future__ import annotations

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
    s = StateManager(tmp_path / "promote.db")
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


@pytest.fixture
def _stub_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path]:
    """Pin PLUGINS_DIR + stub git_commit_promotion + bd_remember + reload_plugins."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    monkeypatch.setattr(indicator_scaffold, "PLUGINS_DIR", plugins)
    monkeypatch.setattr(
        indicator_scaffold,
        "git_commit_promotion",
        lambda _o, _n, *, name, repo_root=None: "f" * 40,
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "bd_remember_indicator",
        lambda **_kw: (True, "saved"),
    )
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


def _seed_scaffold(
    sm: StateManager, *, name: str, plugin_rel_path: str, url: str
) -> tuple[int, int]:
    """Insert item + extraction + scaffold row matching ``proposed_<name>.py``."""
    item_id = sm.create_research_item(
        source="reddit",
        external_id=f"ext-{name}",
        url=url,
        title="t",
        body="b",
        author="u/x",
        posted_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        score=10,
        extras={"comments": []},
    )
    ext_id = sm.create_extraction(
        research_item_id=item_id,
        status="parsed",
        llm_model="t",
        confidence=0.8,
        rationale="ok",
        raw_response="{}",
        dsl_yaml="name: x\n",
        parsed_dsl_json=None,
        parse_error=None,
        proposed_indicators_json="[]",
    )
    sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="ok",
        plugin_path=plugin_rel_path,
        test_path=f"tests/test_proposed_{name.lower()}.py",
        commit_sha="a" * 40,
    )
    return item_id, ext_id


def _promote(client: TestClient, name: str) -> dict:
    resp = client.post(f"/api/research/indicators/{name}/promote")
    return {"status_code": resp.status_code, "body": resp.json()}


# ---------- invalid_name ----------


def test_promote_invalid_name_returns_status_invalid_name(
    client: TestClient,
) -> None:
    out = _promote(client, "lowercase")
    assert out["status_code"] == 200
    assert out["body"]["status"] == "invalid_name"


# ---------- not_found ----------


def test_promote_not_found_when_no_proposed_file(
    client: TestClient, _stub_external: Path
) -> None:
    out = _promote(client, "GHOST")
    assert out["body"]["status"] == "not_found"


# ---------- collision ----------


def test_promote_collision_when_target_already_exists(
    client: TestClient, _stub_external: Path
) -> None:
    (_stub_external / "proposed_demo.py").write_text('"""h"""\n')
    (_stub_external / "demo.py").write_text("# already promoted\n")

    out = _promote(client, "DEMO")
    assert out["body"]["status"] == "collision"
    # Both files still on disk — no write/delete happened.
    assert (_stub_external / "proposed_demo.py").exists()
    assert (_stub_external / "demo.py").exists()


# ---------- ok (happy path) ----------


def test_promote_happy_path_renames_strips_header_and_commits(
    client: TestClient, sm: StateManager, _stub_external: Path
) -> None:
    src = (
        '"""AUTO-GENERATED FROM EXTRACTION 1 ON 2026-01-01 — review."""\n'
        "\n"
        "X = 1\n"
    )
    (_stub_external / "proposed_demo.py").write_text(src)
    _seed_scaffold(
        sm,
        name="DEMO",
        plugin_rel_path="vibe_quant/dsl/plugins/proposed_demo.py",
        url="https://reddit.com/r/x/demo",
    )

    out = _promote(client, "DEMO")
    assert out["body"]["status"] == "ok"
    assert out["body"]["name"] == "DEMO"
    assert out["body"]["commit_sha"] == "f" * 40
    assert out["body"]["bd_remember_ok"] is True
    # Path is repo-relative or absolute — just check the basename lands.
    assert out["body"]["plugin_path"].endswith("demo.py")
    assert not out["body"]["plugin_path"].endswith("proposed_demo.py")

    # File system effects: old gone, new present with header stripped.
    assert not (_stub_external / "proposed_demo.py").exists()
    new_text = (_stub_external / "demo.py").read_text()
    assert "AUTO-GENERATED" not in new_text
    assert "X = 1" in new_text


def test_promote_happy_path_invokes_bd_with_extraction_provenance(
    client: TestClient,
    sm: StateManager,
    _stub_external: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scaffold row's extraction_id + research item URL flow into bd remember."""
    (_stub_external / "proposed_demo.py").write_text('"""h"""\n')
    _, ext_id = _seed_scaffold(
        sm,
        name="DEMO",
        plugin_rel_path="vibe_quant/dsl/plugins/proposed_demo.py",
        url="https://example.com/post/123",
    )

    captured: dict[str, object] = {}

    def fake_bd(**kwargs):
        captured.update(kwargs)
        return True, "saved"

    monkeypatch.setattr(indicator_scaffold, "bd_remember_indicator", fake_bd)

    out = _promote(client, "DEMO")
    assert out["body"]["status"] == "ok"
    assert captured["name"] == "DEMO"
    assert captured["extraction_id"] == ext_id
    assert captured["source_url"] == "https://example.com/post/123"


def test_promote_happy_path_when_no_provenance_still_promotes(
    client: TestClient, _stub_external: Path
) -> None:
    """No scaffold row matching the file — promote runs, bd gets None provenance."""
    (_stub_external / "proposed_orphan.py").write_text('"""h"""\nA = 1\n')

    out = _promote(client, "ORPHAN")
    assert out["body"]["status"] == "ok"
    assert (_stub_external / "orphan.py").exists()
    assert not (_stub_external / "proposed_orphan.py").exists()


# ---------- commit_failed ----------


def test_promote_commit_failure_rolls_back_new_file(
    client: TestClient,
    sm: StateManager,
    _stub_external: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibe_quant.research.indicator_scaffold import PromoteError

    (_stub_external / "proposed_demo.py").write_text('"""h"""\n')

    def fail(*_a, **_kw):
        raise PromoteError("commit_failed", "pre-commit nope")

    monkeypatch.setattr(indicator_scaffold, "git_commit_promotion", fail)

    out = _promote(client, "DEMO")
    assert out["body"]["status"] == "commit_failed"
    assert "nope" in (out["body"].get("error") or "")
    # Rolled back: new file is gone.
    assert not (_stub_external / "demo.py").exists()


# ---------- bd remember non-fatal ----------


def test_promote_bd_failure_is_non_fatal_status_still_ok(
    client: TestClient,
    sm: StateManager,
    _stub_external: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (_stub_external / "proposed_demo.py").write_text('"""h"""\n')
    monkeypatch.setattr(
        indicator_scaffold,
        "bd_remember_indicator",
        lambda **_kw: (False, "bd not on PATH"),
    )

    out = _promote(client, "DEMO")
    assert out["body"]["status"] == "ok"
    assert out["body"]["bd_remember_ok"] is False
    assert "PATH" in (out["body"].get("bd_remember_output") or "")
