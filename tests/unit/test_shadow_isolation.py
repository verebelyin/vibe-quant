"""R44: a real shadow update, triggered by the data-update hook, touches nothing but shadow state.

Design arch-c7-2 contract row R44. A subprocess runs ``python -m vibe_quant.data update``'s
``main`` (downloads stubbed: the network is guarded) on a private cwd: a hard-linked BTCUSDT
catalog and a BTCUSDT-only archive cut from the real one. Guards: every socket connect / DNS
lookup raises, importing ``vibe_quant.paper`` / ``vibe_quant.ethereal`` /
``nautilus_trader.adapters`` raises, and ``TradingNode.__init__`` raises. A backdated test row
runs through the real engine on its catalog snapshot. The run must complete, run the engine
once, write only shadow tables, and leave the catalog and archive byte-for-byte unchanged with
no snapshot dir left behind.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_REAL_CATALOG = _REPO / "data" / "catalog" / "data"
_REAL_ARCHIVE = _REPO / "data" / "archive" / "raw_data.db"
_SYM = "BTCUSDT"
_CUT_MS = 1_756_684_800_000  # 2025-09-01: the archive slice the row's window needs

_SCRIPT = textwrap.dedent(
    r"""
    import hashlib, importlib.abc, json, socket, sqlite3, sys
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    BLOCKED = ("vibe_quant.paper", "vibe_quant.ethereal", "nautilus_trader.adapters")

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if any(name == b or name.startswith(b + ".") for b in BLOCKED):
                raise ImportError(f"isolation: {name} imported")
            return None

    sys.meta_path.insert(0, Block())

    def no_net(*a, **k):
        raise OSError("isolation: network")

    socket.socket.connect = no_net
    socket.socket.connect_ex = no_net
    socket.create_connection = no_net
    socket.getaddrinfo = no_net

    from nautilus_trader.live.node import TradingNode

    def no_node(self, *a, **k):
        raise RuntimeError("isolation: TradingNode constructed")

    TradingNode.__init__ = no_node

    from vibe_quant.data import ingest
    from vibe_quant.data.catalog import WRITE_LOCK_NAME, CatalogManager
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.shadow import tracker

    DSL = {
        "name": "shadow_iso_rsi",
        "timeframe": "4h",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 45"], "short": ["rsi > 60"]},
        "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 40"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }
    db = Path(sys.argv[1])
    now = datetime.now(UTC)
    tail = CatalogManager(Path("data/catalog")).get_bar_date_range("BTCUSDT", "4h")[1].date()
    first = tail - timedelta(days=8)  # F: about a week of forward days to decide
    state = StateManager(db)
    sid = tracker.enrol(
        state,
        tracker.EnrolRequest(
            name="shadow_iso_rsi",
            source_kind="strategy",
            dsl_json=json.dumps(DSL),
            symbols=["BTCUSDT"],
            parameters={},
            pre_registration="Isolation probe: a real update must not touch paper or live.",
            max_dd_limit=0.5,
            warmup_days=30,
            backdate_days=(now.date() + timedelta(days=1) - first).days,
        ),
        now=now,
    )
    state.close()

    def table_digest(path, skip_prefix=None):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        out = {}
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            if skip_prefix and name.startswith(skip_prefix):
                continue
            h = hashlib.sha256()
            for row in conn.execute(f'SELECT * FROM "{name}" ORDER BY rowid'):
                h.update(repr(row).encode())
            out[name] = h.hexdigest()
        conn.close()
        return out

    def tree(root):  # the empty flock file catalog_write_lock creates is no catalog data
        return {
            str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*"))
            if p.name != WRITE_LOCK_NAME
        }

    before = {
        "state": table_digest(db, "shadow_"),
        "archive": table_digest("data/archive/raw_data.db"),
        "catalog": tree(Path("data/catalog")),
    }
    summaries = []
    real_update_all = tracker.update_all

    def spy(*a, **k):
        summaries.append(real_update_all(*a, **k))
        return summaries[-1]

    tracker.update_all = spy
    ingest.update_all = lambda symbols=None, verbose=True: {
        "BTCUSDT": {"new_klines": 0, "failed_ranges": []}
    }
    rc = ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)])
    after = {
        "state": table_digest(db, "shadow_"),
        "archive": table_digest("data/archive/raw_data.db"),
        "catalog": tree(Path("data/catalog")),
    }
    state = StateManager(db)
    row = next(r for r in state.list_shadow_strategies() if r["id"] == sid)
    print(json.dumps({
        "rc": rc,
        "runs": [s.runs for s in summaries],
        "rows": [list(r) for s in summaries for r in s.rows],
        "state": row["state"],
        "days": len(state.get_shadow_days(sid)),
        "unchanged": {k: before[k] == after[k] for k in before},
        "catalog_diff": sorted(
            k for k in set(before["catalog"]) | set(after["catalog"])
            if before["catalog"].get(k) != after["catalog"].get(k)
        ),
        "lock_file": (Path("data/catalog") / WRITE_LOCK_NAME).stat().st_size,
        "snapshots_left": sorted(p.name for p in Path("data").glob(".shadow-snap-*")),
        "blocked_loaded": sorted(m for m in sys.modules if m.startswith(BLOCKED)),
    }))
    """
)


def _link_catalog(dst: Path) -> None:
    for src in [
        *(_REAL_CATALOG / "bar").glob(f"{_SYM}-PERP.BINANCE-*-LAST-EXTERNAL"),
        _REAL_CATALOG / "crypto_perpetual" / f"{_SYM}-PERP.BINANCE",
    ]:
        for f in src.rglob("*"):
            if f.is_file():
                target = dst / "data" / f.relative_to(_REAL_CATALOG)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(f, target)


def _cut_archive(dst: Path) -> None:
    from vibe_quant.data.archive import ARCHIVE_SCHEMA

    real = sqlite3.connect(f"file:{_REAL_ARCHIVE}?mode=ro", uri=True)
    conn = sqlite3.connect(dst)
    conn.executescript(ARCHIVE_SCHEMA)
    queries = {
        "raw_klines": (
            "SELECT * FROM raw_klines WHERE symbol = ? AND interval = '1m' AND open_time >= ?",
            (_SYM, _CUT_MS),
        ),
        "raw_funding_rates": ("SELECT * FROM raw_funding_rates WHERE symbol = ?", (_SYM,)),
        "raw_funding_intervals": ("SELECT * FROM raw_funding_intervals WHERE symbol = ?", (_SYM,)),
    }
    for table, (sql, params) in queries.items():
        rows = real.execute(sql, params).fetchall()
        marks = ", ".join("?" * len(rows[0]))
        conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)  # noqa: S608
    conn.commit()
    conn.close()
    real.close()


@pytest.mark.slow(
    reason="R44 owner: a real engine run in a guarded subprocess (NT import + enrol + "
    "~5 weeks of 4h BTC) is the only proof the hook never reaches paper/live code"
)
@pytest.mark.skipif(
    not (_REAL_ARCHIVE.exists() and (_REAL_CATALOG / "bar").is_dir()),
    reason="needs the real BTCUSDT catalog and archive",
)
def test_real_update_touches_only_shadow_state(tmp_path: Path) -> None:
    (tmp_path / "data" / "archive").mkdir(parents=True)
    _link_catalog(tmp_path / "data" / "catalog")
    _cut_archive(tmp_path / "data" / "archive" / "raw_data.db")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TELEGRAM_", "HTTP", "HTTPS"))}
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path / "state.db")],
        cwd=tmp_path,
        env={**env, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["rc"] == 0
    assert res["runs"] == [1], (res, out.stderr[-4000:])  # the engine really ran, guarded
    assert not str(res["rows"][0][2]).startswith("update error"), res
    assert res["state"] in ("tracking", "stalled", "futile", "passed"), res
    assert res["unchanged"] == {"state": True, "archive": True, "catalog": True}, res[
        "catalog_diff"
    ]
    assert res["lock_file"] == 0
    assert res["snapshots_left"] == []
    assert res["blocked_loaded"] == []
