"""The state-DB-free validation variant seam (bankroll report 2gyk1.3, T2a/T2b).

``run_variant`` replays a completed validation run's recorded inputs through ``run()``'s own
helpers and engine path, on a catalog clone and an archive snapshot it is handed, without
the state DB or the event log. With no overrides it must equal ``run()`` field for field,
and it refuses (before any engine) when its resolved inputs differ from the parent's
recorded ``notes["validation_input"]``. Real-engine cases use clones of the real BTCUSDT /
ETHUSDT catalog bars (skipped when absent) and a synthetic funded archive.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.validation.results import TradeRecord, ValidationResult
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError
from vibe_quant.validation.variant import VariantInput, VariantInputsMismatch, run_variant

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
_BARS = CATALOG / "data" / "bar"
_NEEDED = [
    f"{s}-PERP.BINANCE-{tf}-LAST-EXTERNAL"
    for s in ("BTCUSDT", "ETHUSDT")
    for tf in ("4-HOUR", "1-MINUTE")
]
needs_catalog = pytest.mark.skipif(
    not all((_BARS / d).is_dir() for d in _NEEDED),
    reason="needs the BTCUSDT/ETHUSDT 4h and 1m catalog",
)

START, END = "2025-03-01", "2025-04-01"
_DAY_MS = 86_400_000
_START_MS = int(datetime(2025, 3, 1, tzinfo=UTC).timestamp() * 1000)
_END_MS = int(datetime(2025, 4, 1, tzinfo=UTC).timestamp() * 1000)

RSI_DSL: dict[str, object] = {
    "name": "variant_probe_rsi",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 45"], "short": ["rsi > 60"]},
    "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 40"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}
# The same entries gated by a required-aux (taker) indicator that is always true when
# defined: the run must read the archive's 1m klines.
TAKER_DSL: dict[str, object] = {
    **RSI_DSL,
    "name": "variant_probe_taker",
    "indicators": {
        "rsi": {"type": "RSI", "period": 14},
        "tk": {"type": "TAKER_IMBALANCE", "period": 10},
    },
    "entry_conditions": {"long": ["rsi < 45", "tk > -2"], "short": ["rsi > 60", "tk > -2"]},
}

IDENTITY_KEYS = {
    "v", "strategy_name", "dsl_sha256", "symbols", "timeframe", "start_date", "end_date",
    "detail_timeframe", "latency_preset", "venue", "launch", "entry_order",
}


# --- fixtures -----------------------------------------------------------------


def clone_catalog(dst: Path, symbols: Sequence[str]) -> Path:
    """APFS clone (``cp -c``) of the symbols' instruments and 4h/1m LAST bars."""
    src = CATALOG.resolve()
    for symbol in symbols:
        inst = f"{symbol}-PERP.BINANCE"
        pairs = [(src / "data" / "crypto_perpetual" / inst, dst / "data" / "crypto_perpetual" / inst)]
        pairs += [
            (src / "data" / "bar" / f"{inst}-{tf}-LAST-EXTERNAL",
             dst / "data" / "bar" / f"{inst}-{tf}-LAST-EXTERNAL")
            for tf in ("4-HOUR", "1-MINUTE")
        ]
        for s, d in pairs:
            d.parent.mkdir(parents=True, exist_ok=True)
            done = subprocess.run(["cp", "-c", "-R", str(s), str(d)], capture_output=True)
            if done.returncode != 0:
                pytest.skip(f"cp -c clone unsupported here: {done.stderr!r}")
    return dst


def funded_archive(path: Path, symbols: Sequence[str], *, taker: bool = False) -> Path:
    """Nonzero 8h funding (+ Vision intervals) over the window; optional 1m taker klines."""
    from vibe_quant.data.archive import RawDataArchive

    archive = RawDataArchive(path)
    try:
        first = _START_MS - 3 * _DAY_MS
        settlements = range(first, _END_MS + 2 * _DAY_MS, 8 * 3_600_000)
        for n, symbol in enumerate(symbols):
            archive.insert_funding_rates(
                symbol,
                [(t, (0.0003 if i % 2 else -0.0001) * (n + 1), None) for i, t in enumerate(settlements)],
                "test",
            )
            archive.insert_funding_intervals(symbol, [(t, 8) for t in settlements], "test")
            if taker:
                archive.conn.executemany(
                    "INSERT INTO raw_klines (symbol, interval, open_time, open, high, low, close, "
                    "volume, close_time, taker_buy_volume, source) "
                    "VALUES (?, '1m', ?, 1, 1, 1, 1, 10, ?, ?, 'test')",
                    [
                        (symbol, t, t + 59_999, 4 + (t // 60_000) % 3)
                        for t in range(_START_MS - 2 * _DAY_MS, _END_MS + _DAY_MS, 60_000)
                    ],
                )
                archive.conn.commit()
    finally:
        archive.close()
    return path


def sqlite_backup(src: Path, dst: Path) -> Path:
    """Consistent copy from a read-only source connection."""
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    target = sqlite3.connect(str(dst))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return dst


@dataclasses.dataclass
class Parent:
    """A completed ``run()`` validation and everything its variants need."""

    db: Path
    logs: Path
    catalog: Path
    archive: Path
    run_id: int
    result: ValidationResult
    inp: VariantInput


def make_parent(
    root: Path,
    dsl: dict[str, object],
    symbols: list[str],
    *,
    taker: bool = False,
    parameters: dict[str, object] | None = None,
) -> Parent:
    root.mkdir(parents=True, exist_ok=True)
    db = root / "state.db"
    state = StateManager(db)
    sid = state.create_strategy(name=str(dsl["name"]), dsl_config=dsl)
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=symbols, timeframe="4h",
        start_date=START, end_date=END, parameters=parameters or {},
    )
    state.close()
    catalog = clone_catalog(root / "catalog0", symbols)
    archive = funded_archive(root / "archive.db", symbols, taker=taker)
    runner = ValidationRunner(
        db_path=db, logs_path=root / "logs", catalog_path=catalog, funding_archive_path=archive
    )
    try:
        result = runner.run(run_id=run_id)
    finally:
        runner.close()
    return Parent(db, root / "logs", catalog, archive, run_id, result, variant_input(db, run_id))


def variant_input(db: Path, run_id: int) -> VariantInput:
    """The bankroll admission's snapshot of a parent: strategy, run row, recorded identity."""
    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
        assert run is not None
        strategy = state.get_strategy(int(run["strategy_id"]))
        assert strategy is not None
        stored = state.get_backtest_result(run_id)
        assert stored is not None
    finally:
        state.close()
    return VariantInput(
        strategy_name=str(strategy["name"]),
        dsl_json=json.dumps(strategy["dsl_config"]),
        run_config_json=json.dumps(run),
        parent_identity=json.loads(str(stored["notes"]))["validation_input"],
    )


_NOT_COMPARED = {"run_id", "execution_time_seconds", "notes", "mtm_probe"}


def comparable(res: ValidationResult) -> str:
    """Every result field but ids/timing/notes, with the recorded identity (NaN == NaN)."""
    out = {f.name: getattr(res, f.name) for f in dataclasses.fields(res) if f.name not in _NOT_COMPARED}
    out["validation_input"] = res.notes.get("validation_input")
    return json.dumps(out, default=repr, sort_keys=True)


def tree_state(*paths: Path) -> dict[str, tuple[int, str]]:
    """relpath -> (size, sha256) of every file under each path (a file is itself)."""
    out: dict[str, tuple[int, str]] = {}
    for base in paths:
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for f in files:
            out[str(f)] = (f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest())
    return out


@pytest.fixture(scope="module")
def single_parent(tmp_path_factory: pytest.TempPathFactory) -> Parent:
    if not all((_BARS / d).is_dir() for d in _NEEDED):
        pytest.skip("needs the BTCUSDT/ETHUSDT 4h and 1m catalog")
    return make_parent(tmp_path_factory.mktemp("single"), TAKER_DSL, ["BTCUSDT"], taker=True)


# --- C16: state-DB-free, cold and funded, in a fresh process ------------------

_CHILD = textwrap.dedent(
    """
    import json, sqlite3, sys
    from pathlib import Path

    spec = json.loads(Path(sys.argv[1]).read_text())
    opened = []
    _connect = sqlite3.connect
    def recording_connect(database, *args, **kwargs):
        opened.append(str(database))
        return _connect(database, *args, **kwargs)
    sqlite3.connect = recording_connect

    import vibe_quant.db.connection as connection
    _get = connection.get_connection
    def recording_get(path, *args, **kwargs):
        opened.append("get_connection:" + str(path))
        return _get(path, *args, **kwargs)
    connection.get_connection = recording_get

    # Any fallback to a default would land on the parent's own archive / catalog.
    import vibe_quant.data.archive as archive_mod
    import vibe_quant.data.catalog as catalog_mod
    archive_mod.DEFAULT_ARCHIVE_PATH = Path(spec["decoy_archive"])
    catalog_mod.DEFAULT_CATALOG_PATH = Path(spec["decoy_catalog"])

    import vibe_quant.validation.runner as runner_mod
    def no_db(*a, **k):
        raise AssertionError("variant constructed a StateManager / EventWriter")
    runner_mod.StateManager = no_db
    runner_mod.EventWriter = no_db

    from vibe_quant.validation.runner import ValidationRunnerError
    from vibe_quant.validation.variant import VariantInput, run_variant

    inp = VariantInput(**spec["input"])
    kw = {"catalog_path": Path(spec["catalog"]), "archive_path": Path(spec["archive"]), "mtm_probe": False}
    res = run_variant(inp, {}, **kw)
    try:
        run_variant(inp, {"rsi_perid": 3}, **kw)
        failure = None
    except ValidationRunnerError as exc:
        failure = str(exc)
    out = {
        "opened": opened,
        "failure": failure,
        "result": {k: v for k, v in res.__dict__.items() if k not in ("trades", "notes")},
        "trades": [t.to_dict() for t in res.trades],
        "identity": res.notes["validation_input"],
    }
    Path(spec["out"]).write_text(json.dumps(out, default=str))
    """
)


@pytest.mark.slow(reason="fresh subprocess + real NT validation run() and variant, ~6 s")
@needs_catalog
def test_variant_state_db_free_cold_funded(single_parent: Parent, tmp_path: Path) -> None:
    parent = single_parent
    snapshot = sqlite_backup(parent.archive, tmp_path / "archive-snapshot.db")
    clone = clone_catalog(tmp_path / "w0" / "catalog", ["BTCUSDT"])
    work = tmp_path / "cwd"
    work.mkdir()
    spec = {
        "input": dataclasses.asdict(parent.inp),
        "catalog": str(clone),
        "archive": str(snapshot),
        "decoy_archive": str(parent.archive),
        "decoy_catalog": str(parent.catalog),
        "out": str(tmp_path / "out.json"),
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (tmp_path / "child.py").write_text(_CHILD)
    db_files = [p for p in (parent.db, parent.db.with_name("state.db-wal")) if p.exists()]
    before = tree_state(*db_files, parent.logs, parent.archive, parent.catalog)
    listing = sorted(p.name for p in parent.db.parent.iterdir())

    env = {**os.environ, "PYTHONPATH": str(REPO)}
    done = subprocess.run(
        [sys.executable, str(tmp_path / "child.py"), str(tmp_path / "spec.json")],
        cwd=work, env=env, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads((tmp_path / "out.json").read_text())

    # The only database the variant process opened is the archive snapshot.
    resolved = {Path(p.removeprefix("get_connection:").removeprefix("file:").split("?")[0]).resolve()
                for p in out["opened"]}
    assert resolved == {snapshot.resolve()}, out["opened"]
    assert any(p.startswith("get_connection:") for p in out["opened"])  # funding read
    assert any(p.startswith("file:") for p in out["opened"])  # taker klines read
    # Nothing of the parent moved: state DB (+ wal), events, original archive, catalog.
    assert tree_state(*db_files, parent.logs, parent.archive, parent.catalog) == before
    assert sorted(p.name for p in parent.db.parent.iterdir()) == listing
    assert list(work.iterdir()) == []
    assert out["failure"] is not None and "rsi_perid" in out["failure"]

    # The funded variant equals ordinary run() validation on the same inputs.
    want = parent.result
    assert want.total_trades > 0
    assert want.total_funding != 0.0
    skip = _NOT_COMPARED | {"trades"}
    expected = {k: v for k, v in want.__dict__.items() if k not in skip}
    got = {k: v for k, v in out["result"].items() if k not in skip}
    assert got == json.loads(json.dumps(expected, default=str))
    assert out["trades"] == [t.to_dict() for t in want.trades]
    assert out["identity"] == want.notes["validation_input"]


# --- C15: variant == run() --------------------------------------------------


@pytest.mark.slow(reason="two-symbol real NT validation run() + two variants, ~9 s")
@needs_catalog
def test_variant_equals_run(single_parent: Parent, tmp_path: Path) -> None:
    parents = [single_parent, make_parent(tmp_path / "multi", RSI_DSL, ["BTCUSDT", "ETHUSDT"])]
    for i, parent in enumerate(parents):
        symbols = json.loads(parent.inp.run_config_json)["symbols"]
        got = run_variant(
            parent.inp, {},
            catalog_path=clone_catalog(tmp_path / f"w{i}" / "catalog", symbols),
            archive_path=sqlite_backup(parent.archive, tmp_path / f"snap{i}.db"),
            mtm_probe=False,
        )
        want = parent.result
        assert want.total_trades > 0 and want.total_funding != 0.0
        assert {t.symbol for t in want.trades} == {f"{s}-PERP.BINANCE" for s in symbols}
        assert comparable(got) == comparable(want)
        assert [_trade_key(t) for t in got.trades] == [_trade_key(t) for t in want.trades]
        assert got.engine_force_stopped is False


def _trade_key(t: TradeRecord) -> tuple[object, ...]:
    return (t.symbol, t.entry_time, t.exit_time, t.entry_price, t.exit_price, t.quantity, t.net_pnl)


# --- C10a: NT force stop is read while the engine is alive ------------------


@pytest.fixture
def force_stop_after_profit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every compiled strategy raises AccountError on its first bar after a profitable
    trade closed, while flat: NT force-stops the one-shot run with feed remaining."""
    from nautilus_trader.accounting.error import AccountError

    from vibe_quant.dsl.compiler import StrategyCompiler

    original = StrategyCompiler.compile_to_module

    def compile_to_module(self: StrategyCompiler, dsl: Any) -> Any:
        module = original(self, dsl)
        cls_name = "".join(w.capitalize() for w in dsl.name.split("_")) + "Strategy"
        cls = getattr(module, cls_name)
        on_bar = cls.on_bar

        def halting_on_bar(strategy: Any, bar: Any) -> None:
            on_bar(strategy, bar)
            closed = strategy.cache.positions_closed()
            if (
                not strategy.cache.positions_open()
                and any(p.realized_pnl.as_double() > 0 for p in closed)
            ):
                raise AccountError("synthetic force stop")

        monkeypatch.setattr(cls, "on_bar", halting_on_bar)
        return module

    monkeypatch.setattr(StrategyCompiler, "compile_to_module", compile_to_module)


@needs_catalog
def test_force_stop_flag(
    single_parent: Parent, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    parent = single_parent
    kw: dict[str, Any] = {
        "archive_path": sqlite_backup(parent.archive, tmp_path / "snap.db"), "mtm_probe": False
    }
    request.getfixturevalue("force_stop_after_profit")
    stopped = run_variant(
        parent.inp, {}, catalog_path=clone_catalog(tmp_path / "a", ["BTCUSDT"]), **kw
    )
    assert stopped.engine_force_stopped is True
    assert 0 < stopped.total_trades < parent.result.total_trades  # feed remained
    assert max(t.net_pnl for t in stopped.trades) > 0
    assert parent.result.engine_force_stopped is False


# A position held to the end of the run is marked there: a normal end, not a force stop.
HOLD_DSL: dict[str, object] = {
    "name": "variant_probe_hold",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi > 0"]},
    "stop_loss": {"type": "fixed_pct", "percent": 50.0},
    "take_profit": {"type": "fixed_pct", "percent": 100.0},
}


@pytest.mark.slow(reason="real NT validation run() + variant, ~3 s")
@needs_catalog
def test_terminal_mark_is_not_a_force_stop(tmp_path: Path) -> None:
    parent = make_parent(tmp_path / "hold", HOLD_DSL, ["BTCUSDT"])
    assert parent.result.total_trades == 1
    assert parent.result.engine_force_stopped is False
    got = run_variant(
        parent.inp, {},
        catalog_path=clone_catalog(tmp_path / "w", ["BTCUSDT"]),
        archive_path=sqlite_backup(parent.archive, tmp_path / "snap.db"), mtm_probe=False,
    )
    assert got.engine_force_stopped is False
    assert comparable(got) == comparable(parent.result)


# --- C18: invalid overrides fail like run(), before any engine, writing nothing -----


@pytest.mark.parametrize(
    "override",
    [
        {"leverage": 0},
        {"risk_per_trade": 0.6},
        {"max_position_pct": 50.0, "liquidation_mode": "bogus"},
    ],
    ids=["leverage-0", "risk-0.6", "cap-and-mode"],
)
@needs_catalog
def test_variant_invalid_override(
    single_parent: Parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
) -> None:
    import nautilus_trader.backtest.node as nt_node

    parent = single_parent
    # run() on a DB copy, the row's parameters carrying the override.
    copy = sqlite_backup(parent.db, tmp_path / "copy.db")
    run_row = json.loads(parent.inp.run_config_json)
    state = StateManager(copy)
    rid = state.create_backtest_run(
        strategy_id=int(run_row["strategy_id"]), run_mode="validation", symbols=["BTCUSDT"],
        timeframe="4h", start_date=START, end_date=END,
        parameters={**run_row["parameters"], **override},
    )
    state.close()
    runner = ValidationRunner(
        db_path=copy, logs_path=tmp_path / "logs", catalog_path=parent.catalog,
        funding_archive_path=parent.archive,
    )
    with pytest.raises(ValidationRunnerError) as run_err:
        runner.run(run_id=rid)
    runner.close()
    state = StateManager(copy)
    row = state.get_backtest_run(rid)
    state.close()
    assert row is not None and row["status"] == "failed"
    assert row["error_message"] == str(run_err.value)

    built: list[object] = []
    monkeypatch.setattr(nt_node, "BacktestNode", lambda *a, **k: built.append(a))
    clone = clone_catalog(tmp_path / "w", ["BTCUSDT"])
    snap = sqlite_backup(parent.archive, tmp_path / "snap.db")
    db_files = [p for p in (parent.db, parent.db.with_name("state.db-wal")) if p.exists()]
    before = tree_state(*db_files, parent.logs, clone, snap)
    with pytest.raises(ValidationRunnerError) as variant_err:
        run_variant(parent.inp, override, catalog_path=clone, archive_path=snap, mtm_probe=False)
    assert type(variant_err.value) is ValidationRunnerError
    assert str(variant_err.value) == str(run_err.value)
    assert built == []
    assert tree_state(*db_files, parent.logs, clone, snap) == before


# --- stubbed-engine cases: identity (C20a) and the maker parent (C19c) -------

_COVERED = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 3, 17, tzinfo=UTC))


class _Coverage:
    """Catalog coverage the window clamp sees: every symbol and interval, same span."""

    span: tuple[datetime, datetime] = _COVERED


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[list[dict[str, Any]]]:
    """No engine: ``_run_backtest`` records its call and returns a 3-trade result."""
    calls: list[dict[str, Any]] = []
    _Coverage.span = _COVERED

    def date_range(self: CatalogManager, symbol: str, interval: str) -> tuple[datetime, datetime]:
        return _Coverage.span

    def fake_run_backtest(self: ValidationRunner, **kw: Any) -> ValidationResult:
        calls.append(kw)
        notes: dict[str, object] = {}
        if kw.get("entry_order_override") is None and "post_only" in json.dumps(
            kw["run_config"].get("parameters")
        ):
            notes["maker_entries"] = {"orders": 3}
        return ValidationResult(run_id=0, strategy_name="s", total_trades=3, notes=notes)

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)
    monkeypatch.setattr(ValidationRunner, "_run_backtest", fake_run_backtest)
    yield calls


def stub_parent(
    tmp_path: Path,
    dsl: dict[str, object],
    *,
    symbols: list[str] | None = None,
    timeframe: str = "4h",
    parameters: dict[str, object] | None = None,
    latency_preset: str | None = None,
    detail_timeframe: str | None = None,
    name: str = "p",
) -> tuple[ValidationResult, VariantInput]:
    db = tmp_path / f"{name}.db"
    state = StateManager(db)
    sid = state.create_strategy(name=str(dsl["name"]), dsl_config=dsl)
    rid = state.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=symbols or ["BTCUSDT"],
        timeframe=timeframe, start_date="2025-01-01", end_date="2025-02-01",
        parameters=parameters or {},
    )
    state.close()
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", catalog_path=tmp_path)
    try:
        result = runner.run(rid, latency_preset=latency_preset, detail_timeframe=detail_timeframe)
    finally:
        runner.close()
    return result, variant_input(db, rid)


def _variant(inp: VariantInput, tmp_path: Path, overrides: dict[str, object] | None = None) -> ValidationResult:
    return run_variant(
        inp, overrides or {}, catalog_path=tmp_path / "clone", archive_path=tmp_path / "snap.db",
        mtm_probe=False,
    )


def _canonical_sha(dsl: dict[str, object]) -> str:
    dumped = validate_strategy_dict(dsl).model_dump(mode="json")
    blob = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def test_input_identity(stub_engine: list[dict[str, Any]], tmp_path: Path) -> None:
    result, inp = stub_parent(tmp_path, RSI_DSL, parameters={"stop_loss_percent": 2.5})
    identity = result.notes["validation_input"]
    assert isinstance(identity, dict)
    assert set(identity) == IDENTITY_KEYS
    assert json.loads(json.dumps(identity)) == identity
    assert {k: identity[k] for k in IDENTITY_KEYS - {"venue"}} == {
        "v": 1,
        "strategy_name": "variant_probe_rsi",
        "dsl_sha256": _canonical_sha(RSI_DSL),
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "detail_timeframe": "1m",
        "latency_preset": "cloud",
        "launch": '{"stop_loss_percent":2.5}',
        "entry_order": "market",
    }
    assert '"starting_balance_usdt":1000,' in str(identity["venue"])
    assert dict(inp.parent_identity) == identity

    # A second, unrelated run: every forwarded value follows its own inputs.
    other_dsl = {**RSI_DSL, "name": "variant_probe_other", "timeframe": "1h"}
    other, _ = stub_parent(
        tmp_path, other_dsl, symbols=["ETHUSDT", "BTCUSDT"], timeframe="1h",
        parameters={"initial_balance": 5000}, latency_preset="retail",
        detail_timeframe="5m", name="other",
    )
    other_id = other.notes["validation_input"]
    assert isinstance(other_id, dict)
    assert {k: other_id[k] for k in IDENTITY_KEYS - {"venue"}} == {
        "v": 1,
        "strategy_name": "variant_probe_other",
        "dsl_sha256": _canonical_sha(other_dsl),
        "symbols": ["ETHUSDT", "BTCUSDT"],
        "timeframe": "1h",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "detail_timeframe": "5m",
        "latency_preset": "retail",
        "launch": '{"initial_balance":5000}',
        "entry_order": "market",
    }
    assert '"starting_balance_usdt":5000' in str(other_id["venue"])

    # A variant with no overrides reproduces the recorded identity exactly.
    stub_engine.clear()
    replay = _variant(inp, tmp_path)
    assert replay.notes["validation_input"] == identity
    (call,) = stub_engine
    assert call["writer"] is None and call["write_trade_events"] is False


def test_identity_flips_name_the_changed_input(
    stub_engine: list[dict[str, Any]], tmp_path: Path
) -> None:
    _, inp = stub_parent(tmp_path, RSI_DSL)
    stub_engine.clear()

    def refused(variant: VariantInput) -> tuple[str, ...]:
        with pytest.raises(VariantInputsMismatch) as err:
            _variant(variant, tmp_path)
        return err.value.keys

    edited = {**RSI_DSL, "indicators": {"rsi": {"type": "RSI", "period": 15}}}
    assert refused(dataclasses.replace(inp, dsl_json=json.dumps(edited))) == ("dsl_sha256",)
    # Catalog coverage moved: the clamp gives another end date.
    _Coverage.span = (_COVERED[0], datetime(2025, 1, 20, 12, tzinfo=UTC))
    assert refused(inp) == ("end_date",)
    _Coverage.span = _COVERED
    assert stub_engine == []  # no engine for a refused replay

    # Parents run with a CLI-only detail / latency override record it ...
    detail, detail_inp = stub_parent(tmp_path, RSI_DSL, detail_timeframe="15m", name="d")
    latency, latency_inp = stub_parent(tmp_path, RSI_DSL, latency_preset="retail", name="l")
    base = dict(inp.parent_identity)
    detail_id = detail.notes["validation_input"]
    latency_id = latency.notes["validation_input"]
    assert isinstance(detail_id, dict) and isinstance(latency_id, dict)
    assert {k for k in base if base[k] != detail_id[k]} == {"detail_timeframe"}
    assert "latency_preset" in {k for k in base if base[k] != latency_id[k]}
    # ... and the variant replays with it, not with the stored row's default.
    stub_engine.clear()
    assert _variant(detail_inp, tmp_path).notes["validation_input"] == detail_id
    assert stub_engine[0]["detail_timeframe"] == "15m"
    assert _variant(latency_inp, tmp_path).notes["validation_input"] == latency_id


def test_override_variant_skips_the_identity_check(
    stub_engine: list[dict[str, Any]], tmp_path: Path
) -> None:
    _, inp = stub_parent(tmp_path, RSI_DSL, parameters={"stop_loss_percent": 2.5})
    stub_engine.clear()
    got = _variant(inp, tmp_path, {"max_position_pct": 2.0})
    params = stub_engine[0]["run_config"]["parameters"]
    assert params == {"stop_loss_percent": 2.5, "max_position_pct": 2.0}
    assert json.loads(inp.run_config_json)["parameters"] == {"stop_loss_percent": 2.5}
    launch = got.notes["validation_input"]
    assert isinstance(launch, dict)
    assert launch["launch"] == '{"max_position_pct":2.0,"stop_loss_percent":2.5}'


def test_variant_maker(stub_engine: list[dict[str, Any]], tmp_path: Path) -> None:
    result, inp = stub_parent(tmp_path, RSI_DSL, parameters={"entry_order": "post_only"})
    # run() ran the primary and its market twin ...
    assert [c.get("entry_order_override") for c in stub_engine] == [None, "market"]
    maker = result.notes["maker_entries"]
    assert isinstance(maker, dict) and "market_twin" in maker
    stub_engine.clear()
    # ... the variant runs the primary only.
    got = _variant(inp, tmp_path)
    assert [c.get("entry_order_override") for c in stub_engine] == [None]
    assert got.notes["maker_entries"] == {"orders": 3}
    identity = got.notes["validation_input"]
    assert isinstance(identity, dict) and identity["entry_order"] == "post_only"


def test_detached_runner_has_no_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vibe_quant.validation.runner as runner_mod

    def no_state(*a: object, **k: object) -> None:
        raise AssertionError("StateManager constructed")

    monkeypatch.setattr(runner_mod, "StateManager", no_state)
    runner = ValidationRunner.detached(tmp_path / "cat", tmp_path / "a.db")
    assert runner._catalog_path == tmp_path / "cat"
    assert runner._funding_archive_path == tmp_path / "a.db"
    with pytest.raises(ValidationRunnerError, match="variant runner has no database"):
        runner.run(1)
    assert not (tmp_path / "cat").exists() and not (tmp_path / "a.db").exists()


# --- T2b: the MTM probe (C19, C19b) -----------------------------------------

_H = 3_600_000_000_000
_DAY_NS = 86_400_000_000_000


def _engine(positions: list[Any], closes: dict[str, list[tuple[int, float, float]]]) -> Any:
    """Fake engine whose 1m LAST execution bars are ``(ts, close, low)`` per instrument."""
    from types import SimpleNamespace

    from tests.unit.fake_engine import fake_engine
    from tests.unit.test_validation_liquidation import _ExecBarType

    bars: dict[str, list[SimpleNamespace]] = {}
    for iid, rows in closes.items():
        bar_type = _ExecBarType(iid)
        bars[str(bar_type)] = [
            SimpleNamespace(bar_type=bar_type, ts_init=ts, volume=0.0, open=c, high=max(c, lo),
                            low=lo, close=c)
            for ts, c, lo in rows
        ]
    return fake_engine(positions, bars_by_type=bars)


def _extract(
    engine: Any,
    start: str,
    end: str,
    *,
    probe: bool = True,
    mode: str = "off",
    funding: Any = None,
    venue: Any = None,
) -> ValidationResult:
    from tests.unit.test_validation_liquidation import OFF_SLIPPAGE, START_BAL, VENUE
    from vibe_quant.validation.extraction import extract_trades

    res = ValidationResult(starting_balance=START_BAL)
    extract_trades(
        res, engine, VENUE if venue is None else venue,  # type: ignore[arg-type]
        funding_calculator=funding, run_start_date=start, run_end_date=end,
        execution_timeframe="1m", catalog_path="unused", liquidation_mode=mode,
        slippage=OFF_SLIPPAGE, cost_ledger=None, mtm_probe=probe,
    )
    return res


def _equity(res: ValidationResult) -> list[float]:
    """Daily close equity implied by the probe's daily returns."""
    assert res.mtm_probe is not None and res.mtm_probe.daily is not None
    out, bal = [], res.starting_balance
    for r in res.mtm_probe.daily.values:
        bal *= 1.0 + r
        out.append(bal)
    return out


def _spy_paths(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    import vibe_quant.validation.extraction as ex

    real = ex.mark_to_market_paths
    seen: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        paths = real(*args, **kwargs)
        seen.append(paths)
        return paths

    monkeypatch.setattr(ex, "mark_to_market_paths", spy)
    return seen


def test_mtm_probe_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_validation_liquidation import (
        END,
        START,
        START_BAL,
        crash_marks,
        engine_for,
        long_btc,
    )
    from vibe_quant.validation.extraction import date_to_ns, mtm_daily_returns

    seen = _spy_paths(monkeypatch)
    mk = {"BTCUSDT-PERP.BINANCE": crash_marks(to=39000.0, at=20)}
    off = _extract(engine_for([long_btc(sell_bar=40)], mk), START, END, probe=False)
    assert off.mtm_probe is None and len(seen) == 1
    on = _extract(engine_for([long_btc(sell_bar=40)], mk), START, END)
    assert len(seen) == 2  # the probe reuses the drawdown's one path build
    probe = on.mtm_probe
    assert probe is not None and (probe.paths, probe.ledger) == ("ok", "ordinary")
    assert on.max_drawdown == off.max_drawdown == probe.dd
    assert probe.dd == pytest.approx(1020.0 / START_BAL)  # 1 BTC 40000 -> 39000, entry fee 20
    start_ns, end_ns = date_to_ns(START), date_to_ns(END)
    assert start_ns is not None and end_ns is not None
    assert probe.daily == mtm_daily_returns(START_BAL, seen[1], start_ns, end_ns)
    assert (probe.bust_ts, probe.path_divergence) == (None, False)


@pytest.mark.slow(reason="real NT validation variant with the probe on, ~3 s")
@needs_catalog
def test_variant_mtm_probe_matches_the_parent(single_parent: Parent, tmp_path: Path) -> None:
    parent = single_parent
    got = run_variant(
        parent.inp, {},
        catalog_path=clone_catalog(tmp_path / "w", ["BTCUSDT"]),
        archive_path=sqlite_backup(parent.archive, tmp_path / "snap.db"),
        mtm_probe=True,
    )
    assert comparable(got) == comparable(parent.result)
    probe = got.mtm_probe
    assert probe is not None and (probe.paths, probe.ledger) == ("ok", "ordinary")
    assert probe.dd == got.max_drawdown == parent.result.max_drawdown
    assert probe.daily is not None and len(probe.daily) == 31  # 2025-03-01..2025-03-31
    assert parent.result.mtm_probe is None


def test_mtm_daily_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from nautilus_trader.model.enums import OrderSide

    from tests.unit.test_validation_liquidation import (
        BTC,
        END,
        START,
        START_BAL,
        VENUE,
        Fill,
        Pos,
        _AtFunding,
        bar,
        crash_marks,
        engine_for,
        install_apply,
        liq_note,
        long_btc,
        no_last_bars,
    )
    from vibe_quant.validation.extraction import date_to_ns

    day0 = date_to_ns("2024-01-01")
    assert day0 is not None
    hourly = [(day0 + k * _H - 1_000_000, 40000.0, 40000.0) for k in range(1, 97)]

    # 1. Entry on day 0, funding settled on day 1, exit on day 2: each lands on its own day.
    trip = Pos("P-1", BTC, [Fill(day0 + 10 * _H, OrderSide.BUY, 1.0, 40000.0, 20.0),
                            Fill(day0 + 2 * _DAY_NS + 12 * _H, OrderSide.SELL, 1.0, 40000.0, 20.5)])
    res = _extract(_engine([trip], {BTC: hourly}), "2024-01-01", "2024-01-05",
                   funding=_AtFunding((day0 + _DAY_NS + 8 * _H,)))
    assert res.mtm_probe is not None and res.mtm_probe.daily is not None
    assert res.mtm_probe.daily.first_day == day0 // _DAY_NS
    assert _equity(res) == pytest.approx([9980.0, 9975.0, 9954.5, 9954.5], abs=1e-9)
    assert res.mtm_probe.dd == res.max_drawdown == pytest.approx(45.5 / START_BAL)

    # 2. A surviving position is marked at the run end, its estimated exit fee included.
    rising = [*hourly[:72], *[(ts, 41000.0, 41000.0) for ts, _, _ in hourly[72:]]]
    held = Pos("P-2", BTC, [Fill(day0 + 10 * _H, OrderSide.BUY, 1.0, 40000.0, 20.0)], is_closed=False)
    venue = SimpleNamespace(**vars(VENUE), taker_fee=0.0005)
    res = _extract(_engine([held], {BTC: rising}), "2024-01-01", "2024-01-05", venue=venue)
    (trade,) = res.trades
    assert trade.exit_reason == "end_of_data"
    assert trade.entry_fee + trade.exit_fee == pytest.approx(20.0 + 20.5)  # + est. exit 41000*.0005
    assert _equity(res) == pytest.approx([9980.0, 9980.0, 9980.0, 10959.5], abs=1e-9)
    assert _equity(res)[-1] == pytest.approx(START_BAL + trade.net_pnl, abs=1e-9)

    # 3. An exact-end bar belongs to the drawdown (inclusive end), never to the daily closes.
    for end in ("2024-01-01T12:00:00", "2024-01-02"):
        end_ns = date_to_ns(end)
        assert end_ns is not None
        bars = [(ts, c, lo) for ts, c, lo in hourly if ts < end_ns] + [(end_ns, 39000.0, 39000.0)]
        opened = Pos("P-3", BTC, [Fill(day0 + 2 * _H, OrderSide.BUY, 1.0, 40000.0, 0.0)],
                     is_closed=False)
        res = _extract(_engine([opened], {BTC: bars}), "2024-01-01", end, venue=venue)
        assert res.mtm_probe is not None
        # the end bar's dip to 39000 and the terminal mark's exit fee (39000 * .0005)
        assert res.mtm_probe.dd == res.max_drawdown == pytest.approx(1019.5 / START_BAL)
        assert _equity(res) == [START_BAL], end

    # 4. No bars for a traded instrument: unavailable, the DD falls back to closed trades.
    no_last_bars(monkeypatch)
    res = _extract(_engine([long_btc()], {}), START, END)
    assert res.mtm_probe is not None
    assert (res.mtm_probe.paths, res.mtm_probe.daily, res.mtm_probe.dd) == ("unavailable", None, None)
    assert res.max_drawdown == 0.0  # closed-trade fallback on a winning trade
    monkeypatch.undo()

    # 5. No positions: a flat account, one zero return per day.
    res = _extract(_engine([], {BTC: hourly}), "2024-01-01", "2024-01-04")
    assert res.mtm_probe is not None
    assert (res.mtm_probe.paths, res.mtm_probe.dd) == ("none", 0.0)
    assert res.mtm_probe.daily is not None and list(res.mtm_probe.daily.values) == [0.0, 0.0, 0.0]

    # 6. Isolated truncation: overlay ledgers, no account bust.
    calls: list[dict[str, Any]] = []
    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, calls, mk)
    res = _extract(engine_for([long_btc(sell_bar=None)], mk), START, END, mode="isolated")
    probe = res.mtm_probe
    assert liq_note(res)["count"] == 1
    assert probe is not None and (probe.paths, probe.ledger, probe.bust_ts) == ("ok", "overlay", None)
    assert probe.path_divergence is liq_note(res)["path_divergence"]
    assert probe.dd == res.max_drawdown
    assert _equity(res) == pytest.approx([START_BAL + sum(t.net_pnl for t in res.trades)], abs=1e-9)

    # 7. Cross bust: the bust instant is carried; the account ends at zero (no daily series).
    later = Pos("P-9", BTC, [Fill(bar(45), OrderSide.BUY, 1.0, 30000.0, 15.0)], is_closed=False)
    mk = {BTC: crash_marks(to=30000.0)}
    install_apply(monkeypatch, calls, mk)
    res = _extract(engine_for([long_btc(sell_bar=None), later], mk), START, END, mode="cross")
    probe = res.mtm_probe
    assert liq_note(res)["bust_ts"] is not None
    assert probe is not None and (probe.paths, probe.ledger) == ("ok", "overlay")
    assert probe.bust_ts == liq_note(res)["bust_ts"]
    assert probe.dd == res.max_drawdown == 1.0
    assert probe.daily is None
