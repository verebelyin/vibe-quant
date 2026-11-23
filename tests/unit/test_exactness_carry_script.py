"""Exit-code contract of ``scripts/agents/exactness_carry.py`` (R6, task T6b).

``main()`` is the PASS/FAIL gate the swarm gate reads: it returns 0 when the NT
engine run equals the no-NT reference and every recorded headline pin holds, and
1 on any drift. The heavy run (real catalog + archive, four coins, 2023-2026) is
monkeypatched here; both codes are asserted from literals, so a mutation of the
return (always 0) fails.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from vibe_quant.portfolio.spec import CarrySpec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from vibe_quant.portfolio.ticks import CarryInputs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts/agents/exactness_carry.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exactness_carry", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script module with the heavy run replaced by fakes that pass every pin."""
    module = _load_script()
    spec = module.CarrySpec.from_dates(module.SYMBOLS, module.START, module.END)
    inputs = SimpleNamespace(spec=spec, decisions=tuple(range(module.EXPECTED_DECISIONS)))
    report = SimpleNamespace(
        summary=dict(module.RECORDED), daily=[], weekly=[], legs=[], by_year=[]
    )
    engine = SimpleNamespace(fills=(), payments=())
    reference = SimpleNamespace()
    monkeypatch.setattr(module, "load_carry_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(module, "run_carry_engine", lambda inputs: engine)
    monkeypatch.setattr(module, "simulate_reference", lambda inputs: reference)
    monkeypatch.setattr(module, "carry_metrics", lambda *args, **kwargs: report)
    return module


def test_main_returns_zero_when_the_runs_are_identical(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing ``compare_runs`` (0 mismatches) and matching pins -> exit code 0."""
    monkeypatch.setattr(script, "compare_runs", lambda *args, **kwargs: [])

    assert script.main() == 0

    out = capsys.readouterr().out
    assert "0 mismatches" in out
    assert "carry exactness: IDENTICAL" in out


def test_main_returns_one_on_any_drift(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One mismatch from ``compare_runs`` -> exit code 1 (the gate is fail-closed)."""
    monkeypatch.setattr(
        script, "compare_runs", lambda *args, **kwargs: ["weekly_sharpe: 0.0 != 1.0"]
    )

    assert script.main() == 1

    out = capsys.readouterr().out
    assert "1 mismatches" in out
    assert "carry exactness: DRIFT" in out


# --- --variants (x0axf.14 T7): each variant vs a matched control ---------------------------


@pytest.fixture
def variants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """The script with its loader swapped for the synthetic 4-symbol parity fixture.

    The fixture keeps every field the script toggles (divisor, weighting, threshold,
    liquidation mode); its own dates, a 5-day vol lookback and its vol/volume maps
    stand in for the real catalog and archive.
    """
    from tests.unit import test_carry_parity as fixture

    module = _load_script()
    loads = itertools.count()

    def load(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        inverse_vol = spec.weighting == "inverse_vol"
        local = CarrySpec(
            symbols=fixture._SYMBOLS,
            start_ns=fixture.START,
            end_ns=fixture.END,
            legs_divisor=spec.legs_divisor,
            weighting=spec.weighting,
            vol_lookback_days=5 if inverse_vol else 30,
            min_quote_volume=spec.min_quote_volume,
            liquidation_mode=spec.liquidation_mode,
        )
        return fixture._inputs(
            tmp_path,
            name=f"{next(loads)}-{local.spec_hash()[:12]}",  # one fresh tmp archive per load
            spec=local,
            vol_closes=fixture._vol_closes() if inverse_vol else None,
            quote_volume=fixture._quote_volume() if spec.min_quote_volume is not None else None,
        )

    monkeypatch.setattr(module, "load_carry_inputs", load)
    return module


def _variant_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("variant ")]


def _portfolio_state() -> tuple[bytes | None, list[str] | None]:
    root = _REPO_ROOT / "data/portfolio"
    trials = root / "trials.jsonl"
    return (
        trials.read_bytes() if trials.exists() else None,
        sorted(os.listdir(root)) if root.is_dir() else None,
    )


def test_variants_flag(variants: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    """R47: three variant lines ``mismatches=0 feature=ok``, exit 0, nothing written."""
    before = _portfolio_state()

    assert variants.main(["--variants"]) == 0

    lines = _variant_lines(capsys.readouterr().out)
    assert [line.split(":")[0] for line in lines] == [
        "variant inverse_vol",
        "variant liquidity",
        "variant cross",
    ]
    assert all(" mismatches=0 feature=ok" in line for line in lines)
    assert _portfolio_state() == before


def test_variants_inverse_vol_needs_reweighting(
    variants: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R47a: proof (a) runs both arms at divisor 2; an identity re-weighting fails it."""
    specs: list[CarrySpec] = []
    load = variants.load_carry_inputs

    def recording(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        specs.append(spec)
        return load(spec, catalog, archive)

    monkeypatch.setattr(variants, "load_carry_inputs", recording)
    monkeypatch.setattr(
        "vibe_quant.portfolio.sizing.inverse_vol_weights",
        lambda weights, vols, gross: dict(weights),
    )

    assert variants.main(["--variants"]) == 1

    lines = _variant_lines(capsys.readouterr().out)
    assert lines[0].startswith("variant inverse_vol: mismatches=0 feature=FAIL")
    assert all("feature=ok" in line for line in lines[1:])
    arms = [s for s in specs if s.legs_divisor == 2]
    assert sorted(s.weighting for s in arms) == ["equal", "inverse_vol"]
    assert all(s.min_quote_volume is None and s.liquidation_mode == "off" for s in arms)


def test_variants_liquidity_predicate(
    variants: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R47b: a threshold that excludes nothing the control selects fails proof (b)."""
    assert variants.MIN_QUOTE_VOLUME == 5e8
    monkeypatch.setattr(variants, "MIN_QUOTE_VOLUME", 1e5)  # below every fixture day

    assert variants.main(["--variants"]) == 1

    lines = _variant_lines(capsys.readouterr().out)
    assert lines[1].startswith("variant liquidity: mismatches=0 feature=FAIL")
    assert "feature=ok" in lines[0] and "feature=ok" in lines[2]


def test_default_invocation_checks_the_shipped_pins(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R47a: without ``--variants`` the proof loads only the shipped divisor-3 spec."""
    specs: list[CarrySpec] = []
    load = script.load_carry_inputs
    monkeypatch.setattr(
        script, "load_carry_inputs", lambda spec, *args: specs.append(spec) or load(spec, *args)
    )
    monkeypatch.setattr(script, "compare_runs", lambda *args, **kwargs: [])

    assert script.main([]) == 0

    assert specs == [CarrySpec.from_dates(script.SYMBOLS, script.START, script.END)]
    assert specs[0].legs_divisor == 3
    out = capsys.readouterr().out
    assert "carry exactness: IDENTICAL" in out
    assert _variant_lines(out) == []


def test_variants_fail_on_any_mismatch_or_note_disagreement(
    variants: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R47: engine != reference fails a variant whose feature holds; cross needs equal notes."""
    from dataclasses import replace

    monkeypatch.setattr(variants, "compare_runs", lambda *args, **kwargs: ["drift"])
    assert variants.main(["--variants"]) == 1
    lines = _variant_lines(capsys.readouterr().out)
    assert all(" mismatches=1 feature=ok" in line for line in lines)

    monkeypatch.setattr(variants, "compare_runs", lambda *args, **kwargs: [])
    real = variants.carry_metrics

    def engine_notes_differ(run: object, inputs: CarryInputs) -> object:
        report = real(run, inputs)
        notes = report.summary["liquidation"]
        if getattr(run, "engine", None) == "nt" and isinstance(notes, dict):
            return replace(report, summary={**report.summary, "liquidation": {**notes, "count": 1}})
        return report

    monkeypatch.setattr(variants, "carry_metrics", engine_notes_differ)
    assert variants.main(["--variants"]) == 1
    lines = _variant_lines(capsys.readouterr().out)
    assert lines[2].startswith("variant cross: mismatches=0 feature=FAIL")
    assert "feature=ok" in lines[0] and "feature=ok" in lines[1]


# --- --signal trend (arch-c9-2 R67/R68): the 1m proof, the prefix run, the 1h tier ---------


def _archive(path: Path, intervals: Mapping[str, tuple[str, ...]]) -> Path:
    """A raw_klines table holding one row per (symbol, interval)."""
    import sqlite3

    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE raw_klines (symbol TEXT, interval TEXT, open_time INTEGER, close REAL)"
        )
        conn.executemany(
            "INSERT INTO raw_klines VALUES (?, ?, 0, 1.0)",
            [(symbol, interval) for symbol, tiers in intervals.items() for interval in tiers],
        )
    conn.close()
    return path


def _fake_run(spec: CarrySpec) -> SimpleNamespace:
    """A run whose decision, fill and payment times depend only on the spec's window."""
    from vibe_quant.portfolio.spec import decision_times

    times = decision_times(spec.start_ns, spec.end_ns, spec.schedule)
    return SimpleNamespace(
        spec=spec,
        decisions=tuple(SimpleNamespace(ts=t, weights={"BTCUSDT": 0.1}) for t in times),
        fills=tuple(SimpleNamespace(ts=t + 3_599_999_000_000) for t in times),
        payments=tuple(SimpleNamespace(ts=t + 7 * 3_600 * 10**9) for t in times),
    )


@pytest.fixture
def trend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """The script with every heavy call faked; ``specs`` records each load in order."""
    module = _load_script()
    specs: list[CarrySpec] = []

    def load(spec: CarrySpec, catalog: Path, archive: Path) -> SimpleNamespace:
        specs.append(spec)
        return SimpleNamespace(spec=spec, decisions=tuple(range(len(_fake_run(spec).decisions))))

    report = SimpleNamespace(
        summary=dict(module.TREND_RECORDED), daily=[], weekly=[], legs=[], by_year=[]
    )
    monkeypatch.setattr(module, "load_carry_inputs", load)
    monkeypatch.setattr(module, "run_carry_engine", lambda inputs: _fake_run(inputs.spec))
    monkeypatch.setattr(module, "simulate_reference", lambda inputs: _fake_run(inputs.spec))
    monkeypatch.setattr(module, "carry_metrics", lambda *args, **kwargs: report)
    monkeypatch.setattr(module, "compare_runs", lambda *args, **kwargs: [])
    archive = _archive(tmp_path / "a.db", dict.fromkeys(module.SYMBOLS, ("1m",)))
    monkeypatch.setattr(module, "ARCHIVE_DB_PATH", archive)
    return SimpleNamespace(module=module, specs=specs, tmp_path=tmp_path)


def test_trend_flag_proves_the_pinned_1m_book_and_its_prefix(
    trend: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """R67: ``--signal trend`` loads the pinned long_short book (literal fields), then the
    same book ending 2025-06-01; no archive 1h rows -> the R68 SKIP line, exit 0."""
    assert trend.module.main(["--signal", "trend"]) == 0

    out = capsys.readouterr().out
    assert "carry trend exactness: IDENTICAL" in out
    assert "SKIP 1h: no archive 1h rows" in out
    assert "carry exactness: IDENTICAL" not in out  # the funding proof did not run
    full, prefix = trend.specs
    assert full == CarrySpec(
        symbols=("BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"),
        start_ns=1654041600000000000,  # 2022-06-01
        end_ns=1790553600000000000,  # 2026-09-28
        initial_balance=100_000.0,
        gross=1.0,
        band=0.10,
        weighting="time_series",
        vol_lookback_days=30,
        signal="trend",
        schedule="daily",
        fill_granularity="1m",
        direction="long_short",
        vol_target=0.40,
        spread=0.0001,
    )
    assert prefix.end_ns == 1748736000000000000  # 2025-06-01
    assert prefix == CarrySpec(**{**vars(full), "end_ns": prefix.end_ns})


def test_trend_flag_runs_the_hour_tier_only_with_1h_rows_for_all_four(
    trend: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """R68: 1h rows for every coin -> a third (1h) parity run and no SKIP; 3 of 4 -> SKIP."""
    module = trend.module
    tiers = dict.fromkeys(module.SYMBOLS, ("1m", "1h"))
    module.ARCHIVE_DB_PATH = _archive(trend.tmp_path / "h.db", tiers)

    assert module.main(["--signal", "trend"]) == 0
    out = capsys.readouterr().out
    assert "SKIP 1h" not in out
    assert "trend 1h parity: 0 mismatches" in out
    assert [s.fill_granularity for s in trend.specs] == ["1m", "1m", "1h"]
    assert trend.specs[2].end_ns == trend.specs[0].end_ns

    trend.specs.clear()
    module.ARCHIVE_DB_PATH = _archive(trend.tmp_path / "g.db", {**tiers, "SOLUSDT": ("1m",)})
    assert module.main(["--signal", "trend"]) == 0
    assert "SKIP 1h: no archive 1h rows" in capsys.readouterr().out
    assert [s.fill_granularity for s in trend.specs] == ["1m", "1m"]


def test_trend_flag_fails_closed(
    trend: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 1 on a 1m mismatch, a moved pin, a broken prefix or a 1h mismatch."""
    module = trend.module
    monkeypatch.setattr(module, "compare_runs", lambda *args, **kwargs: ["daily: 1 != 2"])
    assert module.main(["--signal", "trend"]) == 1
    assert "carry trend exactness: DRIFT" in capsys.readouterr().out

    monkeypatch.setattr(module, "compare_runs", lambda *args, **kwargs: [])
    first = next(iter(module.TREND_RECORDED))
    monkeypatch.setitem(module.TREND_RECORDED, first, "moved")
    assert module.main(["--signal", "trend"]) == 1
    assert "carry trend exactness: DRIFT" in capsys.readouterr().out
    monkeypatch.undo()

    module = _load_script()
    monkeypatch.setattr(module, "load_carry_inputs", lambda spec, *a: SimpleNamespace(spec=spec))
    monkeypatch.setattr(module, "simulate_reference", lambda inputs: _fake_run(inputs.spec))
    report = SimpleNamespace(summary=dict(module.TREND_RECORDED), daily=[])
    monkeypatch.setattr(module, "carry_metrics", lambda *args, **kwargs: report)
    monkeypatch.setattr(module, "compare_runs", lambda *args, **kwargs: [])
    tiers = dict.fromkeys(module.SYMBOLS, ("1m", "1h"))
    monkeypatch.setattr(module, "ARCHIVE_DB_PATH", _archive(trend.tmp_path / "x.db", tiers))

    def short_drops_a_fill(inputs: SimpleNamespace) -> SimpleNamespace:
        run = _fake_run(inputs.spec)
        if inputs.spec.end_ns < module.EXPECTED_END_NS:
            run.fills = run.fills[:-1]
        return run

    monkeypatch.setattr(module, "run_carry_engine", short_drops_a_fill)
    assert module.main(["--signal", "trend"]) == 1
    assert "carry trend exactness: DRIFT" in capsys.readouterr().out

    monkeypatch.setattr(module, "run_carry_engine", lambda inputs: _fake_run(inputs.spec))
    hour_only = ["fills: 1 != 2"]
    monkeypatch.setattr(
        module,
        "compare_runs",
        lambda engine, *a: hour_only if engine.spec.fill_granularity == "1h" else [],
    )
    assert module.main(["--signal", "trend"]) == 1
    out = capsys.readouterr().out
    assert "carry trend exactness: IDENTICAL" in out
    assert "trend 1h parity: 1 mismatches" in out


def test_trend_prefix_compares_everything_before_the_cutoff(trend: SimpleNamespace) -> None:
    """The prefix rule: decisions are a prefix; fills/payments with ts < cutoff are equal
    (an event AT the cutoff is outside, one ns before is inside); nothing at/after it."""
    module = trend.module
    cutoff = 1_000

    def run(decisions: tuple[int, ...], fills: tuple[int, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            decisions=tuple(SimpleNamespace(ts=t) for t in decisions),
            fills=tuple(SimpleNamespace(ts=t, qty=1) for t in fills),
            payments=(),
        )

    long = run((100, 900, 1_000, 1_100), (150, 999, 1_000, 1_200))
    assert module.prefix_mismatches(run((100, 900, 1_000), (150, 999)), long, cutoff) == []
    assert module.prefix_mismatches(run((100, 900), (150,)), long, cutoff) != []  # lost 999
    assert module.prefix_mismatches(run((100, 900), (150, 999, 1_000)), long, cutoff) != []
    assert module.prefix_mismatches(run((100, 1_001), (150, 999)), long, cutoff) != []
    assert module.prefix_mismatches(run((100, 950), (150, 999)), long, cutoff) != []
    # a dropped last decision with the same fills (band skip) is still a mismatch
    assert module.prefix_mismatches(run((100, 900), (150, 999)), long, cutoff) == [
        "decisions: 2 != long's 3 up to the cutoff"
    ]


@pytest.mark.parametrize("argv", [["--signal", "carry"], ["--signal", "trend", "--variants"]])
def test_signal_flag_parse(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown signal, and --variants with trend, exit 2 before any load."""
    module = _load_script()
    with pytest.raises(SystemExit) as raised:
        module.main(argv)
    assert raised.value.code == 2
