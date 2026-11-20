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
