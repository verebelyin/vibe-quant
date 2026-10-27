"""Exit-code contract of ``scripts/agents/fast_tier_parity.py --strict`` (bead f6e0k).

A genome the fast tier cannot evaluate raises ``FastTierUnsupported``; the script reports it on
its own ``UNSUPPORTED`` line and keeps it out of both the match and the mismatch count. The
plain run must still exit 0 when every *supported* genome matched, while ``--strict`` turns any
unsupported genome into exit 1 so a fast-tier coverage regression fails the parity gate instead
of quietly shrinking the sample. A run with no unsupported genome exits 0 either way.

``main`` imports the engine, the indicator pool and both tiers inside the function; this test
replaces those imports, the genome pool and both tiers with fakes, so no catalog, archive,
market data or scratch state DB is needed and every case finishes in well under a second.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts/agents/fast_tier_parity.py"
_METRICS = {
    "total_trades": 1,
    "total_return": 0.5,
    "sharpe_ratio": 1.0,
    "profit_factor": 2.0,
    "max_drawdown": 0.1,
}


class _FastTierUnsupported(Exception):
    """Stand-in for ``vibe_quant.fast_tier.FastTierUnsupported``."""


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fast_tier_parity", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _genome(direction: str, indicator: str, *, fast: bool) -> SimpleNamespace:
    """A genome stand-in: ``fast`` is what the faked fast tier reads to reject it."""
    return SimpleNamespace(
        direction=SimpleNamespace(name=direction),
        entry_genes=[SimpleNamespace(indicator_type=indicator)],
        exit_genes=[],
        fast=fast,
    )


def _fake_package(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def run_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., int]:
    """``main`` with the pool, both tiers and every ``vibe_quant`` import faked."""
    script = _load_script()

    def run(genomes: list[SimpleNamespace], argv: list[str]) -> int:
        def fast_run_single(self: Any, genome: Any, start: str, end: str) -> dict[str, Any]:
            if not genome.fast:
                raise _FastTierUnsupported("context indicator")
            return dict(_METRICS)

        def nt_run_single(self: Any, genome: Any, start: str, end: str) -> dict[str, Any]:
            return dict(_METRICS)

        def fast_init(self: Any, *args: Any, **kwargs: Any) -> None:
            self.last_positions = []

        class _FastBacktestFn:
            __init__ = fast_init
            _run_single = fast_run_single
            last_positions: list[Any] = []

        class _NTBacktestFn:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            _run_single = nt_run_single

        class _NTScreeningRunner:
            @staticmethod
            def _extract_metrics(*args: Any, **kwargs: Any) -> None:
                return None

        operators = _fake_package(
            monkeypatch,
            "vibe_quant.discovery.operators",
            _ensure_pool=lambda: None,
            INDICATOR_POOL={"RSI": object()},
            _INDICATOR_NAMES=["RSI"],
            initialize_population=lambda size: list(genomes),
        )
        _fake_package(monkeypatch, "vibe_quant.discovery", operators=operators)
        _fake_package(monkeypatch, "vibe_quant.discovery.backtest_fn", NTBacktestFn=_NTBacktestFn)
        indicators = _fake_package(
            monkeypatch,
            "vibe_quant.dsl.indicators",
            indicator_registry=SimpleNamespace(
                get=lambda name: SimpleNamespace(needs_context=False)
            ),
        )
        _fake_package(monkeypatch, "vibe_quant.dsl", indicators=indicators)
        _fake_package(monkeypatch, "vibe_quant.fast_tier", FastTierUnsupported=_FastTierUnsupported)
        _fake_package(
            monkeypatch, "vibe_quant.fast_tier.backtest_fn", FastBacktestFn=_FastBacktestFn
        )
        _fake_package(
            monkeypatch, "vibe_quant.liquidation_settings", LiquidationSettings=SimpleNamespace
        )
        nt_runner = _fake_package(
            monkeypatch, "vibe_quant.screening.nt_runner", NTScreeningRunner=_NTScreeningRunner
        )
        _fake_package(monkeypatch, "vibe_quant.screening", nt_runner=nt_runner)
        extraction = _fake_package(
            monkeypatch, "vibe_quant.validation.extraction", all_positions=lambda engine: []
        )
        _fake_package(monkeypatch, "vibe_quant.validation", extraction=extraction)
        monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))

        saved_cwd = os.getcwd()
        saved_env = {key: os.environ.get(key) for key in ("VIBE_QUANT_DB", "VIBE_QUANT_LOG_DIR")}
        try:
            return script.main(
                [
                    "--start",
                    "2026-03-02",
                    "--end",
                    "2026-03-09",
                    "--n",
                    str(len(genomes)),
                    *argv,
                ]
            )
        finally:
            os.chdir(saved_cwd)
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return run


@pytest.mark.parametrize("unsupported_first", [True, False], ids=["first", "last"])
def test_strict_exits_one_when_a_genome_is_unsupported(
    run_parity: Callable[..., int],
    capsys: pytest.CaptureFixture[str],
    unsupported_first: bool,
) -> None:
    """One ``FastTierUnsupported`` genome among matching ones -> ``--strict`` exits 1."""
    unsupported = _genome("SHORT", "WILLR", fast=False)
    supported = _genome("LONG", "RSI", fast=True)
    genomes = [unsupported, supported] if unsupported_first else [supported, unsupported]

    assert run_parity(genomes, ["--strict"]) == 1

    out = capsys.readouterr().out
    assert "UNSUPPORTED" in out
    assert "match=1 mismatch=0 unsupported=1" in out


@pytest.mark.parametrize("unsupported_first", [True, False], ids=["first", "last"])
def test_without_strict_an_unsupported_genome_still_exits_zero(
    run_parity: Callable[..., int],
    capsys: pytest.CaptureFixture[str],
    unsupported_first: bool,
) -> None:
    """Same run without ``--strict``: the supported genome matched, so exit 0."""
    unsupported = _genome("SHORT", "WILLR", fast=False)
    supported = _genome("LONG", "RSI", fast=True)
    genomes = [unsupported, supported] if unsupported_first else [supported, unsupported]

    assert run_parity(genomes, []) == 0

    out = capsys.readouterr().out
    assert "UNSUPPORTED" in out
    assert "match=1 mismatch=0 unsupported=1" in out


@pytest.mark.parametrize("argv", [[], ["--strict"]], ids=["plain", "strict"])
def test_supported_only_genomes_exit_zero_in_both_modes(
    run_parity: Callable[..., int],
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    """No unsupported genome: ``--strict`` changes nothing, both modes exit 0."""
    genomes = [_genome("LONG", "RSI", fast=True), _genome("SHORT", "WILLR", fast=True)]

    assert run_parity(genomes, argv) == 0

    out = capsys.readouterr().out
    assert "UNSUPPORTED" not in out
    assert "match=2 mismatch=0 unsupported=0" in out
