"""Launch-parameter parse phases through today's entry points (row R27e).

These four probes pin WHEN each entry point parses each launch field, so
``rabmo.3`` (``launch_settings.iter_fields``) can move the parsing without moving
the errors. They are written on base and stay unchanged through S2/S3/S5.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening.nt_runner import MissingBarDataError
from vibe_quant.screening.pipeline import create_screening_pipeline
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

_DSL_GOLDEN = Path(__file__).parent / "goldens" / "strategy_239_dsl.json"


def _dsl() -> Any:
    return validate_strategy_dict(json.loads(_DSL_GOLDEN.read_text())["dsl"])


def _runner(tmp_path: Path) -> ValidationRunner:
    return ValidationRunner(db_path=tmp_path / "phases.db")


def test_mock_ignores_fee_tier(tmp_path: Path) -> None:
    """The mock branch stops before the fee parse: a bad fee_tier cannot fail it."""
    pipeline = create_screening_pipeline(
        _dsl(), use_mock=True, max_workers=1, parameters={"fee_tier": "VIP99"}
    )
    result = pipeline.run()
    assert result.total_combinations == 1


def test_missing_ticks_beat_bad_fee_tier(tmp_path: Path) -> None:
    """Real branch: fill ticks resolve BEFORE the fee parse (pipeline.py:610/613)."""
    with pytest.raises(MissingBarDataError, match="no 1m bars"):
        create_screening_pipeline(
            _dsl(),
            use_mock=False,
            symbols=["BTCUSDT"],
            start_date="2024-01-01",
            end_date="2024-01-08",
            catalog_path=str(tmp_path),
            parameters={"fee_tier": "VIP99"},
        )


def test_direct_venue_balance_before_leverage(tmp_path: Path) -> None:
    """A direct ``_create_venue_config`` parses balance before liquidation."""
    runner = _runner(tmp_path)
    try:
        with pytest.raises(ValidationRunnerError) as excinfo:
            runner._create_venue_config(  # noqa: SLF001 - the documented direct entry point
                {"parameters": {"initial_balance": 0, "leverage": 0}}, None
            )
    finally:
        runner.close()
    assert str(excinfo.value) == (
        "invalid initial_balance: initial_balance must be a whole number in "
        "[1, 1000000000], got 0"
    )


def test_direct_venue_skips_sizing_check(tmp_path: Path) -> None:
    """A direct venue build never parses sizing, so cap 20 over leverage 3 runs."""
    runner = _runner(tmp_path)
    try:
        created: Any = runner._create_venue_config(  # noqa: SLF001
            {"parameters": {"max_position_pct": 20, "leverage": 3}}, None
        )
    finally:
        runner.close()
    # rabmo.3's S5 makes this return (VenueConfig, LaunchSettings); the venue is first.
    venue = created[0] if isinstance(created, tuple) else created
    assert venue.default_leverage == Decimal("3")
    assert venue.starting_balance_usdt == 1000
