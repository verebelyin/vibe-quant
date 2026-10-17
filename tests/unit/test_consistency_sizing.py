"""A screening reference must share the validation run's sizing (vibe-quant-2gyk1.2 S3).

Rows S28-S29c. ``max_position_pct`` / ``risk_per_trade`` are run-level settings: they never
appear in a sweep row, so they filter the exact-row param match out (S28), and a screening run
whose recorded sizing differs from the validation run's is not a like-for-like reference —
comparing across caps compares two different exposure policies rather than overfitting (S29).
A reference whose stored sizing cannot be read is skipped, never fatal (S29c).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.consistency import find_screening_reference

if TYPE_CHECKING:
    from pathlib import Path

WINDOW = ("2024-01-01", "2025-01-01")
SYMBOLS = ["BTCUSDT"]


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    s = StateManager(db_path=tmp_path / "t.db")
    yield s
    s.close()


def _screening(
    state: StateManager,
    sid: int,
    parameters: dict[str, object] | None = None,
    *,
    rows: list[tuple[dict[str, object], float, int]] | None = None,
) -> int:
    """A completed screening run over WINDOW/SYMBOLS with one sweep row per entry."""
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="screening", symbols=SYMBOLS, timeframe="4h",
        start_date=WINDOW[0], end_date=WINDOW[1], parameters=parameters or {},
    )
    state.update_backtest_run_status(run_id, "completed")
    for row_params, sharpe, trades in rows if rows is not None else [({}, 2.0, 50)]:
        state.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
            " VALUES (?, ?, ?, ?)",
            (run_id, json.dumps(row_params), sharpe, trades),
        )
    state.conn.commit()
    return run_id


def _scoped(state: StateManager, sid: int, **kwargs: object) -> object:
    return find_screening_reference(
        state, sid, "s1", val_window=WINDOW, val_symbols=SYMBOLS, **kwargs  # type: ignore[arg-type]
    )


def test_sizing_keys_do_not_block_exact_row_match(state: StateManager) -> None:
    """S28: the validated params match a sweep row once the run-level sizing keys are dropped."""
    sid = state.create_strategy("s1", {"name": "s1"})
    run_id = _screening(
        state, sid, {"max_position_pct": 1.0},
        rows=[({"rsi_period": 14}, 2.0, 50), ({"rsi_period": 21}, 9.9, 60)],
    )
    ref = _scoped(state, sid, validated_params={"rsi.period": 14, "max_position_pct": 1.0})
    assert ref is not None
    assert ref.source == f"screening_run:{run_id}", "fell back to the best row instead of matching"
    assert ref.sharpe == 2.0


def test_reference_run_must_share_sizing(state: StateManager) -> None:
    """S29: the newest same-scope run at other sizing is skipped for the one that matches."""
    sid = state.create_strategy("s1", {"name": "s1"})
    capped = _screening(state, sid, {"max_position_pct": 1.0}, rows=[({}, 1.5, 50)])
    defaults = _screening(state, sid, {}, rows=[({}, 9.0, 50)])
    assert defaults > capped

    covered = _scoped(state, sid, val_sizing={"max_position_pct": 1.0})
    assert covered is not None
    assert covered.source == f"screening_run:{capped}"
    assert covered.sharpe == 1.5

    # The default validation run is compared against the default-sizing screening run.
    plain = _scoped(state, sid, val_sizing={})
    assert plain is not None
    assert plain.source == f"screening_run:{defaults}"

    # val_sizing=None keeps the legacy behaviour: no sizing filter at all.
    unfiltered = _scoped(state, sid)
    assert unfiltered is not None
    assert unfiltered.source == f"screening_run:{defaults}"

    # No run carries that sizing -> no reference at all, not a wrong one.
    assert _scoped(state, sid, val_sizing={"max_position_pct": 2.0}) is None


def test_discovery_fallback_must_share_sizing(state: StateManager) -> None:
    """S29: the discovery champion is only a reference at the same recorded sizing."""
    sid = state.create_strategy("genome_abc123", {"name": "genome_abc123"})
    disc_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=SYMBOLS, timeframe="4h",
        start_date=WINDOW[0], end_date=WINDOW[1], parameters={},
    )
    state.update_backtest_run_status(disc_id, "completed")
    state.save_backtest_result(
        disc_id,
        {
            "sharpe_ratio": 1.8,
            "notes": json.dumps(
                {
                    "sizing": {"max_position_pct": 1.0},
                    "top_strategies": [
                        {"dsl": {"name": "genome_abc123"}, "sharpe": 1.8, "trades": 64}
                    ],
                }
            ),
        },
    )

    covered = find_screening_reference(state, sid, "genome_abc123", val_sizing={"max_position_pct": 1.0})
    assert covered is not None
    assert covered.source == f"discovery_run:{disc_id}"

    assert find_screening_reference(state, sid, "genome_abc123", val_sizing={}) is None
    assert (
        find_screening_reference(state, sid, "genome_abc123", val_sizing={"risk_per_trade": 0.01})
        is None
    )
    # val_sizing=None = no filter: legacy callers are unchanged.
    unfiltered = find_screening_reference(state, sid, "genome_abc123")
    assert unfiltered is not None
    assert unfiltered.source == f"discovery_run:{disc_id}"


def test_default_sizing_matches_legacy_reference(state: StateManager) -> None:
    """S29b: a legacy reference (no sizing keys stored) is at the defaults: ``{} == {}``."""
    sid = state.create_strategy("s1", {"name": "s1"})
    run_id = _screening(state, sid, {}, rows=[({}, 2.0, 50)])
    ref = _scoped(state, sid, val_sizing={})
    assert ref is not None
    assert ref.source == f"screening_run:{run_id}"

    # Other run-level knobs in the column do not read as a sizing.
    legacy = _screening(
        state, sid, {"initial_balance": 1000, "sweep": {"rsi_period": [7, 14]}}, rows=[({}, 3.0, 50)]
    )
    ref2 = _scoped(state, sid, val_sizing={})
    assert ref2 is not None
    assert ref2.source == f"screening_run:{legacy}"


def test_garbage_reference_sizing_is_skipped(state: StateManager) -> None:
    """S29c: an unreadable stored sizing skips its run instead of raising or matching."""
    sid = state.create_strategy("s1", {"name": "s1"})
    good = _screening(state, sid, {}, rows=[({}, 2.0, 50)])
    bad_value = _screening(state, sid, {"max_position_pct": -1}, rows=[({}, 9.9, 50)])
    bad_json = _screening(state, sid, {}, rows=[({}, 9.8, 50)])
    bad_shape = _screening(state, sid, {}, rows=[({}, 9.7, 50)])
    state.conn.execute("UPDATE backtest_runs SET parameters = ? WHERE id = ?", ("{not json", bad_json))
    state.conn.execute("UPDATE backtest_runs SET parameters = ? WHERE id = ?", ("[]", bad_shape))
    state.conn.commit()

    ref = _scoped(state, sid, val_sizing={})
    assert ref is not None
    assert ref.source == f"screening_run:{good}", "an unreadable reference was used or raised"

    # A garbage value is never interpreted as a sizing a caller could ask for.
    assert _scoped(state, sid, val_sizing={"max_position_pct": -1}) is None
    assert bad_value > good


def _champion_notes(**extra: object) -> dict[str, object]:
    """The persisted notes of a discovery run whose champion is ``genome_abc123``."""
    notes: dict[str, object] = {
        "top_strategies": [{"dsl": {"name": "genome_abc123"}, "sharpe": 1.8, "trades": 64}]
    }
    notes.update(extra)
    return notes


def _discovery_with_notes(state: StateManager, notes: dict[str, object]) -> int:
    """A completed discovery run whose persisted notes JSON is exactly ``notes``."""
    disc_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=SYMBOLS, timeframe="4h",
        start_date=WINDOW[0], end_date=WINDOW[1], parameters={},
    )
    state.update_backtest_run_status(disc_id, "completed")
    state.save_backtest_result(disc_id, {"sharpe_ratio": 1.8, "notes": json.dumps(notes)})
    return disc_id


def test_absent_notes_sizing_reads_as_defaults(state: StateManager) -> None:
    """S29d: a discovery run with no ``notes.sizing`` key was launched at the defaults."""
    sid = state.create_strategy("genome_abc123", {"name": "genome_abc123"})
    disc_id = _discovery_with_notes(state, _champion_notes())
    ref = find_screening_reference(state, sid, "genome_abc123", val_sizing={})
    assert ref is not None
    assert ref.source == f"discovery_run:{disc_id}"


@pytest.mark.parametrize(
    "recorded", [None, [], "garbage", 7], ids=["null", "list", "string", "number"]
)
def test_non_dict_notes_sizing_is_skipped(state: StateManager, recorded: object) -> None:
    """S29d (SF1): a PRESENT non-dict ``notes.sizing`` is unreadable — skip, never defaults."""
    sid = state.create_strategy("genome_abc123", {"name": "genome_abc123"})
    _discovery_with_notes(state, _champion_notes(sizing=recorded))
    assert find_screening_reference(state, sid, "genome_abc123", val_sizing={}) is None


def test_invalid_dict_notes_sizing_is_skipped(state: StateManager) -> None:
    """S29d/S23b: a dict ``notes.sizing`` holding an invalid value is unreadable too."""
    sid = state.create_strategy("genome_abc123", {"name": "genome_abc123"})
    _discovery_with_notes(state, _champion_notes(sizing={"max_position_pct": -1}))
    assert find_screening_reference(state, sid, "genome_abc123", val_sizing={}) is None


def test_max_margin_filtered(state: StateManager) -> None:
    """R41: ``max_margin_pct`` is a run-level sizing field, so it never blocks the exact-row
    match and a reference at another margin limit is not like-for-like."""
    sid = state.create_strategy("s1", {"name": "s1"})
    limited = _screening(
        state, sid, {"max_margin_pct": 0.5},
        rows=[({"rsi_period": 14}, 2.0, 50), ({"rsi_period": 21}, 9.9, 60)],
    )
    plain = _screening(state, sid, {}, rows=[({}, 7.0, 50)])
    assert plain > limited
    ref = _scoped(
        state, sid, validated_params={"rsi.period": 14, "max_margin_pct": 0.5},
        val_sizing={"max_margin_pct": 0.5},
    )
    assert ref is not None
    assert ref.source == f"screening_run:{limited}", "margin key blocked the exact-row match"
    assert ref.sharpe == 2.0
    # the default-margin (off) validation run does not compare against the limited run
    off = _scoped(state, sid, val_sizing={})
    assert off is not None
    assert off.source == f"screening_run:{plain}"
