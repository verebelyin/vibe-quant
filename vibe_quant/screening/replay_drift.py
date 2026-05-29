"""Post-promote replay drift check.

When a discovery genome is promoted via /api/discovery/results/{run_id}/promote,
the resulting screening replay should reproduce the discovery-reported metrics.
If the screening replay's trade count or Sharpe drifts outside tolerance, the
run is flagged so operators can see the divergence in the UI before trusting
the strategy for paper/live.

See bd-l6ml and bd-r8i7 for the non-reproducibility incident that motivated
this guardrail.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

TRADE_DRIFT_THRESHOLD = 0.9
SHARPE_DRIFT_THRESHOLD = 0.8


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return num/den, or None if inputs are missing or denominator is ~0."""
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-9:
        return None
    return float(numerator) / float(denominator)


def _get_promote_source(parameters: object) -> dict[str, object] | None:
    """Extract promote_source dict from a backtest_runs.parameters value."""
    params: dict[str, object]
    if isinstance(parameters, str):
        try:
            parsed = json.loads(parameters)
        except (json.JSONDecodeError, TypeError):
            return None
        params = parsed if isinstance(parsed, dict) else {}
    elif isinstance(parameters, dict):
        params = parameters
    else:
        return None
    src = params.get("promote_source")
    return src if isinstance(src, dict) else None


def _load_discovery_metrics(
    state: StateManager, discovery_run_id: int, strategy_index: int
) -> tuple[float | None, int | None] | None:
    """Return (discovery_sharpe, discovery_trades) for a genome, or None if unavailable."""
    result = state.get_backtest_result(discovery_run_id)
    if result is None:
        return None
    notes_raw = result.get("notes")
    if not isinstance(notes_raw, str) or not notes_raw:
        return None
    try:
        notes = json.loads(notes_raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(notes, dict):
        return None
    strategies = notes.get("top_strategies")
    if not isinstance(strategies, list):
        return None
    if strategy_index < 0 or strategy_index >= len(strategies):
        return None
    entry = strategies[strategy_index]
    if not isinstance(entry, dict):
        return None
    # bd vibe-quant-rewru: prefer the full-range headline (one continuous backtest
    # over the whole discovery range) so we compare like-for-like with the screening
    # replay. Fall back to the multi-window aggregate sharpe/trades for runs
    # persisted before rewru, which lack the full_range_* fields.
    sharpe_raw = entry.get("full_range_sharpe", entry.get("sharpe"))
    trades_raw = entry.get("full_range_trades", entry.get("trades"))
    sharpe = float(sharpe_raw) if isinstance(sharpe_raw, (int, float)) else None
    trades = int(trades_raw) if isinstance(trades_raw, (int, float)) else None
    return sharpe, trades


def _load_screening_metrics(
    state: StateManager, run_id: int
) -> tuple[float | None, int | None]:
    """Return (screening_sharpe, screening_trades) for the best sweep row."""
    sweeps = state.get_sweep_results(run_id)
    if not sweeps:
        return None, None
    best = sweeps[0]  # ordered by sharpe DESC
    sharpe_raw = best.get("sharpe_ratio")
    trades_raw = best.get("total_trades")
    sharpe = float(sharpe_raw) if isinstance(sharpe_raw, (int, float)) else None
    trades = int(trades_raw) if isinstance(trades_raw, (int, float)) else None
    return sharpe, trades


def _build_drift_payload(
    discovery_sharpe: float | None,
    discovery_trades: int | None,
    screening_sharpe: float | None,
    screening_trades: int | None,
) -> dict[str, object]:
    """Compute drift ratios + flagged verdict from the four metric values."""
    trade_ratio = _safe_ratio(screening_trades, discovery_trades)
    sharpe_ratio = _safe_ratio(screening_sharpe, discovery_sharpe)
    trade_flagged = trade_ratio is not None and trade_ratio < TRADE_DRIFT_THRESHOLD
    sharpe_flagged = (
        sharpe_ratio is not None and sharpe_ratio < SHARPE_DRIFT_THRESHOLD
    )
    return {
        "discovery_sharpe": discovery_sharpe,
        "discovery_trades": discovery_trades,
        "screening_sharpe": screening_sharpe,
        "screening_trades": screening_trades,
        "trade_ratio": trade_ratio,
        "sharpe_ratio": sharpe_ratio,
        "trade_drift_threshold": TRADE_DRIFT_THRESHOLD,
        "sharpe_drift_threshold": SHARPE_DRIFT_THRESHOLD,
        "flagged": trade_flagged or sharpe_flagged,
    }


def _store_drift_payload(
    state: StateManager, run_id: int, payload: dict[str, object]
) -> None:
    """Persist drift payload in backtest_results.notes if present, else in parameters."""
    conn = state.conn
    row = conn.execute(
        "SELECT notes FROM backtest_results WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is not None and row[0]:
        try:
            notes = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            notes = {}
        if not isinstance(notes, dict):
            notes = {}
        notes["replay_drift"] = payload
        conn.execute(
            "UPDATE backtest_results SET notes = ? WHERE run_id = ?",
            (json.dumps(notes), run_id),
        )
        conn.commit()
        return

    row2 = conn.execute(
        "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    params: dict[str, object] = {}
    if row2 and row2[0]:
        try:
            parsed = json.loads(row2[0])
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        if isinstance(parsed, dict):
            params = parsed
    params["replay_drift"] = payload
    conn.execute(
        "UPDATE backtest_runs SET parameters = ? WHERE id = ?",
        (json.dumps(params), run_id),
    )
    conn.commit()


def check_replay_drift(state: StateManager, run_id: int) -> dict[str, object] | None:
    """If `run_id` is a post-promote screening run, compute & persist drift vs discovery.

    Returns the drift payload dict when a check ran, else None (no promote_source,
    or missing discovery metrics). Non-critical: swallows unexpected exceptions
    so drift accounting never fails a screening run.
    """
    try:
        run = state.get_backtest_run(run_id)
        if run is None:
            return None
        source = _get_promote_source(run.get("parameters"))
        if source is None:
            return None

        disc_run_id_raw = source.get("discovery_run_id")
        idx_raw = source.get("strategy_index")
        if not isinstance(disc_run_id_raw, (int, float)) or not isinstance(
            idx_raw, (int, float)
        ):
            return None
        discovery_run_id = int(disc_run_id_raw)
        strategy_index = int(idx_raw)

        disc_metrics = _load_discovery_metrics(state, discovery_run_id, strategy_index)
        if disc_metrics is None:
            logger.warning(
                "replay_drift: discovery run=%d idx=%d metrics unavailable",
                discovery_run_id, strategy_index,
            )
            return None
        discovery_sharpe, discovery_trades = disc_metrics
        screening_sharpe, screening_trades = _load_screening_metrics(state, run_id)

        payload = _build_drift_payload(
            discovery_sharpe, discovery_trades, screening_sharpe, screening_trades
        )
        _store_drift_payload(state, run_id, payload)
        if payload["flagged"]:
            logger.warning(
                "replay_drift: run=%d FLAGGED (trade_ratio=%s sharpe_ratio=%s)",
                run_id, payload["trade_ratio"], payload["sharpe_ratio"],
            )
        else:
            logger.info(
                "replay_drift: run=%d clean (trade_ratio=%s sharpe_ratio=%s)",
                run_id, payload["trade_ratio"], payload["sharpe_ratio"],
            )
        return payload
    except Exception:
        logger.exception("replay_drift: unexpected failure on run=%d", run_id)
        return None
