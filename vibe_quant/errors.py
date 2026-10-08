"""Shared run-level error types."""

from __future__ import annotations


class DataUnavailableError(ValueError):
    """Input data a run needs is missing (bars, funding archive, ...).

    A configuration/data problem, not a backtest failure: runners re-raise it
    instead of masking it as a -inf or zero-trade result.
    """
