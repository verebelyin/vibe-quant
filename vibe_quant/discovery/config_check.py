"""Pure window-config checks for discovery (vibe-quant-zhcq7).

Cross-window / WFA / eval-window knobs that can't produce valid windows used
to surface only AFTER the whole GA (the gates failed closed -> 0 champions).
These helpers derive the windows without a pipeline so the CLI, the API and
``DiscoveryPipeline._run`` can all reject a bad config BEFORE any evaluation.

``DiscoveryConfigError`` is a plain ``ValueError`` and deliberately NOT a
``DataUnavailableError``: it is a config problem, never missing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from vibe_quant.utils import split_date_range, split_into_windows

if TYPE_CHECKING:
    from collections.abc import Sequence

# Shortest shifted cross-window (days) worth backtesting
MIN_CROSS_WINDOW_DAYS: int = 7


class DiscoveryConfigError(ValueError):
    """Discovery window/gate config is invalid (abort before the GA)."""


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def assert_windows_outside_holdout(
    windows: Sequence[tuple[str, str]], holdout_start: str, label: str
) -> None:
    """Raise if any [start, end) window reaches past ``holdout_start``.

    Windows are half-open: a window ending exactly at the holdout start (the
    train end) does not overlap it.
    """
    if not holdout_start:
        return
    for ws, we in windows:
        if we > holdout_start or ws >= holdout_start:
            msg = (
                f"{label} window {ws}..{we} overlaps the holdout starting "
                f"{holdout_start} -- the holdout must stay unseen until the final gate"
            )
            raise ValueError(msg)


def cross_window_ranges_for(
    months_list: Sequence[int],
    start_date: str,
    end_date: str,
    holdout_start: str = "",
) -> list[tuple[int, str, str]]:
    """Shifted cross-windows ``(offset_months, start, end)`` inside the TRAIN range.

    Window k starts ``offset_k`` months after the train start and ends at
    the train end -- shifting the END forward would run into the holdout
    (or past the data), so windows are clipped to the train range.

    Raises:
        ValueError: A window is shorter than ``MIN_CROSS_WINDOW_DAYS`` or
            (defensively) overlaps the holdout.
    """
    from dateutil.relativedelta import relativedelta

    base_start = _parse_date(start_date)
    base_end = _parse_date(end_date)
    windows: list[tuple[int, str, str]] = []
    for months in months_list:
        ws_dt = base_start + relativedelta(months=months)
        if base_end - ws_dt < timedelta(days=MIN_CROSS_WINDOW_DAYS):
            msg = (
                f"Cross-window: +{months}mo window {ws_dt:%Y-%m-%d}..{end_date} "
                f"shorter than {MIN_CROSS_WINDOW_DAYS}d inside the train range "
                f"{start_date}..{end_date}"
            )
            raise ValueError(msg)
        windows.append((months, ws_dt.strftime("%Y-%m-%d"), end_date))
    if holdout_start:
        assert_windows_outside_holdout(
            [(ws, we) for _, ws, we in windows], holdout_start, "Cross-window",
        )
    return windows


def wfa_window_ranges_for(
    step_days: int,
    range_start: str,
    range_end: str,
    holdout_start: str = "",
) -> list[tuple[str, str]]:
    """Rolling ``step_days`` windows tiling [range_start, range_end]."""
    if step_days < 1:
        raise ValueError(f"WFA step must be >= 1 day, got {step_days}")
    start = _parse_date(range_start)
    end = _parse_date(range_end)
    windows: list[tuple[str, str]] = []
    current = start
    while current + timedelta(days=step_days) <= end:
        nxt = current + timedelta(days=step_days)
        windows.append((current.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        current = nxt
    if holdout_start:
        assert_windows_outside_holdout(windows, holdout_start, "WFA")
    return windows


def parse_cross_window_months(raw: str) -> list[int]:
    """Parse the ``--cross-window-months`` string ("1,3,6") into ints.

    Raises:
        DiscoveryConfigError: A token is not an integer.
    """
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise DiscoveryConfigError(
                f"cross_window_months: {token!r} is not an integer"
            ) from None
    return out


def cross_window_months_problems(months_list: Sequence[int]) -> list[str]:
    """Problems with the raw months list: values < 1 and duplicates."""
    problems: list[str] = []
    bad = [m for m in months_list if m < 1]
    if bad:
        problems.append(f"cross_window_months must be >= 1, got {bad}")
    dupes = sorted({m for m in months_list if list(months_list).count(m) > 1})
    if dupes:
        problems.append(f"cross_window_months has duplicates: {dupes}")
    return problems


@dataclass(frozen=True)
class WindowPlan:
    """Windows derived from a raw discovery date range + window knobs."""

    train_start: str
    train_end: str
    holdout_start: str | None
    holdout_end: str | None
    eval_windows: list[tuple[str, str]] | None
    cross_windows: list[tuple[int, str, str]]
    wfa_windows: list[tuple[str, str]]


def plan_windows(
    start_date: str,
    end_date: str,
    *,
    train_test_split: float,
    eval_windows: int,
    cross_window_months: Sequence[int],
    wfa_oos_step_days: int,
) -> WindowPlan:
    """Derive every window discovery will use; collect ALL problems.

    Uses the same ``split_date_range`` / ``split_into_windows`` as the CLI, so
    dates are byte-identical to what the run will use.

    Raises:
        DiscoveryConfigError: One error listing every problem, joined by "; ".
    """
    problems = cross_window_months_problems(cross_window_months)
    train_start, train_end = start_date, end_date
    holdout_start: str | None = None
    holdout_end: str | None = None
    split_ok = True
    if train_test_split > 0:
        try:
            train_start, train_end, holdout_start, holdout_end = split_date_range(
                start_date, end_date, train_test_split,
            )
        except ValueError as exc:
            problems.append(str(exc))
            split_ok = False

    eval_list: list[tuple[str, str]] | None = None
    cross: list[tuple[int, str, str]] = []
    wfa: list[tuple[str, str]] = []
    if split_ok:
        n_eval = max(1, eval_windows)
        if n_eval >= 2:
            try:
                eval_list = split_into_windows(train_start, train_end, n_eval)
            except ValueError as exc:
                problems.append(f"eval_windows: {exc}")
        valid_months = [m for m in cross_window_months if m >= 1]
        try:
            cross = cross_window_ranges_for(
                list(dict.fromkeys(valid_months)), train_start, train_end, holdout_start or "",
            )
        except ValueError as exc:
            problems.append(str(exc))
        if wfa_oos_step_days > 0:
            try:
                wfa = wfa_window_ranges_for(
                    wfa_oos_step_days, train_start, train_end, holdout_start or "",
                )
                if not wfa:
                    problems.append(
                        f"WFA: range {train_start}..{train_end} too short for "
                        f"{wfa_oos_step_days}d windows"
                    )
            except ValueError as exc:
                problems.append(str(exc))
        elif wfa_oos_step_days < 0:
            problems.append(f"wfa_oos_step_days must be >= 0, got {wfa_oos_step_days}")

    if problems:
        raise DiscoveryConfigError("; ".join(problems))
    return WindowPlan(
        train_start=train_start,
        train_end=train_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        eval_windows=eval_list,
        cross_windows=cross,
        wfa_windows=wfa,
    )
