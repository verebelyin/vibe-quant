"""Stage-1 evaluator for hypothesis ticket H-20261029-1 (new-listing fade).

Ticket: scripts/research/listing_fade/H-20261029-1.json (SHA-256
dfce29c8617aae85221c7935c2b854159571ce114e9d88304a06d6541be57083).
Pre-registration commit: a97bfc49 (docs/discovery-journal.md, "PRE-REGISTRATION
H-20261029-1 stage 1 ..."). Implementation conventions: SPEC_IMPL.md beside the ticket.

Short the close of D0+2 and cover at the last close <= D0+29, against five date-matched
aged-coin shorts; test the per-event difference D with a month-cluster bootstrap, a planted
-10 % drift (power) and an age placebo. Prints one JSON object on stdout.
"""

from __future__ import annotations

import bisect
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
KLINE_DIRS = (ROOT / "data/swarm/20261011-h9-raw", ROOT / "data/swarm/20261011-h10b-raw")
H10B_DIR = ROOT / "data/swarm/20261011-h10b-raw"
FUNDING_DIRS = (ROOT / "data/swarm/20261011-h11-funding-a", ROOT / "data/swarm/20261011-h11-funding-b")

FILE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})\.csv$")
EPOCH = date(1970, 1, 1)
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
COST = 0.0010
FLOOR = -1.0
MIN_AGE_DAYS = 365
CONTROLS = 5
MIN_POOL = 5
VOL_DAYS = 30
MIN_MEAN_QV = 5_000_000.0
D0_FLOOR = (date(2022, 5, 1) - EPOCH).days
D0_CEIL = (date(2023, 12, 31) - EPOCH).days
ENTRY = 2
EXIT = 29
PLACEBO_ENTRY = 400
PLACEBO_EXIT = 427
HOLDS = {"hold14": 16, "hold60": 62}
SEED_CONTROLS = 20261029
SEED_BOOT = 20261030
SEED_POWER = 20261031
N_BOOT = 10_000
N_POWER = 200
POWER_BAR = 0.05
DRIFT = 0.9


def _day(y: int, m: int, d: int) -> int:
    return (date(y, m, d) - EPOCH).days


def _month_key(day: int) -> int:
    d = EPOCH + timedelta(days=day)
    return d.year * 12 + d.month - 1


def load_klines(directory: Path) -> dict[str, dict[int, tuple[float, float, float]]]:
    """symbol -> day -> (close, quote_volume, count), first row per day winning."""
    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    for name in sorted(os.listdir(directory)):
        match = FILE_RE.match(name)
        if match is None:
            continue
        rows = out.setdefault(match.group(1), {})
        with (directory / name).open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.split(",", 1)[0].isdigit():
                    continue
                fields = line.rstrip("\r\n").split(",")
                open_time = int(fields[0])
                if open_time >= 100_000_000_000_000:
                    open_time //= 1000
                day = open_time // DAY_MS
                if day in rows:
                    continue
                rows[day] = (float(fields[4]), float(fields[7]), float(fields[8]))
    return out


def load_funding(directory: Path) -> dict[str, dict[int, float]]:
    """symbol -> settlement hour (ms) -> rate, first row per settlement winning."""
    out: dict[str, dict[int, float]] = {}
    for name in sorted(os.listdir(directory)):
        match = FILE_RE.match(name)
        if match is None:
            continue
        rows = out.setdefault(match.group(1), {})
        with (directory / name).open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.split(",", 1)[0].isdigit():
                    continue
                fields = line.rstrip("\r\n").split(",")
                settlement = round(int(fields[0]) / HOUR_MS) * HOUR_MS
                if settlement in rows:
                    continue
                rows[settlement] = float(fields[2])
    return out


def funding_prefix(rows: dict[int, float]) -> tuple[list[int], list[float]]:
    times = sorted(rows)
    cumulative = [0.0]
    for settlement in times:
        cumulative.append(cumulative[-1] + rows[settlement])
    return times, cumulative


def has_data(rows: dict[int, tuple[float, float, float]], day: int) -> bool:
    row = rows.get(day)
    return row is not None and row[2] > 0 and row[0] > 0


def main() -> None:
    klines: dict[str, dict[int, tuple[float, float, float]]] = {}
    for directory in KLINE_DIRS:
        for symbol, rows in load_klines(directory).items():
            klines.setdefault(symbol, {}).update(rows)

    funding_raw: dict[str, dict[int, float]] = {}
    for directory in FUNDING_DIRS:
        for symbol, rows in load_funding(directory).items():
            funding_raw.setdefault(symbol, {}).update(rows)
    funding = {symbol: funding_prefix(rows) for symbol, rows in funding_raw.items()}

    valid_days: dict[str, list[int]] = {}
    first_day: dict[str, int] = {}
    for symbol, rows in klines.items():
        days = sorted(day for day in rows if has_data(rows, day))
        valid_days[symbol] = days
        if days:
            first_day[symbol] = days[0]
    symbols = sorted(first_day)

    h10b = {m.group(1) for name in os.listdir(H10B_DIR) if (m := FILE_RE.match(name))}

    def funding_sum(symbol: str, low: int, high: int) -> tuple[float, int]:
        entry = funding.get(symbol)
        if entry is None:
            return 0.0, 0
        times, cumulative = entry
        lo = bisect.bisect_left(times, low)
        hi = bisect.bisect_left(times, high)
        return cumulative[hi] - cumulative[lo], hi - lo

    def short_return(symbol: str, entry_day: int, exit_day: int, drift: bool = False) -> tuple[
        float, bool
    ]:
        rows = klines[symbol]
        c_entry = rows[entry_day][0]
        c_exit = rows[exit_day][0]
        if drift:
            c_exit = c_exit * DRIFT ** ((exit_day - entry_day) / 27.0)
        rate, count = funding_sum(symbol, (entry_day + 1) * DAY_MS, (exit_day + 1) * DAY_MS)
        return max(FLOOR, -(c_exit / c_entry - 1.0) - COST + rate), count == 0

    def pool_for(symbol: str, entry_day: int, exit_day: int) -> list[str]:
        pool = []
        for other in symbols:
            if other == symbol:
                continue
            if first_day[other] > entry_day - MIN_AGE_DAYS:
                continue
            rows = klines[other]
            total = 0.0
            for day in range(entry_day - VOL_DAYS + 1, entry_day + 1):
                row = rows.get(day)
                if row is None or row[2] <= 0 or row[0] <= 0:
                    break
                total += row[1]
            else:
                if total / VOL_DAYS < MIN_MEAN_QV:
                    continue
                if not has_data(rows, exit_day):
                    continue
                pool.append(other)
        return pool

    def evaluate(
        candidates: list[tuple[str, int]], entry_offset: int, exit_offset: int
    ) -> tuple[dict[str, int], list[dict[str, object]]]:
        excluded = dict.fromkeys(("short_data", "no_entry_close", "no_exit", "pool_lt_5"), 0)
        staged = []
        for symbol, d0 in candidates:
            rows = klines[symbol]
            days = valid_days[symbol]
            if len(days) - bisect.bisect_right(days, d0) < 20:
                excluded["short_data"] += 1
                continue
            entry_day = d0 + entry_offset
            if not has_data(rows, entry_day):
                excluded["no_entry_close"] += 1
                continue
            limit = d0 + exit_offset
            keep = bisect.bisect_left(days, limit + 1) - 1
            exit_day = days[keep] if keep >= 0 else -1
            if exit_day <= entry_day:
                excluded["no_exit"] += 1
                continue
            staged.append((symbol, d0, entry_day, exit_day))

        staged.sort(key=lambda item: (item[2], item[0]))
        rng = np.random.default_rng(SEED_CONTROLS)
        records = []
        for symbol, d0, entry_day, exit_day in staged:
            pool = pool_for(symbol, entry_day, exit_day)
            if len(pool) < MIN_POOL:
                excluded["pool_lt_5"] += 1
                continue
            picks = rng.choice(len(pool), size=CONTROLS, replace=False)
            returns = [short_return(pool[i], entry_day, exit_day)[0] for i in picks]
            event_return, missing = short_return(symbol, entry_day, exit_day)
            records.append(
                {
                    "symbol": symbol,
                    "d0": d0,
                    "entry": entry_day,
                    "exit": exit_day,
                    "r_event": event_return,
                    "r_control": float(np.mean(returns)),
                    "d": event_return - float(np.mean(returns)),
                    "funding_missing": missing,
                }
            )
        return excluded, records

    candidates = sorted(
        ((symbol, first_day[symbol]) for symbol in h10b if symbol in first_day),
        key=lambda item: item[1],
    )
    candidates = [item for item in candidates if D0_FLOOR < item[1] <= D0_CEIL]

    excluded, records = evaluate(candidates, ENTRY, EXIT)
    n_candidates = len(candidates)
    n = len(records)
    d_values = np.array([record["d"] for record in records], dtype=float)
    r_events = np.array([record["r_event"] for record in records], dtype=float)
    r_controls = np.array([record["r_control"] for record in records], dtype=float)
    mean_d = float(d_values.mean())
    median_d = float(np.median(d_values))

    months = [_month_key(int(record["entry"])) for record in records]
    keys = sorted(set(months))
    index = {key: i for i, key in enumerate(keys)}
    m = len(keys)
    sums = np.zeros(m)
    counts = np.zeros(m)
    for value, month in zip(d_values, months, strict=True):
        sums[index[month]] += value
        counts[index[month]] += 1

    rng_b = np.random.default_rng(SEED_BOOT)
    picks = rng_b.integers(0, m, size=(N_BOOT, m))
    boot = sums[picks].sum(1) / counts[picks].sum(1)
    p_boot = float((1 + int((boot <= 0).sum())) / (N_BOOT + 1))

    drift_delta = np.array(
        [
            short_return(str(record["symbol"]), int(record["entry"]), int(record["exit"]), True)[0]
            - float(record["r_event"])
            for record in records
        ]
    )
    planted = d_values - mean_d + drift_delta
    planted_by_month = [planted[[i for i, month in enumerate(months) if month == key]] for key in keys]

    rng_p = np.random.default_rng(SEED_POWER)
    hits = 0
    for _ in range(N_POWER):
        outer = rng_p.integers(0, m, size=m)
        sums_outer = np.empty(m)
        counts_outer = np.empty(m)
        for k, month_index in enumerate(outer):
            values = planted_by_month[int(month_index)]
            sums_outer[k] = values.sum()
            counts_outer[k] = values.size
        inner = rng_p.integers(0, m, size=(N_BOOT, m))
        boot_outer = sums_outer[inner].sum(1) / counts_outer[inner].sum(1)
        p_r = (1 + int((boot_outer <= 0).sum())) / (N_BOOT + 1)
        if p_r <= POWER_BAR:
            hits += 1
    power = hits / N_POWER

    placebo_excluded, placebo_records = evaluate(candidates, PLACEBO_ENTRY, PLACEBO_EXIT)
    placebo_d = [float(record["d"]) for record in placebo_records]

    report: dict[str, object] = {}
    for name, offset in HOLDS.items():
        _, hold_records = evaluate(candidates, ENTRY, offset)
        report[f"{name}_mean_D"] = float(np.mean([record["d"] for record in hold_records]))

    per_year: dict[str, dict[str, object]] = {}
    for value, record in zip(d_values, records, strict=True):
        year = str((EPOCH + timedelta(days=int(record["entry"]))).year)
        bucket = per_year.setdefault(year, {"n": 0, "sum": 0.0})
        bucket["n"] = int(bucket["n"]) + 1
        bucket["sum"] = float(bucket["sum"]) + value
    per_year_out = {
        year: {"n": int(bucket["n"]), "mean_D": float(bucket["sum"]) / int(bucket["n"])}
        for year, bucket in sorted(per_year.items())
    }

    if power < 0.80 or n < 80:
        verdict = "INSUFFICIENT"
    elif mean_d < 0.02 or p_boot > 0.5:
        verdict = "REJECT"
    elif abs(float(np.mean(placebo_d))) > 0.02:
        verdict = "INSUFFICIENT"
    else:
        verdict = "CONTINUE"

    out = {
        "n_candidates": n_candidates,
        "excluded": excluded,
        "n": n,
        "mean_D": mean_d,
        "median_D": median_d,
        "p_boot": p_boot,
        "power": power,
        "placebo": {
            "n": len(placebo_records),
            "excluded": placebo_excluded,
            "mean_D": float(np.mean(placebo_d)),
        },
        "squeeze_share": float((r_events <= -0.5).mean()),
        "r_event_p10": float(np.percentile(r_events, 10)),
        "mean_R_event": float(r_events.mean()),
        "mean_R_control": float(r_controls.mean()),
        "per_year": per_year_out,
        "hold14_mean_D": report["hold14_mean_D"],
        "hold60_mean_D": report["hold60_mean_D"],
        "funding_missing": sum(int(record["funding_missing"]) for record in records),
        "verdict": verdict,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
