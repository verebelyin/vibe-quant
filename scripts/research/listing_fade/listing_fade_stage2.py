"""Stage-2 evaluator for hypothesis ticket H-20261029-1 (new-listing fade).

Ticket: scripts/research/listing_fade/H-20261029-1.json (SHA-256
dfce29c8617aae85221c7935c2b854159571ce114e9d88304a06d6541be57083).
Pre-registration commit: cbc6c248 (docs/discovery-journal.md, "PRE-REGISTRATION
H-20261029-1 stage 2 (new-listing fade, confirmation sample 2024-01..2026-09)").
Implementation conventions: SPEC_IMPL_STAGE2.md, which overrides SPEC_IMPL.md
(stage 1) wherever the two differ.

Recomputes stage 1 fail-closed, evaluates the 2024-01..2026-09 confirmation
sample, the pooled tests, the BTC-hedged calendar-time book and the verdict.
Prints one JSON object on stdout. The downloaded files are read as data only.
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
SPEC_DIR = ROOT / "scripts/research/listing_fade"
S1_KLINE_DIRS = (ROOT / "data/swarm/20261011-h9-raw", ROOT / "data/swarm/20261011-h10b-raw")
H10B_DIR = ROOT / "data/swarm/20261011-h10b-raw"
S1_FUNDING_DIRS = (
    ROOT / "data/swarm/20261011-h11-funding-a",
    ROOT / "data/swarm/20261011-h11-funding-b",
)
S2_RAW = ROOT / "data/swarm/20261102-listing-s2/raw"
S2_KLINE_DIR = S2_RAW / "klines"
S2_FUNDING_DIR = S2_RAW / "funding"
S2_FILL_DIR = S2_RAW / "klines_fill"

MONTH_FILE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})\.csv$")
FILL_FILE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})-(\d{2})\.csv$")
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
CAL_COST = 0.0005

S2_D0_FLOOR = (date(2024, 1, 1) - EPOCH).days
S2_D0_CEIL = (date(2026, 9, 1) - EPOCH).days
CAL_START = (date(2023, 1, 1) - EPOCH).days
CAL_END = (date(2026, 9, 30) - EPOCH).days

STAGE1_N = 112
STAGE1_MEAN_D = 0.06444773338270435
STAGE1_PLACEBO_N = 101
STAGE1_PLACEBO_MEAN_D = 0.015016243777912584
REPRO_TOL = 1e-12


def _day(year: int, month: int, day: int) -> int:
    return (date(year, month, day) - EPOCH).days


def _month_key(day: int) -> int:
    d = EPOCH + timedelta(days=day)
    return d.year * 12 + d.month - 1


def _quarter_key(day: int) -> int:
    d = EPOCH + timedelta(days=day)
    return d.year * 4 + (d.month - 1) // 3


def _read_kline_file(path: Path, rows: dict[int, tuple[float, float, float]]) -> None:
    """Add day rows to `rows`, the first row for a day key winning."""
    with path.open(encoding="utf-8", errors="replace") as fh:
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


def load_monthly_klines(directory: Path) -> dict[str, dict[int, tuple[float, float, float]]]:
    """symbol -> day -> (close, quote_volume, count) from monthly files, month ascending."""
    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    for name in sorted(os.listdir(directory)):
        match = MONTH_FILE_RE.match(name)
        if match is None:
            continue
        _read_kline_file(directory / name, out.setdefault(match.group(1), {}))
    return out


def add_fill_klines(directory: Path, klines: dict[str, dict[int, tuple[float, float, float]]]) -> int:
    """Add daily fill rows for interior missing days; existing days win. Returns rows used."""
    used = 0
    for name in sorted(os.listdir(directory)):
        match = FILL_FILE_RE.match(name)
        if match is None:
            continue
        rows = klines.setdefault(match.group(1), {})
        before = len(rows)
        _read_kline_file(directory / name, rows)
        used += len(rows) - before
    return used


def load_funding_dir(directory: Path) -> dict[str, dict[int, float]]:
    """symbol -> settlement hour (ms) -> rate, first row per settlement winning."""
    out: dict[str, dict[int, float]] = {}
    for name in sorted(os.listdir(directory)):
        match = MONTH_FILE_RE.match(name)
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


def _dir_symbols(directory: Path) -> set[str]:
    return {m.group(1) for name in os.listdir(directory) if (m := MONTH_FILE_RE.match(name))}


def main() -> None:
    klines: dict[str, dict[int, tuple[float, float, float]]] = {}
    for directory in S1_KLINE_DIRS:
        for symbol, rows in load_monthly_klines(directory).items():
            klines.setdefault(symbol, {}).update(rows)
    for symbol, rows in load_monthly_klines(S2_KLINE_DIR).items():
        klines.setdefault(symbol, {}).update(rows)
    fill_rows = add_fill_klines(S2_FILL_DIR, klines)

    funding_raw: dict[str, dict[int, float]] = {}
    for directory in S1_FUNDING_DIRS:
        for symbol, rows in load_funding_dir(directory).items():
            funding_raw.setdefault(symbol, {}).update(rows)
    for symbol, rows in load_funding_dir(S2_FUNDING_DIR).items():
        funding_raw.setdefault(symbol, {}).update(rows)
    funding = {symbol: funding_prefix(rows) for symbol, rows in funding_raw.items()}

    valid_days: dict[str, list[int]] = {}
    first_day: dict[str, int] = {}
    for symbol, rows in klines.items():
        days = sorted(day for day in rows if has_data(rows, day))
        valid_days[symbol] = days
        if days:
            first_day[symbol] = days[0]

    local_names = _dir_symbols(S1_KLINE_DIRS[0]) | _dir_symbols(H10B_DIR)
    s1_universe = sorted(symbol for symbol in first_day if symbol in local_names)

    spec = json.loads((SPEC_DIR / "stage2_symbols.json").read_text(encoding="utf-8"))
    exclusions = json.loads((SPEC_DIR / "stage2_exclusions.json").read_text(encoding="utf-8"))
    tradfi = set(exclusions["tradfi"])
    pre_market = set(exclusions["pre_market"])
    universe_names = local_names | set(spec["S2"]) | set(spec["P"])
    s2_universe = sorted(
        symbol for symbol in first_day if symbol in universe_names and symbol not in tradfi
    )

    def funding_sum(symbol: str, low: int, high: int) -> tuple[float, int]:
        entry = funding.get(symbol)
        if entry is None:
            return 0.0, 0
        times, cumulative = entry
        lo = bisect.bisect_left(times, low)
        hi = bisect.bisect_left(times, high)
        return cumulative[hi] - cumulative[lo], hi - lo

    def short_return(
        symbol: str, entry_day: int, exit_day: int, drift: bool = False
    ) -> tuple[float, bool]:
        rows = klines[symbol]
        c_entry = rows[entry_day][0]
        c_exit = rows[exit_day][0]
        if drift:
            c_exit = c_exit * DRIFT ** ((exit_day - entry_day) / 27.0)
        rate, count = funding_sum(symbol, (entry_day + 1) * DAY_MS, (exit_day + 1) * DAY_MS)
        return max(FLOOR, -(c_exit / c_entry - 1.0) - COST + rate), count == 0

    def gross_return(symbol: str, entry_day: int, exit_day: int) -> float:
        rows = klines[symbol]
        c_entry = rows[entry_day][0]
        c_exit = rows[exit_day][0]
        return max(FLOOR, -(c_exit / c_entry - 1.0) - COST)

    def pool_for(symbol: str, entry_day: int, exit_day: int, universe: list[str]) -> list[str]:
        pool = []
        for other in universe:
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
        candidates: list[tuple[str, int]],
        entry_offset: int,
        exit_offset: int,
        universe: list[str],
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
            pool = pool_for(symbol, entry_day, exit_day, universe)
            if len(pool) < MIN_POOL:
                excluded["pool_lt_5"] += 1
                continue
            picks = rng.choice(len(pool), size=CONTROLS, replace=False)
            returns = [short_return(pool[i], entry_day, exit_day)[0] for i in picks]
            grosses = [gross_return(pool[i], entry_day, exit_day) for i in picks]
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
                    "r_event_gross": gross_return(symbol, entry_day, exit_day),
                    "r_control_gross": float(np.mean(grosses)),
                }
            )
        return excluded, records

    def bootstrap_p(records: list[dict[str, object]], key_fn, seed: int) -> float:
        values = np.array([record["d"] for record in records], dtype=float)
        cluster_keys = [key_fn(int(record["entry"])) for record in records]
        keys = sorted(set(cluster_keys))
        index = {key: i for i, key in enumerate(keys)}
        m = len(keys)
        sums = np.zeros(m)
        counts = np.zeros(m)
        for value, key in zip(values, cluster_keys, strict=True):
            sums[index[key]] += value
            counts[index[key]] += 1
        rng_b = np.random.default_rng(seed)
        picks = rng_b.integers(0, m, size=(N_BOOT, m))
        boot = sums[picks].sum(1) / counts[picks].sum(1)
        return float((1 + int((boot <= 0).sum())) / (N_BOOT + 1))

    def pooled_power(records: list[dict[str, object]]) -> float:
        d_values = np.array([record["d"] for record in records], dtype=float)
        mean_d = float(d_values.mean())
        drift_delta = np.array(
            [
                short_return(str(record["symbol"]), int(record["entry"]), int(record["exit"]), True)[0]
                - float(record["r_event"])
                for record in records
            ]
        )
        planted = d_values - mean_d + drift_delta
        months = [_month_key(int(record["entry"])) for record in records]
        keys = sorted(set(months))
        m = len(keys)
        planted_by_month = [
            planted[[i for i, month in enumerate(months) if month == key]] for key in keys
        ]
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
        return hits / N_POWER

    def calendar_book(records: list[dict[str, object]]) -> dict[str, object]:
        events = [
            (str(record["symbol"]), int(record["entry"]), int(record["exit"])) for record in records
        ]

        def close_at(symbol: str, day: int) -> float:
            days = valid_days[symbol]
            return klines[symbol][days[bisect.bisect_right(days, day) - 1]][0]

        def day_funding(symbol: str, day: int) -> float:
            return funding_sum(symbol, day * DAY_MS, (day + 1) * DAY_MS)[0]

        returns: list[float] = []
        unhedged: list[float] = []
        active_days = 0
        first_prev = [
            i
            for i, (_, entry, exit_day) in enumerate(events)
            if entry < CAL_START - 1 <= exit_day
        ]
        prev_n = len(first_prev)
        prev_weights: dict[int, float] = (
            {i: 1.0 / prev_n for i in first_prev} if prev_n else {}
        )
        for t in range(CAL_START, CAL_END + 1):
            current = [i for i, (_, entry, exit_day) in enumerate(events) if entry < t <= exit_day]
            n = len(current)
            if n == 0:
                returns.append(0.0)
                unhedged.append(0.0)
                prev_weights = {}
                prev_n = 0
                continue
            active_days += 1
            weights = {i: 1.0 / n for i in current}
            short_sum = 0.0
            for i in current:
                symbol = events[i][0]
                short_sum += -(close_at(symbol, t) / close_at(symbol, t - 1) - 1.0) + day_funding(
                    symbol, t
                )
            short_book = short_sum / n
            btc = (
                close_at("BTCUSDT", t) / close_at("BTCUSDT", t - 1)
                - 1.0
                - day_funding("BTCUSDT", t)
            )
            event_turnover = sum(
                abs(weights.get(i, 0.0) - prev_weights.get(i, 0.0))
                for i in set(weights) | set(prev_weights)
            )
            btc_prev = 1.0 if prev_n >= 1 else 0.0
            cost = CAL_COST * (event_turnover + abs(1.0 - btc_prev))
            returns.append(short_book + btc - cost)
            unhedged.append(short_book - CAL_COST * event_turnover)
            prev_weights = weights
            prev_n = n

        r = np.array(returns, dtype=float)
        u = np.array(unhedged, dtype=float)
        mean_r = float(r.mean())
        return {
            "days": len(returns),
            "active_days": active_days,
            "sharpe_252": mean_r / float(r.std(ddof=1)) * float(np.sqrt(252)),
            "sharpe_365": mean_r / float(r.std(ddof=1)) * float(np.sqrt(365)),
            "ann_return": mean_r * 365,
            "unhedged_sharpe_252": float(u.mean()) / float(u.std(ddof=1)) * float(np.sqrt(252)),
        }

    # Stage 1 reproduction (fail closed).
    h10b = _dir_symbols(H10B_DIR)
    s1_candidates = sorted(
        ((symbol, first_day[symbol]) for symbol in h10b if symbol in first_day),
        key=lambda item: item[1],
    )
    s1_candidates = [item for item in s1_candidates if D0_FLOOR < item[1] <= D0_CEIL]
    _, s1_records = evaluate(s1_candidates, ENTRY, EXIT, s1_universe)
    s1_d = np.array([record["d"] for record in s1_records], dtype=float)
    s1_mean_d = float(s1_d.mean())
    _, s1_placebo_records = evaluate(s1_candidates, PLACEBO_ENTRY, PLACEBO_EXIT, s1_universe)
    s1_placebo_mean_d = float(np.mean([record["d"] for record in s1_placebo_records]))
    stage1_ok = bool(
        len(s1_records) == STAGE1_N
        and abs(s1_mean_d - STAGE1_MEAN_D) <= REPRO_TOL
        and len(s1_placebo_records) == STAGE1_PLACEBO_N
        and abs(s1_placebo_mean_d - STAGE1_PLACEBO_MEAN_D) <= REPRO_TOL
    )
    stage1_check = {
        "n": len(s1_records),
        "mean_D": s1_mean_d,
        "placebo_n": len(s1_placebo_records),
        "placebo_mean_D": s1_placebo_mean_d,
        "ok": stage1_ok,
    }
    if not stage1_ok:
        print(json.dumps({"stage1_check": stage1_check, "verdict": "INVALID"}))
        return

    # Stage-2 candidates and the four pre-data exclusions.
    excluded_s2 = dict.fromkeys(
        (
            "tradfi",
            "pre_market",
            "no_data",
            "d0_out_of_window",
            "short_data",
            "no_entry_close",
            "no_exit",
            "pool_lt_5",
        ),
        0,
    )
    candidates = sorted(set(spec["candidates"]))
    s2_base: list[tuple[str, int]] = []
    for symbol in candidates:
        if symbol in tradfi:
            excluded_s2["tradfi"] += 1
            continue
        if symbol in pre_market:
            excluded_s2["pre_market"] += 1
            continue
        if symbol not in first_day:
            excluded_s2["no_data"] += 1
            continue
        d0 = first_day[symbol]
        if d0 > S2_D0_CEIL or d0 < S2_D0_FLOOR:
            excluded_s2["d0_out_of_window"] += 1
            continue
        s2_base.append((symbol, d0))

    s2_stage_excluded, s2_records = evaluate(s2_base, ENTRY, EXIT, s2_universe)
    excluded_s2.update(s2_stage_excluded)

    s2_d = np.array([record["d"] for record in s2_records], dtype=float)
    s2_r_events = np.array([record["r_event"] for record in s2_records], dtype=float)
    s2_r_controls = np.array([record["r_control"] for record in s2_records], dtype=float)
    s2_mean_d = float(s2_d.mean())

    s2_per_year: dict[str, dict[str, object]] = {}
    for value, record in zip(s2_d, s2_records, strict=True):
        year = str((EPOCH + timedelta(days=int(record["entry"]))).year)
        bucket = s2_per_year.setdefault(year, {"n": 0, "sum": 0.0})
        bucket["n"] = int(bucket["n"]) + 1
        bucket["sum"] = float(bucket["sum"]) + value
    s2_per_year_out = {
        year: {"n": int(bucket["n"]), "mean_D": float(bucket["sum"]) / int(bucket["n"])}
        for year, bucket in sorted(s2_per_year.items())
    }

    _, hold_records = evaluate(s2_base, ENTRY, HOLDS["hold14"], s2_universe)
    hold14_mean_d = float(np.mean([record["d"] for record in hold_records]))
    _, hold_records = evaluate(s2_base, ENTRY, HOLDS["hold60"], s2_universe)
    hold60_mean_d = float(np.mean([record["d"] for record in hold_records]))

    stage2_out = {
        "n_candidates": len(candidates),
        "excluded": excluded_s2,
        "n": len(s2_records),
        "mean_D": s2_mean_d,
        "median_D": float(np.median(s2_d)),
        "p_boot": bootstrap_p(s2_records, _month_key, SEED_BOOT),
        "squeeze_share": float((s2_r_events <= -0.5).mean()),
        "r_event_p10": float(np.percentile(s2_r_events, 10)),
        "mean_R_event": float(s2_r_events.mean()),
        "mean_R_control": float(s2_r_controls.mean()),
        "per_year": s2_per_year_out,
        "hold14_mean_D": hold14_mean_d,
        "hold60_mean_D": hold60_mean_d,
        "funding_missing": sum(int(record["funding_missing"]) for record in s2_records),
    }

    # Pooled tests.
    pooled_records = s1_records + s2_records
    pooled_d = np.array([record["d"] for record in pooled_records], dtype=float)
    pooled_r_events = np.array([record["r_event"] for record in pooled_records], dtype=float)
    pooled_gross = np.array(
        [
            float(record["r_event_gross"]) - float(record["r_control_gross"])
            for record in pooled_records
        ],
        dtype=float,
    )
    pooled_out = {
        "n": len(pooled_records),
        "mean_D": float(pooled_d.mean()),
        "median_D": float(np.median(pooled_d)),
        "p_boot": bootstrap_p(pooled_records, _month_key, SEED_BOOT),
        "p_boot_quarter": bootstrap_p(pooled_records, _quarter_key, SEED_BOOT),
        "power": pooled_power(pooled_records),
        "squeeze_share": float((pooled_r_events <= -0.5).mean()),
        "r_event_p10": float(np.percentile(pooled_r_events, 10)),
        "mean_D_gross_of_funding": float(pooled_gross.mean()),
    }

    # Placebo.
    s2_placebo_excluded, s2_placebo_records = evaluate(
        s2_base, PLACEBO_ENTRY, PLACEBO_EXIT, s2_universe
    )
    pooled_placebo_records = s1_placebo_records + s2_placebo_records
    placebo_out = {
        "stage2_n": len(s2_placebo_records),
        "stage2_excluded": s2_placebo_excluded,
        "stage2_mean_D": float(np.mean([record["d"] for record in s2_placebo_records])),
        "pooled_n": len(pooled_placebo_records),
        "pooled_mean_D": float(np.mean([record["d"] for record in pooled_placebo_records])),
    }

    calendar_out = calendar_book(pooled_records)

    pre_market_base: list[tuple[str, int]] = []
    for symbol in sorted(pre_market):
        if symbol not in first_day:
            continue
        d0 = first_day[symbol]
        if d0 > S2_D0_CEIL or d0 < S2_D0_FLOOR:
            continue
        pre_market_base.append((symbol, d0))
    _, pre_market_records = evaluate(pre_market_base, ENTRY, EXIT, s2_universe)
    pre_market_out = {
        "n": len(pre_market_records),
        "mean_D": float(np.mean([record["d"] for record in pre_market_records])),
    }

    checks = {
        "stage2_mean_D": bool(stage2_out["mean_D"] >= 0.02),
        "pooled_mean_D": bool(pooled_out["mean_D"] >= 0.04),
        "pooled_p_boot": bool(pooled_out["p_boot"] <= 0.05),
        "pooled_median_D": bool(pooled_out["median_D"] > 0),
        "pooled_squeeze": bool(pooled_out["squeeze_share"] <= 0.12),
        "calendar_sharpe": bool(calendar_out["sharpe_252"] >= 0.8),
    }

    if pooled_out["power"] < 0.80 or pooled_out["n"] < 150:
        verdict = "INSUFFICIENT"
    elif not all(checks.values()):
        verdict = "REJECT"
    elif abs(placebo_out["pooled_mean_D"]) > 0.02:
        verdict = "INSUFFICIENT"
    else:
        verdict = "PASS"

    out = {
        "stage1_check": stage1_check,
        "stage2": stage2_out,
        "pooled": pooled_out,
        "placebo": placebo_out,
        "calendar": calendar_out,
        "pre_market": pre_market_out,
        "fill_rows": fill_rows,
        "checks": checks,
        "verdict": verdict,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
