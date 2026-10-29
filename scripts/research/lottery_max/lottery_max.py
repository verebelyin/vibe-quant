"""Evaluator for ticket H-20261029-2: lottery MAX cross-section (weekly dollar-neutral book).

Pre-registration commit a97bfc49 (docs/discovery-journal.md, 2026-10-29).
Frozen conventions: scripts/research/lottery_max/SPEC_IMPL.md.

Reads the read-only daily kline and funding CSVs under data/swarm/20261011-*, builds the
weekly long/short tercile book (short the highest 28-day maximum daily return), the
score-permutation null, the anti-rediscovery regression and the two construction checks,
and prints one JSON object on stdout.

Run: .venv/bin/python scripts/research/lottery_max/lottery_max.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
LOOKBACK = 28
VOL_MIN = 10_000_000.0
COST = 0.0005
FUND_CAP = 0.001
SEED_NULL = 20261029
SEED_ORACLE = 7
SEED_ORACLE_NULL = 8
SEED_RANDOM_BASE = 1000
SEED_RANDOM_NULL_OFFSET = 100_000
N_RANDOM_SEEDS = 100
DRAWS_NULL = 20_000
DRAWS_ORACLE = 2_000
DRAWS_RANDOM = 1_000
SLICE_A_END = "2022-05-31"
B_START = "2022-06-01"
END = "2026-09-30"
U66_START = "2020-10-01"

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "swarm"
U189_KLINES = DATA / "20261011-h10b-raw"
U189_FUNDING = DATA / "20261011-h11-funding-b"
U66_KLINES = DATA / "20261011-h9-raw"
U66_FUNDING = DATA / "20261011-h11-funding-a"

_EPOCH_ORD = date(1970, 1, 1).toordinal()
_NAME = re.compile(r"^(.+)-(\d{4})-(\d{2})\.csv$")


def day_number(iso: str) -> int:
    return date.fromisoformat(iso).toordinal() - _EPOCH_ORD


def _month_index(year: int, month: int) -> int:
    return year * 12 + month - 1


def _month_bounds(lo_day: int, hi_day: int) -> tuple[int, int]:
    lo = date.fromordinal(lo_day + _EPOCH_ORD)
    hi = date.fromordinal(hi_day + _EPOCH_ORD)
    return _month_index(lo.year, lo.month), _month_index(hi.year, hi.month)


def _read_csv(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    with path.open("rb") as fh:
        first = fh.readline().decode("utf-8", "replace")
    skip = 0 if first.split(",")[0].strip().isdigit() else 1
    try:
        return pd.read_csv(
            path,
            header=None,
            skiprows=skip,
            usecols=cols,
            names=names,
            dtype=np.float64,
            engine="c",
        )
    except (ValueError, pd.errors.ParserError):
        frame = pd.read_csv(
            path, header=None, usecols=cols, names=names, dtype=str, engine="python"
        )
        for col in names:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna()


@dataclass
class Universe:
    symbols: list[str]
    lo_day: int
    n_days: int
    n_coins: int
    close: np.ndarray
    has: np.ndarray
    qv: np.ndarray
    ret: np.ndarray
    fund: np.ndarray
    fund_cap: np.ndarray


def load_universe(klines_dir: Path, funding_dir: Path, lo_day: int, hi_day: int) -> Universe:
    lo_mi, hi_mi = _month_bounds(lo_day, hi_day)
    symbols: set[str] = set()
    kline_paths: list[tuple[Path, str]] = []
    for path in sorted(klines_dir.iterdir()):
        match = _NAME.match(path.name)
        if match is None:
            continue
        symbols.add(match.group(1))
        mi = _month_index(int(match.group(2)), int(match.group(3)))
        if lo_mi <= mi <= hi_mi:
            kline_paths.append((path, match.group(1)))

    frames = []
    for path, sym in kline_paths:
        frame = _read_csv(path, [0, 4, 7, 8], ["t", "c", "qv", "n"])
        frame["sym"] = sym
        frames.append(frame)
    klines = pd.concat(frames, ignore_index=True)
    stamp = klines["t"].to_numpy()
    stamp = np.where(stamp >= 1e14, stamp / 1000.0, stamp)
    klines["day"] = np.floor(stamp / DAY_MS).astype(np.int64)
    klines = klines[(klines["day"] >= lo_day) & (klines["day"] <= hi_day)]
    klines = klines.drop_duplicates(subset=["sym", "day"], keep="first")

    fund_paths = []
    for path in sorted(funding_dir.iterdir()):
        match = _NAME.match(path.name)
        if match is None:
            continue
        mi = _month_index(int(match.group(2)), int(match.group(3)))
        if lo_mi <= mi <= hi_mi:
            fund_paths.append((path, match.group(1)))

    fund_frames = []
    for path, sym in fund_paths:
        frame = _read_csv(path, [0, 2], ["t", "rate"])
        frame["sym"] = sym
        fund_frames.append(frame)
    if fund_frames:
        funding = pd.concat(fund_frames, ignore_index=True)
        settle = np.rint(funding["t"].to_numpy() / HOUR_MS) * HOUR_MS
        funding["s"] = settle.astype(np.int64)
        funding = funding.drop_duplicates(subset=["sym", "s"], keep="first")
        funding["day"] = funding["s"] // DAY_MS
        funding["rate_cap"] = funding["rate"].clip(-FUND_CAP, FUND_CAP)
        funding = funding[(funding["day"] >= lo_day) & (funding["day"] <= hi_day)]
        funding = funding.groupby(["sym", "day"], as_index=False)[["rate", "rate_cap"]].sum()
    else:
        funding = pd.DataFrame({"sym": [], "day": [], "rate": [], "rate_cap": []})

    ordered = sorted(symbols)
    index = {sym: i for i, sym in enumerate(ordered)}
    n_days = hi_day - lo_day + 1
    shape = (len(ordered), n_days)
    close = np.full(shape, np.nan)
    qv = np.full(shape, np.nan)
    count = np.zeros(shape)
    fund = np.zeros(shape)
    fund_cap = np.zeros(shape)

    rows = klines["sym"].map(index).to_numpy()
    cols = (klines["day"] - lo_day).to_numpy()
    close[rows, cols] = klines["c"].to_numpy()
    qv[rows, cols] = klines["qv"].to_numpy()
    count[rows, cols] = klines["n"].to_numpy()

    if len(funding):
        rows = funding["sym"].map(index).to_numpy()
        cols = (funding["day"] - lo_day).to_numpy()
        fund[rows, cols] = funding["rate"].to_numpy()
        fund_cap[rows, cols] = funding["rate_cap"].to_numpy()

    has = (count > 0) & (close > 0)
    qv = np.where(has, qv, 0.0)
    ret = np.full(shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = close[:, 1:] / close[:, :-1] - 1.0
    ok = has[:, 1:] & has[:, :-1]
    ret[:, 1:] = np.where(ok, ratio, np.nan)

    return Universe(
        symbols=ordered,
        lo_day=lo_day,
        n_days=n_days,
        n_coins=len(ordered),
        close=close,
        has=has,
        qv=qv,
        ret=ret,
        fund=fund,
        fund_cap=fund_cap,
    )


@dataclass(frozen=True)
class Week:
    pos: int
    cand: np.ndarray
    k: int
    max_score: np.ndarray
    keep: np.ndarray
    drop: np.ndarray


def build_weeks(univ: Universe, start_pos: int, end_pos: int) -> list[Week]:
    weeks = []
    for pos in range(start_pos, end_pos + 1):
        if (univ.lo_day + pos + 3) % 7 != 0:
            continue
        cand = np.zeros(0, dtype=np.int64)
        max_score = np.zeros(0)
        if pos >= LOOKBACK:
            window = univ.ret[:, pos - LOOKBACK + 1 : pos + 1]
            all28 = ~np.isnan(window).any(axis=1)
            data7 = univ.has[:, pos - 6 : pos + 1].all(axis=1)
            vol7 = univ.qv[:, pos - 6 : pos + 1].mean(axis=1)
            rankable = all28 & data7 & (vol7 >= VOL_MIN)
            cand = np.flatnonzero(rankable)
            max_score = window[cand].max(axis=1)
        k = cand.size // 3
        offer = min(7, end_pos - pos, univ.n_days - 1 - pos)
        keep = np.zeros((univ.n_coins, 7))
        drop = np.zeros((univ.n_coins, 7))
        if offer > 0:
            present = ~np.isnan(univ.ret[:, pos + 1 : pos + 1 + offer])
            keep[:, :offer] = np.cumprod(present, axis=1).astype(bool)
            prior = np.ones((univ.n_coins, 7), dtype=bool)
            prior[:, 1:offer] = keep[:, : offer - 1].astype(bool)
            drop[:, :offer] = prior[:, :offer] & ~present
        weeks.append(Week(pos=pos, cand=cand, k=k, max_score=max_score, keep=keep, drop=drop))
    return weeks


def book_weights(univ: Universe, week: Week, score: np.ndarray) -> np.ndarray:
    weights = np.zeros(univ.n_coins)
    if week.k == 0:
        return weights
    value = 0.5 / week.k
    order = np.lexsort((week.cand, score))
    weights[week.cand[order[: week.k]]] = value
    weights[week.cand[order[-week.k :]]] = -value
    return weights


def permuted_weights(
    univ: Universe, week: Week, score: np.ndarray, perms: np.ndarray
) -> np.ndarray:
    n = week.cand.size
    draws = perms.shape[0]
    weights = np.zeros((draws, univ.n_coins))
    value = 0.5 / week.k
    order = np.argsort(score, kind="stable")
    if np.unique(score).size == n:
        long_flag = np.zeros(n, dtype=bool)
        short_flag = np.zeros(n, dtype=bool)
        long_flag[order[: week.k]] = True
        short_flag[order[-week.k :]] = True
        picks = long_flag[perms].astype(float) - short_flag[perms].astype(float)
        weights[:, week.cand] = value * picks
        return weights
    drawn = score[perms]
    ranked = np.argsort(drawn, axis=1, kind="stable")
    rows = np.arange(draws)[:, None]
    weights[rows, week.cand[ranked[:, : week.k]]] = value
    weights[rows, week.cand[ranked[:, -week.k :]]] = -value
    return weights


def run_book(
    univ: Universe,
    weeks: list[Week],
    weight_iter,
    n_draws: int,
    funding: np.ndarray | None,
    *,
    cost: bool,
    daily: bool,
    window: tuple[int, int],
) -> dict:
    """The single book function: returns the daily series or per-draw Sharpe moments."""
    n_days = univ.n_days
    w0, w1 = window
    n_win = w1 - w0 + 1
    total = np.zeros((n_draws, n_days)) if daily else None
    funding_daily = np.zeros((n_draws, n_days)) if daily else None
    cost_daily = np.zeros((n_draws, n_days)) if daily else None
    sums = np.zeros(n_draws)
    sumsq = np.zeros(n_draws)

    prev_weights = None
    for wi, week in enumerate(weeks):
        weights = np.atleast_2d(next(weight_iter))
        pos = week.pos
        if cost:
            held = 0.0 if wi == 0 else prev_weights * weeks[wi - 1].keep[:, 6]
            entry = COST * np.abs(weights - held).sum(axis=1)
            if daily:
                total[:, pos] -= entry
                cost_daily[:, pos] += entry
            else:
                sums -= entry
                sumsq += entry * entry
        abs_weights = np.abs(weights) if cost else None
        offer = min(7, n_days - 1 - pos, w1 - pos)
        for offset in range(offer):
            day = pos + 1 + offset
            keep = week.keep[:, offset]
            vector = weights @ (np.nan_to_num(univ.ret[:, day]) * keep)
            funding_part = None
            if funding is not None:
                funding_part = -(weights @ (funding[:, day] * keep))
                vector = vector + funding_part
            cost_part = None
            if cost:
                cost_part = COST * (abs_weights @ week.drop[:, offset])
                vector = vector - cost_part
            if daily:
                total[:, day] += vector
                if funding_part is not None:
                    funding_daily[:, day] += funding_part
                if cost_part is not None:
                    cost_daily[:, day] += cost_part
            else:
                sums += vector
                sumsq += vector * vector
        prev_weights = weights

    if daily:
        result: dict = {
            "total": total,
            "funding_daily": funding_daily,
            "cost_daily": cost_daily,
        }
        series = total[:, w0 : w1 + 1]
        result["sharpe"] = series.mean(axis=1) / series.std(axis=1, ddof=1) * np.sqrt(365.0)
    else:
        mean = sums / n_win
        variance = (sumsq - sums * sums / n_win) / (n_win - 1)
        result = {"mean": mean, "std": np.sqrt(variance)}
        result["sharpe"] = mean / np.sqrt(variance) * np.sqrt(365.0)
    return result


def perm_nulls(
    univ: Universe,
    weeks: list[Week],
    scores: list[np.ndarray],
    rng,
    draws: int,
    window: tuple[int, int],
) -> np.ndarray:
    def weights():
        for week, score in zip(weeks, scores, strict=True):
            if week.k == 0:
                yield np.zeros((draws, univ.n_coins))
                continue
            n = week.cand.size
            perms = rng.permuted(np.tile(np.arange(n), (draws, 1)), axis=1)
            yield permuted_weights(univ, week, score, perms)

    return run_book(
        univ,
        weeks,
        weights(),
        draws,
        univ.fund,
        cost=True,
        daily=False,
        window=window,
    )["sharpe"]


def scores_momentum(univ: Universe, weeks: list[Week]) -> list[np.ndarray]:
    out = []
    for week in weeks:
        if week.k == 0:
            out.append(np.zeros(0))
            continue
        start = univ.close[week.cand, week.pos - LOOKBACK]
        end = univ.close[week.cand, week.pos]
        out.append(-(end / start - 1.0))
    return out


def scores_funding(univ: Universe, weeks: list[Week]) -> list[np.ndarray]:
    out = []
    for week in weeks:
        if week.k == 0:
            out.append(np.zeros(0))
            continue
        out.append(univ.fund[week.cand, week.pos - 6 : week.pos + 1].sum(axis=1))
    return out


def scores_oracle(univ: Universe, weeks: list[Week]) -> list[np.ndarray]:
    out = []
    for week in weeks:
        n = week.cand.size
        if n == 0:
            out.append(np.zeros(0))
            continue
        later = week.pos + 7
        has_later = univ.has[week.cand, later]
        future = np.where(
            has_later, univ.close[week.cand, later] / univ.close[week.cand, week.pos] - 1.0, 0.0
        )
        sd = float(future.std(ddof=1)) if n > 1 else 0.0
        eps = np.random.default_rng(SEED_ORACLE).standard_normal(n) * sd
        out.append(-future + eps)
    return out


def scores_random(weeks: list[Week], seed: int) -> list[np.ndarray]:
    return [np.random.default_rng(seed).standard_normal(week.cand.size) for week in weeks]


def weight_list(univ: Universe, weeks: list[Week], scores: list[np.ndarray]) -> list:
    return [book_weights(univ, week, score) for week, score in zip(weeks, scores, strict=True)]


def daily_stats(series: np.ndarray) -> float:
    return float(series.mean() / series.std(ddof=1) * np.sqrt(365.0))


def weekly_series(weeks: list[Week], daily: np.ndarray, end_pos: int) -> np.ndarray:
    out = []
    for week in weeks:
        if week.pos + 7 <= end_pos:
            out.append(daily[week.pos + 1 : week.pos + 8].sum())
    return np.asarray(out)


def equal_weight_daily(univ: Universe) -> np.ndarray:
    window = univ.ret
    count = (~np.isnan(window)).sum(axis=0)
    total = np.nansum(window, axis=0)
    out = np.zeros(window.shape[1])
    np.divide(total, count, out=out, where=count > 0)
    return out


def main() -> None:
    b_start = day_number(B_START)
    b_end = day_number(END)
    u189_lo = b_start - 40
    u189_hi = b_end + 7
    u189 = load_universe(U189_KLINES, U189_FUNDING, u189_lo, u189_hi)
    start_pos = b_start - u189_lo
    end_pos = b_end - u189_lo
    window = (start_pos, end_pos)
    win = slice(start_pos, end_pos + 1)
    weeks = build_weeks(u189, start_pos, end_pos)

    # Construction checks first.
    oracle_scores = scores_oracle(u189, weeks)
    oracle_weights = weight_list(u189, weeks, oracle_scores)
    oracle_book = run_book(
        u189,
        weeks,
        iter(oracle_weights),
        1,
        u189.fund,
        cost=True,
        daily=False,
        window=window,
    )
    oracle_sharpe = float(oracle_book["sharpe"][0])
    oracle_null = perm_nulls(
        u189,
        weeks,
        oracle_scores,
        np.random.default_rng(SEED_ORACLE_NULL),
        DRAWS_ORACLE,
        window,
    )
    oracle_p = float((1 + np.count_nonzero(oracle_null >= oracle_sharpe)) / (DRAWS_ORACLE + 1))

    random_hits = 0
    for seed in range(SEED_RANDOM_BASE, SEED_RANDOM_BASE + N_RANDOM_SEEDS):
        scores = scores_random(weeks, seed)
        observed = run_book(
            u189,
            weeks,
            iter(weight_list(u189, weeks, scores)),
            1,
            u189.fund,
            cost=True,
            daily=False,
            window=window,
        )
        observed_sharpe = float(observed["sharpe"][0])
        null = perm_nulls(
            u189,
            weeks,
            scores,
            np.random.default_rng(seed + SEED_RANDOM_NULL_OFFSET),
            DRAWS_RANDOM,
            window,
        )
        p_value = float((1 + np.count_nonzero(null >= observed_sharpe)) / (DRAWS_RANDOM + 1))
        random_hits += int(p_value > 0.05)
    random_share = random_hits / N_RANDOM_SEEDS

    checks = {
        "oracle_sharpe": oracle_sharpe,
        "oracle_p": oracle_p,
        "random_share_p_gt_005": random_share,
    }
    if not (oracle_sharpe >= 3.0 and oracle_p <= 0.001 and random_share >= 0.90):
        print(json.dumps({"checks": checks, "verdict": "INVALID"}, indent=2))
        return

    max_scores = [week.max_score for week in weeks]
    net = run_book(
        u189,
        weeks,
        iter(weight_list(u189, weeks, max_scores)),
        1,
        u189.fund,
        cost=True,
        daily=True,
        window=window,
    )
    capped = run_book(
        u189,
        weeks,
        iter(weight_list(u189, weeks, max_scores)),
        1,
        u189.fund_cap,
        cost=True,
        daily=True,
        window=window,
    )
    gross = run_book(
        u189,
        weeks,
        iter(weight_list(u189, weeks, max_scores)),
        1,
        None,
        cost=False,
        daily=True,
        window=window,
    )
    net_full = net["total"][0]
    net_row = net_full[win]
    sharpe_net = daily_stats(net_row)
    sharpe_capped = daily_stats(capped["total"][0][win])
    sharpe_gross = daily_stats(gross["total"][0][win])

    ew_full = equal_weight_daily(u189)
    ew189 = ew_full[win]
    design = np.column_stack([np.ones(ew189.size), ew189])
    beta = float(np.linalg.lstsq(design, net_row, rcond=None)[0][1])

    p_perm = float(
        (
            1
            + np.count_nonzero(
                perm_nulls(
                    u189,
                    weeks,
                    max_scores,
                    np.random.default_rng(SEED_NULL),
                    DRAWS_NULL,
                    window,
                )
                >= sharpe_net
            )
        )
        / (DRAWS_NULL + 1)
    )

    ks = np.array([week.k for week in weeks])
    u189_block = {
        "sharpe_net": sharpe_net,
        "sharpe_capped": sharpe_capped,
        "sharpe_gross": sharpe_gross,
        "beta": beta,
        "p_perm": p_perm,
        "mean_k": float(ks.mean()),
        "n_weeks_with_book": int(np.count_nonzero(ks >= 1)),
        "funding_ann": float(net["funding_daily"][0][win].mean() * 365.0),
        "cost_ann": float(net["cost_daily"][0][win].mean() * 365.0),
    }

    # Anti-rediscovery regression: MAX vs EW, 28-day momentum and 7-day funding books.
    mom_scores = scores_momentum(u189, weeks)
    fund_scores = scores_funding(u189, weeks)
    mom = run_book(
        u189,
        weeks,
        iter(weight_list(u189, weeks, mom_scores)),
        1,
        u189.fund,
        cost=True,
        daily=True,
        window=window,
    )
    fund_book = run_book(
        u189,
        weeks,
        iter(weight_list(u189, weeks, fund_scores)),
        1,
        u189.fund,
        cost=True,
        daily=True,
        window=window,
    )
    max_weekly = weekly_series(weeks, net_full, end_pos)
    ew_weekly = weekly_series(weeks, ew_full, end_pos)
    mom_weekly = weekly_series(weeks, mom["total"][0], end_pos)
    fund_weekly = weekly_series(weeks, fund_book["total"][0], end_pos)
    n_weeks = max_weekly.size
    factors = np.column_stack([np.ones(n_weeks), ew_weekly, mom_weekly, fund_weekly])
    coef = np.linalg.lstsq(factors, max_weekly, rcond=None)[0]
    resid = max_weekly - factors @ coef
    ssr = float(resid @ resid)
    alpha_weekly = float(coef[0])
    resid_sharpe = alpha_weekly / np.sqrt(ssr / (n_weeks - 4)) * np.sqrt(52.0)

    # U66 replication: one full series, slices are views of it.
    u66_lo = day_number(U66_START) - 40
    u66_hi = b_end + 7
    u66 = load_universe(U66_KLINES, U66_FUNDING, u66_lo, u66_hi)
    u66_start = day_number(U66_START) - u66_lo
    u66_end = b_end - u66_lo
    u66_weeks = build_weeks(u66, u66_start, u66_end)
    u66_max = [week.max_score for week in u66_weeks]
    u66_book = run_book(
        u66,
        u66_weeks,
        iter(weight_list(u66, u66_weeks, u66_max)),
        1,
        u66.fund,
        cost=True,
        daily=True,
        window=(u66_start, u66_end),
    )
    u66_row = u66_book["total"][0]
    a_end_pos = day_number(SLICE_A_END) - u66_lo
    u66_block = {
        "sharpe_full": daily_stats(u66_row[u66_start : u66_end + 1]),
        "sharpe_A": daily_stats(u66_row[u66_start : a_end_pos + 1]),
        "sharpe_B": daily_stats(u66_row[a_end_pos + 1 : u66_end + 1]),
    }

    verdict = (
        "PASS"
        if (
            sharpe_net >= 1.0
            and sharpe_capped >= 0.8
            and abs(beta) < 0.3
            and u66_block["sharpe_full"] >= 0.5
            and p_perm <= 0.05
            and resid_sharpe >= 0.5
        )
        else "REJECT"
    )

    output = {
        "checks": checks,
        "u189": u189_block,
        "u66": u66_block,
        "resid": {"alpha_weekly": alpha_weekly, "resid_sharpe": float(resid_sharpe)},
        "verdict": verdict,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
