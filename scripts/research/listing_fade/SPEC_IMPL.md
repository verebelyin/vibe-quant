# H-20261029-1 stage 1 implementation conventions (frozen 2026-10-29, before any evaluator; coverage facts only, no prices or returns analysed)

Pre-registration: docs/discovery-journal.md "PRE-REGISTRATION H-20261029-1 stage 1". Ticket: `H-20261029-1.json` here (SHA-256 dfce29c8617aae85221c7935c2b854159571ce114e9d88304a06d6541be57083). These conventions pin details the ticket leaves open so two independent implementations agree exactly; where they differ from the ticket, the journal entry says so.

## Inputs (all read-only, no download)
- Klines (1d): `data/swarm/20261011-h10b-raw/<SYM>-YYYY-MM.csv` (event coins + aged controls, files 2022-05..2026-09) and `data/swarm/20261011-h9-raw/<SYM>-YYYY-MM.csv` (controls, 2020-01..2026-09). Ignore `UNIVERSE.txt`. Symbol = filename minus `-YYYY-MM.csv`.
- Funding: `data/swarm/20261011-h11-funding-b/` (h10b coins) and `data/swarm/20261011-h11-funding-a/` (h9 coins), same file naming.
- CSV parsing: skip any line whose first field is not all digits (some files have a header, some do not). Kline columns by position: 0 open_time, 4 close, 7 quote_volume, 8 count. open_time >= 1e14 means microseconds (divide by 1000). Day key `d = open_time_ms // 86_400_000`. Duplicate day keys: keep the first in (file month ascending, line order). A day "has data" iff count > 0 and close > 0.
- Funding columns: 0 calc_time (ms), 2 last_funding_rate. Settlement time `s = round(calc_time / 3_600_000) * 3_600_000`. Duplicate s: keep the first.
- `T_d = (d + 1) * 86_400_000` (end of UTC day d).

## Events
- `first_i` = first day with data for symbol i. Event iff i is in h10b-raw and `first_i > day(2022-05-01)` (a coin whose first day is 2022-05-01 may predate the file window; it is not an event). D0 = first_i; stage 1 keeps D0 <= 2023-12-31.
- Exclude (count each reason): fewer than 20 days with data after D0 (any later day) -> `short_data`; no data on E = D0+2 -> `no_entry_close`; no day with data in (E, X], X = D0+29 -> `no_exit`.
- Exit day x = the last day with data in (E, X] (x = X unless data ends or gaps at X).
- No product-title exclusions in stage 1 (Pre-Market/TradFi/Pre-IPO launches start in 2024).

## Returns (short, constant quantity, no stop)
`R = max(-1.0, -(c_x / c_E - 1) - 0.0010 + sum(rate_s for settlements s with T_E <= s < T_x))`. The funding term is on entry notional (ticket approximation). A coin with zero settlements in that interval adds 0 and is counted in `funding_missing`.

## Controls
- Pool for an event (E, x): every symbol j in h9-raw or h10b-raw, j != event symbol, with `first_j <= E - 365`, data on all 30 days E-29..E with mean quote_volume over them >= 5_000_000, and data on E and on x. Controls use the same E and x and the same R formula with their own funding.
- Events sorted by (E, symbol). `rng = numpy.random.default_rng(20261029)`. Per event: pool sorted by symbol; if len(pool) < 5 exclude (`pool_lt_5`) and make NO rng call; else `idx = rng.choice(len(pool), size=5, replace=False)`.
- `D = R_event - mean(R_controls)`.

## Test
- Clusters = calendar month of E, sorted ascending; S_m = sum of D, n_m = count, M = number of months.
- Bootstrap: `rng_b = default_rng(20261030)`; `picks = rng_b.integers(0, M, size=(10000, M))`; `boot = S[picks].sum(1) / n[picks].sum(1)`; `p_boot = (1 + (boot <= 0).sum()) / 10001`.
- Power: per event, drifted closes `c'_d = c_d * 0.9 ** ((d - E) / 27)` for E < d <= X (entry unchanged); R' from c'_x with the same funding and floor; `Delta = R' - R`; planted `P = D - mean(D) + Delta`. `rng_p = default_rng(20261031)`; 200 times: `outer = rng_p.integers(0, M, size=M)`; cluster k = all events of month `outer[k]` (repeats are distinct clusters) giving S'_k, n'_k; `inner = rng_p.integers(0, M, size=(10000, M))`; `p_r = (1 + (S'[inner].sum(1) / n'[inner].sum(1) <= 0).sum()) / 10001`. `power = mean(p_r <= 0.05)`.
- Age placebo: same event symbols and D0, E_p = D0+400, X_p = D0+427, same exclusion, exit and pool rules (fresh `default_rng(20261029)`, events sorted by (E_p, symbol)); report n and mean D.
- Report-only (fresh `default_rng(20261029)` each): hold 14 d (X = D0+16) and 60 d (X = D0+62) mean D; median D; 10th percentile of R_event (`numpy.percentile`, linear); squeeze share = share of events with R_event <= -0.5; per-year (year of E) n and mean D; mean R_event, mean R_control.

## Verdict (first match)
1. power < 0.80 -> INSUFFICIENT. 2. n < 80 -> INSUFFICIENT. 3. mean D < 0.02 or p_boot > 0.5 -> REJECT. 4. |placebo mean D| > 0.02 -> INSUFFICIENT (method suspect). 5. else CONTINUE (stage 2 needs the user's OK to ingest; not evidence of an edge).

## Output
One JSON object on stdout, floats unrounded: `n_candidates, excluded{short_data,no_entry_close,no_exit,pool_lt_5}, n, mean_D, median_D, p_boot, power, placebo{n,excluded,mean_D}, squeeze_share, r_event_p10, mean_R_event, mean_R_control, per_year{YYYY:{n,mean_D}}, hold14_mean_D, hold60_mean_D, funding_missing, verdict`.
