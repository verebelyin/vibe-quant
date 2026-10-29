# H-20261029-2 implementation conventions (frozen 2026-10-29, before any evaluator; coverage facts only, no prices or returns analysed)

Pre-registration: docs/discovery-journal.md "PRE-REGISTRATION H-20261029-2". Ticket: `H-20261029-2.json` here (SHA-256 658339573bd59d36625de7e97746fd4310ac38baa737ae2054b7cb05946fc059). These conventions pin details the ticket leaves open so two independent implementations agree exactly; where they differ from the ticket, the journal entry says so.

## Inputs (read-only, no download)
- U189: klines `data/swarm/20261011-h10b-raw/`, funding `data/swarm/20261011-h11-funding-b/`. U66: klines `data/swarm/20261011-h9-raw/`, funding `data/swarm/20261011-h11-funding-a/`. Files `<SYM>-YYYY-MM.csv`; ignore `UNIVERSE.txt`.
- CSV parsing exactly as `scripts/research/listing_fade/SPEC_IMPL.md` § Inputs (header skip, column positions, µs open_time, day key, first duplicate wins, "has data" = count > 0 and close > 0, settlement time rounded to the hour, `T_d = (d + 1) * 86_400_000`).
- Daily return `r_i,d = c_i,d / c_i,d-1 - 1` iff both days have data, else missing.

## Book (one function, reused by every variant)
- Decision days t = Mondays (UTC) with start <= t <= end. Window days d = start..end inclusive; R_d = 0 when no book is held.
- Rankable at t: r_i,d present for all 28 days d in [t-27, t], data on all 7 days t-6..t and mean quote_volume over them >= 10_000_000.
- Rank rankable coins by (score, symbol) ascending; n = count, k = n // 3. k = 0: no book that week. Long the first k at +0.5/k each, short the last k at -0.5/k each. MAX score = max r_i,d over [t-27, t].
- Holding days d = t+1..t+7. Coin i adds `w_i * r_i,d` while active; on its first holding day with r_i,d missing it drops out for the rest of the week (weight 0 from that day).
- Funding: settlement s belongs to day `s // 86_400_000`; an active coin adds `-w_i * rate_s` to that day. Capped variant: rate clipped to [-0.001, 0.001]. No rows: adds 0.
- Costs 0.0005 per unit of weight traded: on decision day t, `0.0005 * sum_i |w_new,i - w_held,i|`, where w_held is the old book's weight still active on day t (0 if dropped or none); on a dropout day d, `0.0005 * |w_i|`. Holding days after end are cut; no final exit cost.
- `R_d = sum(w * r) + funding - costs`. Sharpe = `mean(R) / std(R, ddof=1) * sqrt(365)` over the window days.
- EW_d = mean r_i,d over universe coins with r_i,d present (no filter). beta = OLS slope of R_d on EW_d with intercept.

## Runs
- U189 B: start 2022-06-01, end 2026-09-30: `sharpe_net` (uncapped), `sharpe_capped`, `sharpe_gross` (no funding, no costs; report-only), beta.
- U66 FULL: start 2020-10-01, end 2026-09-30; report Sharpe on the FULL series and on its slices A = 2020-10-01..2022-05-31 and B = 2022-06-01..2026-09-30 (slices of one series, not separate runs).
- Permutation null (U189 B, uncapped): `rng = default_rng(20261029)`; for each decision Monday ascending with k >= 1: `perms = rng.permuted(numpy.tile(numpy.arange(n), (20000, 1)), axis=1)`; draw r scores = (scores of the rankable coins sorted by symbol)[perms[r]], then the same ranking rule. `p_perm = (1 + #{null Sharpe >= sharpe_net}) / 20001`.
- Anti-rediscovery (U189 B): MOM book with score = -(c_t / c_t-28 - 1) (long winners; needs data on t-28 too), FUND book with score = sum of rate_s for T_t-7 <= s < T_t (short high funding; no rows = 0). Weekly series per book = sum of R_d over d = t+1..t+7 for decision Mondays with t + 7 <= end; EW weekly the same. `numpy.linalg.lstsq` of MAX weekly on [1, EW, MOM, FUND]; `resid_sharpe = alpha / sqrt(SSR / (n_weeks - 4)) * sqrt(52)`.

## Construction checks (computed first; any failure -> print only the checks and verdict INVALID)
- Oracle (U189 B): score = -(c_t+7 / c_t - 1) (0 if no data on t+7) + eps, `eps = rng_o.standard_normal(n) * sd_t`, sd_t = std(ddof=1) of those next-week returns over the rankable coins, `rng_o = default_rng(7)` per Monday ascending. Need Sharpe >= 3 and permutation p <= 0.001 with 2,000 draws (`default_rng(8)`, p denominator 2001).
- Random (U189 B): for seed in 1000..1099: score = `default_rng(seed).standard_normal(n)` per Monday ascending; permutation p with 1,000 draws (`default_rng(seed + 100000)`, denominator 1001). Need share of seeds with p > 0.05 >= 0.90.

## Verdict
PASS iff all: sharpe_net >= 1.0; sharpe_capped >= 0.8; |beta| < 0.3; U66 FULL Sharpe >= 0.5; p_perm <= 0.05; resid_sharpe >= 0.5. Else REJECT.

## Output
One JSON object on stdout, floats unrounded: `checks{oracle_sharpe, oracle_p, random_share_p_gt_005}, u189{sharpe_net, sharpe_capped, sharpe_gross, beta, p_perm, mean_k, n_weeks_with_book, funding_ann, cost_ann}, u66{sharpe_full, sharpe_A, sharpe_B}, resid{alpha_weekly, resid_sharpe}, verdict` (`funding_ann` / `cost_ann` = mean daily funding / cost x 365).
