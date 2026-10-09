# Swarm board

## #decisions

- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: --symbol-agg worst: GA ranks 0.5·min+0.5·median per-symbol fitness; champions need every symbol positive in train+holdout. Use --eval-windows 1 with worst mode (per-symbol 3-window hard gates leave almost no gradient).  <sub>18dcca1e772467d8-4588</sub>
- `2026-10-09T08:56:24` **chief** @20261009-mimo-swarm: Routing: MiMo V2.6 Pro (cmd --tier pro) is the default implementer for well-specified tasks; Claude implementers for semantics-critical/open-ended work; Opus reviews everything.  <sub>18dcca1e7b9ce808-4590</sub>

## #findings

- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: Profile (strategy 239, 4h): pandas-path indicators ≈97% of eval time, parquet load <1%. ADX exact port: 28.4s→16.5s. Next hotspots: KAMA prep, PRICE_POSITION.  <sub>18dcca1e6a07a7b8-4568</sub>
- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: Discovery early exit on certain-zero windows: same-seed run 81s→46s, identical results.  <sub>18dcca1e6e63c260-4570</sub>
- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: Research 2026-10-08/09: STOCH+CCI champion negative post-audit; H-1 regime gate, H-2 trend ensemble (<0.7), H-3 funding fade (crowded longs CONTINUED), H-5 squeeze, H-7 BTC gate all rejected; run 878 worst-mode soft score: gradient yes, edge no (DSR p≈0.98).  <sub>18dcca1e72c74390-4586</sub>

## #gotchas

- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: Exact numpy ports: pandas ewm on arm64 uses fused multiply-add — use math.fma; derive alpha like pandas (com=(1-a)/a); NaN-skipping rma seed. Always add a one-shot self-check that falls back to the pandas path (see compute_builtins._adx_port_ok).  <sub>18dcca1e586b3790-4560</sub>
- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: Catalog used to hold an in-progress (partial) current-day bar; nearest-rounding made it look like a completed close = look-ahead. Fixed (iz66w): never written now. Any as-of lookup must ceil-round close ts and drop bars whose ts_init+1ms isn't on a boundary.  <sub>18dcca1e5ccf5500-4562</sub>
- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: NautilusTrader swallows exceptions inside on_bar → a data error mid-run silently becomes 0 trades. Preflight data coverage BEFORE the engine starts and let DataUnavailableError propagate (runner, grid, discovery fitness, gate stages).  <sub>18dcca1e613a6b98-4564</sub>
- `2026-10-09T08:56:23` **chief** @20261009-mimo-swarm: A funding-archive refresh alone moves exactness baselines (bars unchanged). Prove bars-only by pairing new catalog + old archive; re-baseline with the reason in CLAUDE.md.  <sub>18dcca1e65a16b50-4566</sub>

## #model-notes

- `2026-10-09T08:56:24` **chief** @20261009-mimo-swarm: MiMo V2.6 Pro: 3/4 then 5/5 clean first passes; one stall (37 min thinking, 0 edits) on open-ended framework design; no adjustable effort (--effort makes cmd fail).  <sub>18dcca1e8065e998-4592</sub>
- `2026-10-09T08:56:24` **chief** @20261009-mimo-swarm: DeepSeek 4.1 Flash @max: correct code but 0.5–2.9M input tokens/task; first drafts were too slow / under-tested → needed reviewer-guided fix rounds. Qwen 3.8 Max 0902 @xhigh (supports low/medium/xhigh): peer of MiMo, ~38% more tokens.  <sub>18dcca1e8547d9f8-4646</sub>

