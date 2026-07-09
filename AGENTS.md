# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` to load workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

**Version:** bd 1.0.0+ — embedded Dolt backend, no daemon, no `bd sync`. Auto-commits locally on every mutation. Rules: use `bd` for ALL task tracking (no TodoWrite / markdown lists); use `bd remember` for persistent project knowledge.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

**Optional: memory durability.** `.beads/embeddeddolt/` is gitignored, so memories from `bd remember` are machine-local by default. If you want them backed up off-machine, run `bd dolt push` (writes to `refs/dolt/data` on the same remote). Not part of the mandatory flow — matches beads team's default.

## Hard-Won Session Playbooks (2026-07-09 — read before repeating these jobs)

Full details live in `CLAUDE.md` (NautilusTrader Gotchas, Performance Playbook,
Verification Rules, Dev Server Gotchas) and in `bd prime` memories
(`gotcha:nt-1230-upgrade`, `command:perf-profiling-playbook`, `gotcha:dev-servers`,
`pipeline:determinism`, `policy:lint-zero`). Condensed checklist:

**Dependency upgrades:**
1. Record baselines FIRST (`pytest` pass count, `ruff`/`mypy` error lists to files) — diff after each phase, never eyeball.
2. Stage: batch minor/patch → verify → each major separately → verify. Hold majors with explicit pins in the install command.
3. Read the changelog for Python-facing breaks BEFORE bumping (NT publishes no migration guide — RELEASES.md breaking-change sections are it).
4. After upgrading: run a REAL run through the app (E2E), not just tests — the indicator-pool bug, the config-field crash, and the 300× data bug were all invisible to the unit suite.

**Performance hunting:**
1. Engine log lines are ground truth (`Added N Bar elements`, `Iterations`).
2. Coarse phase timing → cProfile on SHORT windows only → fix the top item → prove exactness (bit-identical metrics + zero-tolerance tests) → measure again.
3. Never trust `s/chromosome` (wall across 8 workers) or cross-run discovery comparisons (unseeded RNG).

**E2E / UI testing:**
1. Verify which app is on the port before driving it (5173 may be another project).
2. Restart the backend after plugin/endpoint changes; subprocesses pick up code without restart.
3. Kill servers with `pkill -f "uvicorn vibe_quant"`, never by port.
4. When a run fails with a vague error, the real one is usually swallowed — check `logs/<mode>_<run_id>_*.log` and set `raise_exception=True` in new NT runner code.

**Discovery cycles:**
- `bd recall discovery:champions` for current rankings; journal every batch in `docs/discovery-journal.md`.
- A champion that needed `no_bootstrap_ci=true` will probably collapse in validation — the runner auto-flags this; believe the flag.
- Reddit scraping 403s from this IP (recorded as failed now); research extractions in the DB are the working corpus.

## Custom Indicators (Plugin System)

To add a new indicator: drop a `.py` file in `vibe_quant/dsl/plugins/`. The file registers an `IndicatorSpec` with `compute_fn`, `param_ranges`, and `threshold_range`. Auto-loaded at startup, auto-enrolled in GA discovery, auto-exposed via `/api/indicators/catalog`. See `vibe_quant/dsl/plugins/README.md` for the full field reference and a working example (`example_adaptive_rsi.py`).
