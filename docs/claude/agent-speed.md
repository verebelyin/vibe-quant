# Agent speed: timings, mutants, E2E, test time

CLAUDE.md § Agent speed rules keeps the always-on rules (gate once per round, never run `pytest tests/unit` yourself, fewer fatter turns, short proofs). This file holds the numbers and the narrower rules.

Measured (load 4–5): full `tests/unit` ≈ 98 s under `-n auto` + 33 s timing set, ≈ 10 min serial · `swarm-check.sh` ≈ 20–30 s typical, ≈ 50 s on a hub module (`discovery/__main__.py`), ≈ 135 s `--full`, seconds on a cached PASS (a message-only amend stays cached) · `exactness_239.py` ≈ 6 s · Playwright spec ≈ 1.5 min · one model turn ≈ 4–10 s.

- **Mutants:** `scripts/agents/mutate.py --file F --old X --new Y -- <targeted test>`; ≤ 5 per review, ≤ 3 per maker, only on changed logic. A surviving mutant outside money code becomes a follow-up bead, not a fix round. Stress loops ≤ 10 runs unless flakiness is the finding.
- **E2E:** iterate with `-g <scenario>`, run the full spec once at the end. Tests of anything that returns 201 and runs in a subprocess poll the run's final status.
- **Measure per-test time serially:** `--durations` under `-n auto` is inflated by CPU contention.
- **Timings under parallel jobs** quote the load they ran under: `scripts/agents/cpu_log.py summary --log data/swarm/<job>/cpu.jsonl --since HH:MM` (the chief runs `cpu_log.py sample`).
