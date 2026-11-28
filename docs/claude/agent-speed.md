# Agent speed: timings, mutants, E2E, test time

Always-on rules: CLAUDE.md § Agent speed rules; numbers here.

Measured (load 4–5): full `tests/unit` ≈ 98 s `-n auto` + 33 s timing set, ≈ 10 min serial · `swarm-check.sh` ≈ 20–30 s, one-hop selection (`--full` covers 2-hop callers), ≈ 135 s `--full`, seconds cached · `exactness_239.py` ≈ 6 s · Playwright ≈ 1.5 min.

- **Wall = CPU ÷ cores granted:** the `-n auto` pass burns ≈ 837 s CPU ⇒ 84 s floor on 10 cores; cut test CPU (`-v`, autouse fixtures; `-n 8`: no gain).
- **Mutants (optional):** `scripts/agents/mutate.py --file F --old X --new Y -- <test>`; ≤ 2 per round, only on kill-zone or money code `exactness_239.py` does not cover. Stress loops ≤ 10 runs unless flakiness is the point.
- **E2E:** iterate `-g <scenario>`, full spec once at the end; a 201-in-subprocess test polls the final status.
- **Concurrent gates share cores:** each pytest step registers `data/swarm/_gate-slots/<pid>` and takes `max(2, cpus ÷ gates running)` workers (`SWARM_CHECK_WORKERS` wins).
- **Gate reuse:** red policy tests stop it; a test file's PASS holds until a later diff reaches it (not `--full`).
- **Slow tests:** a new/changed test over 2 s alone fails the gate unless `@pytest.mark.slow(reason=...)`.
- **Chief on main:** `swarm-check.sh .` gates uncommitted edits against `origin/main`.
- **Per-test time only serially:** `-n auto` inflates `--durations`.
- **Quote the load:** `scripts/agents/cpu_log.py summary --log data/swarm/<job>/cpu.jsonl --since HH:MM` (chief: `sample`).
