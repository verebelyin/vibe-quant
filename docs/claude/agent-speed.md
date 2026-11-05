# Agent speed: timings, mutants, E2E, test time

CLAUDE.md § Agent speed rules has the always-on rules (gate once per round, never run `pytest tests/unit` yourself, fewer fatter turns, short proofs); numbers here.

Measured (load 4–5): full `tests/unit` ≈ 98 s `-n auto` + 33 s timing set, ≈ 10 min serial · `swarm-check.sh` ≈ 20–30 s, ≈ 50 s on a hub module (`discovery/__main__.py`), ≈ 135 s `--full`, seconds cached (message-only amend stays) · `exactness_239.py` ≈ 6 s · Playwright ≈ 1.5 min · one turn ≈ 4–10 s.

- **Wall = CPU ÷ cores granted:** the `-n auto` pass burns ≈ 837 s CPU ⇒ 84 s floor on 10 cores; cut test CPU (`-v`, autouse fixtures, `-n 8` measured: no gain).
- **Mutants:** `scripts/agents/mutate.py --file F --old X --new Y -- <test>`; ≤ 5 per review, ≤ 3 per maker, changed logic only; a survivor outside money code is a follow-up bead. Stress loops ≤ 10 runs unless flakiness is the finding.
- **E2E:** iterate `-g <scenario>`, full spec once at the end; a 201-in-subprocess test polls the run's final status.
- **Concurrent gates share cores:** each pytest step registers `data/swarm/_gate-slots/<pid>` and takes `max(2, cpus ÷ gates running)` workers (`SWARM_CHECK_WORKERS` wins).
- **Slow tests:** a new/changed test over 2 s alone fails the gate unless `@pytest.mark.slow(reason=...)`.
- **Per-test time only serially:** `-n auto` inflates `--durations`.
- **Quote the load:** `scripts/agents/cpu_log.py summary --log data/swarm/<job>/cpu.jsonl --since HH:MM` (chief: `sample`).
