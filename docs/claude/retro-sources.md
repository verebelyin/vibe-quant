# Retro sources

Where agent history lives, for a retrospective. Cover **every** child agent: a chief thread's own activity hides most of the friction (retro 2026-11-05 first read 33 of 48 children).

1. **Last retro:** `git log -i --grep=retro --format='%h %ad %s' --date=short | head`; its bead lists what already landed.
2. **Top-level threads** in the window: T3 `t3_thread_list` (newest first), or SQL below.
3. **Every child** (Claude native, Codex, Command Code) of those threads, read-only:
   ```
   sqlite3 -readonly ~/.t3/userdata/statev2.sqlite "select s.thread_id, s.provider, s.status, s.started_at, s.completed_at, s.child_thread_id, t.title from orchestration_v2_projection_subagents s left join orchestration_v2_projection_threads t on t.thread_id=s.child_thread_id where s.started_at >= '<window start>' order by s.started_at"
   ```
   Read each child with `t3_thread_read` view `activity` (tool calls, failures, hook blocks), or SQL on `orchestration_v2_projection_turn_items`. The DB is ~10 GB: always filter by thread id.
4. **Raw transcripts:** Claude sessions `~/.claude/projects/-Users-verebelyin-projects-vibe-quant/*.jsonl` (subagents under `<session>/subagents/`); Codex `~/.codex/sessions` (commands wrapped as JS, see `bd recall gotcha:codex-rollout-command-parsing`). Grep them with `rg -c`, never cat.
5. **Job evidence:** `data/swarm/<job>/` (brief, ledger, handoffs, reviews), `data/swarm/_gate-logs/`, `docs/orchestration/telemetry/tasks.jsonl`.
6. **Steering load:** `tests/unit/test_steering_budget.py` budgets; session-start output of `scripts/agents/memory_index.py` and `bus.py board brief`.
