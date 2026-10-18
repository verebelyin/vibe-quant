# T3 children: DeepSeek makers and GPT reviewers

Command Code serves cheap open models; the only one allowed here is DeepSeek V4.1 Flash at effort high (see [`models.json`](models.json)). It is fast at well-specified work, and weak at judgement, at keeping scope and at noticing it is wrong. Brief it like a new contractor; every output needs a checker. GPT reviewers run through Codex on the same T3 mechanism.

## Dispatch protocol

1. **Target.** `scripts/agents/models.py route <role>` prints the `delegate_task` target (`providerInstanceId`, `model`, `options`). Call `orchestrator_capabilities` only when a dispatch errors or the route names a provider you have not used this session.
2. **Delegate.** `delegate_task` with `task` = the dispatch text `mkbrief.py` printed (a pointer to the brief file plus the TELEMETRY line), `target` from step 1, `mode: "async"`, and a `clientRequestId` unique per round (`cc-<slug>-<round>`, `rev-<slug>-<round>`; reuse it only to retry that same call). Record `taskId` in the ledger.
3. **Wait.** A finished child wakes your thread with a notification: end the turn instead of polling. Then read `handoffs/<id>.md` when the child wrote one (`task_status` prints the report twice). `task_cancel` only when the user asks.
4. **Next round.** A new `delegate_task` with the original brief, the findings verbatim and anything unresolved, under a new `clientRequestId`. Never `t3_thread_send` to a `childThreadId`, and never `t3_thread_launch`/`create_threads` for children.
5. **Errors.** Show the user the `delegate_task` error code and message word for word. No retry on another provider. A usage-limit failure: mark the model off with `models.py set <key> --off --until <reset time>` and re-dispatch on the next route entry.
6. **Done.** Check each child's work yourself (diff scope vs base, main checkout untouched, its verification command, the gate) before routing it to its checker.

## Workspace

Children have no cwd parameter: they start in the main checkout with full access. Every brief names the absolute worktree and requires `cd <worktree> && ` on each command (`cmd-task.md` does this). After the task, `git -C /Users/verebelyin/projects/vibe-quant status --short` must be unchanged. Command Code writes `.commandcode/taste/taste.md` into the main checkout itself; it is gitignored.

## Stalls

DeepSeek can fall into a read loop: re-reading the same files with no edits. A child with no worktree edits after ~15–20 min, or no notification after ~45 min: check its probe output (it may be working in `/tmp`), then tell the user and route the task to a Sonnet `implementer` once they agree. A DeepSeek task gets at most 2 rounds; then the `implementer` takes it with all findings and failure output.

## Briefing cheap models

- Paste the code it must change or call (line-numbered excerpts), the schema, 1–2 worked examples and the 2–3 gotchas that apply. Keep CLAUDE.md out of its reading list.
- Positive scope: "edit only `a.py` and `tests/test_a.py`".
- The exact verification command and expected result.
- Bulk tasks: batch many small items into one brief (each call carries ~17–20k tokens of overhead). Name the output file and its format; the final reply is `DONE <path>` or `FAILED <reason>`.
- Bulk JSON output: validate every file against the schema and spot-check ~5% by hand; cheap models invent fields when a source is ambiguous.

## Without T3

`scripts/agents/cmd-task.sh` (headless `cmd` CLI) is only for sessions with no T3 tools at all, and only when the user authorizes it. Launch each worker with one Bash call using `run_in_background: true`; judge liveness by the NDJSON log's mtime or bus posts, never by `ps`/`pgrep`.
