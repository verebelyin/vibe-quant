# Commit message format (vibe-quant)

User rule (2026-10-11): commits are the project's history and a context source for future agents. One commit per task (bead or tightly coupled bead pair) — never one squash commit per job. The message is verbose and self-contained: a future agent reading only `git log` must understand why the change exists, how it works, which decisions were made and how it was proven.

```
<type>(<area>): <what changed, imperative, ≤ 72 chars> (<bead-id>[, <bead-id>])

Why:
<the problem or bead motivation, with evidence: run ids, numbers, the bug
symptom, the review/audit that found it>

What:
<behaviour change as a user/agent sees it; semantics breaks called out
explicitly (SEMANTICS BREAK: ...); new flags/fields/notes keys>

How:
<the design: key files/functions, the mechanism, why it is bit-identical
or what it changes; non-obvious traps handled>

Decisions:
- <choice> — <who decided: user / chief / reviewer> — <rejected alternative and why>

Verification:
- `<command>` → <decisive output, verbatim numbers>
- red before green: <failing test line before the fix>

Review:
- <rounds, maker/fixer models, reviewer findings fixed, mutants killed>
- Follow-ups: <bead ids filed from this task>

Job: <job id> · <worktree branch>
Co-Authored-By: Claude <model> <noreply@anthropic.com>   (chief adds it at Land; workers add none)
```

Rules:
- Body lines wrapped at ~72 chars; numbers copied verbatim from tool output.
- One commit per task on `main`, in dependency order; prompt/self-improvement and board/telemetry changes stay in their own commits.
- Worker commits on task branches use the same format (the chief rewrites/merges them into the final per-task commit from the brief, handoffs and reviews).
- Never add a `Co-Authored-By` line for DeepSeek, Codex/GPT or any other non-Claude model, and never in worker commits; maker/reviewer models go in the Review section as text. The chief's land script strips any such trailer (user 2026-10-16: GitHub showed DeepSeek as a commit author; history was rewritten and force-pushed).
