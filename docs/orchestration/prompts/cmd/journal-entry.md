<!--
cmd brief: draft a discovery-journal entry from a job's handoffs. Tier: fast (T3 delegate_task, commandcode_command).
Orchestrator fills: <JOB DIR>, <OUT>. Orchestrator reviews the draft and pastes it into docs/discovery-journal.md itself.
Checker: orchestrator diff-reads every number against the handoffs (cheap models round and invent numbers).
-->
You are writing a research-journal entry in Markdown. Create exactly one file: <OUT>. Edit no other file.

READ: every file in <JOB DIR>/handoffs/ and <JOB DIR>/brief.md.
STYLE: entries are newest-first. Read the first two `## ` entries at the top of docs/discovery-journal.md and copy their heading levels, section names and table style.

The entry has, in this order:
1. Heading: `## <YYYY-MM-DD>: Batch <previous highest batch number + 1> — <short title> (runs <ids>)`.
2. Goal: one or two sentences from brief.md.
3. Setup: symbols, timeframes, windows, GA/sweep settings, as stated in the handoffs.
4. Results table: one row per run id with the metrics exactly as written in the handoffs.
5. Verdicts: overfit-auditor and risk-officer verdicts with their decisive gate, quoted.
6. Lessons: up to 3 bullets, only ones a handoff states or directly supports.

Rules:
- Copy every number character-for-character from the handoffs. Leave out any number that isn't in a handoff.
- If a handoff says UNVERIFIED, keep the word UNVERIFIED next to that claim.

Do not run git commands. Your final reply is exactly one line: `DONE <OUT>` or `FAILED <one-line reason>`.
