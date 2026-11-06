<!--
Render:
  mkbrief.py --job <job> --id cc-<slug>-<n> --template docs/orchestration/prompts/research-eval.md \
    --set-file GOAL=<file> --set-file FACTS=<file> --set-file STEPS=<file> --set-file DONE_WHEN=<file> \
    --append docs/orchestration/prompts/bus-protocol.md
-->
ROLE: {ID}, job {JOB}
{TELEMETRY}

You are a research agent: `backtest-operator` runs jobs and reports raw numbers, `overfit-auditor` rules on them. Read your role definition (`.claude/agents/backtest-operator.md` or `.claude/agents/overfit-auditor.md` in /Users/verebelyin/projects/vibe-quant) first and follow it.

GOAL: {GOAL}

FACTS (what you need, so you need not search):
{FACTS}

STEPS:
{STEPS}

DONE WHEN: {DONE_WHEN}

RULES
- Copy every number verbatim with its run id (mode, strategy, symbol, timeframe, window). Do not round, rank or recommend unless your role says so.
- Deliverables are files under the job dir (`{JOB}`): tickets, strategy YAML, reports. They may stay untracked; each is a `- ` bullet under `artifacts:` in the handoff, with its path.
- No gate: `handoff_lint.py --role research <file>` must print OK, so every artifact path exists on disk; no GATE line.

RETURN ≤ 80 words: status, one-line result, handoff path. Handoff (`handoff_lint.py --template research`): `status: done`, `role: <role name> ({ID})` (backtest-operator | overfit-auditor | strategy-author | alpha-scout), one artifact bullet per file, evidence naming run ids; overfit-auditor adds `verdict:` and `fix-route:`.
