<!--
Reviewer brief template (GPT via T3 or the Claude `reviewer`/`reviewer-lite`). Render with:
  mkbrief.py --job <job> --id rev-<slug>-<n> --template docs/orchestration/prompts/review-task.md \
    --worktree <maker wt> --review swarm/<task> --parent <main | swarm/<parent task>> \
    --set-file SPEC=<maker brief or contract> --set PRIOR=<prior findings + fix text verbatim, or "none (round 1)"> \
    --set PEERS=... --append docs/orchestration/prompts/bus-protocol.md
-->
ROLE: reviewer "{ID}", job {JOB}
{TELEMETRY}

Your role definition is /Users/verebelyin/projects/vibe-quant/.claude/agents/reviewer.md: read it first (a Claude `reviewer` subagent already has it as its system prompt) and follow it.

REVIEW: worktree `{WT}`, pinned range `{RANGE}`. Start with `{REVIEW_PACK}`. Read files at the pinned head with `git -C {WT} show {HEAD_SHA}:<path>`. Never check out, reset or stash in the maker's worktree; run tests and mutants in your own scratch copy (`scripts/agents/mutate.py` builds one).

SPEC (what the maker was asked to do, and the contract rows it must meet):
{SPEC}

PRIOR ROUNDS: {PRIOR}
Round ≥ 2 reviews cover the prior findings plus the fix delta only.

DM each CONFIRMED blocking finding to the maker and the chief as you find it (`bus post --to <maker id>` and `--to chief`, `--kind info`).

RETURN the reviewer handoff (status, verdict, fix-route, evidence, open, improve); `scripts/agents/handoff_lint.py --role reviewer -` must accept it.
