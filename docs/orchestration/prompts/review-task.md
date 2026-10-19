<!--
Reviewer brief template (GPT via T3 or the Claude `reviewer`/`reviewer-lite`). Render with:
  mkbrief.py --job <job> --id rev-<slug>-<n> --template docs/orchestration/prompts/review-task.md \
    --worktree <maker wt> --review swarm/<task> --parent <main | swarm/<parent task>> \
    --set-file SPEC=<maker brief or contract> --set PRIOR=<prior findings + fix text verbatim, or "none (round 1)"> \
    --set PEERS=... --append docs/orchestration/prompts/bus-protocol.md
-->
ROLE: reviewer "{ID}", job {JOB}
{TELEMETRY}

Your role definition is /Users/verebelyin/projects/vibe-quant/.claude/agents/reviewer.md: read it first and follow it.

REVIEW: worktree `{WT}`, pinned range `{RANGE}`. Start with `{REVIEW_PACK}`. Read the pinned head in ranges (`git -C {WT} show {HEAD_SHA}:<path> | sed -n a,bp`, `rg -n -C5`), never whole files over 300 lines; gate LOGs only via `rg -n FAILED`. Never check out, reset or stash in the maker's worktree; run tests and mutants in your own scratch copy (`scripts/agents/mutate.py` builds one).

SPEC (what the maker was asked to do, and the contract rows it must meet):
{SPEC}

PRIOR ROUNDS: {PRIOR}
Round ≥ 2 reviews cover the prior findings plus the fix delta only.

DM each CONFIRMED blocking finding to the maker and the chief as you find it (`bus post --to <maker id>` and `--to chief`, `--kind info`).

RETURN ≤ 80 words: status, verdict, fix-route, handoff path. The file holds the full reviewer handoff; `handoff_lint.py --role reviewer <file>` must accept it.
