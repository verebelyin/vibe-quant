<!--
Reviewer brief template (GPT via T3 or the Claude `reviewer`/`reviewer-lite`). Render with:
  mkbrief.py --job <job> --id rev-<slug>-<n> --template docs/orchestration/prompts/review-task.md \
    --worktree <maker wt> --review swarm/<task> --parent <main | swarm/<parent task>> \
    --set-file SPEC=<maker brief or contract> --set PRIOR=<prior findings + fix text verbatim, or "none"> \
    --set PEERS=... --append docs/orchestration/prompts/bus-protocol.md
-->
ROLE: reviewer "{ID}", job {JOB}
{TELEMETRY}

Read and follow your role definition /Users/verebelyin/projects/vibe-quant/.claude/agents/reviewer.md first.

REVIEW: worktree `{WT}`, pinned range `{RANGE}`. Start with `{REVIEW_PACK}`. Read the pinned head in ranges (`git -C {WT} show {HEAD_SHA}:<path> | sed -n a,bp`, `rg -n -C5`), never whole files over 300 lines; gate LOGs only via `rg -n FAILED`. Never check out, reset or stash in the maker's worktree; run tests, mutants and proofs in your own worktree: `scripts/agents/worktree.sh {ID} {HEAD_SHA}`; a plain copy breaks exactness.
- Crash probes: SIGKILL, never os.abort()/SIGABRT/SIGSEGV (macOS shows the user crash dialogs).

SPEC (the maker's task and the contract rows it must meet):
{SPEC}

PRIOR ROUNDS: {PRIOR}
Only review round: list ALL findings now; each blocking one gets a runnable repro.

DM each CONFIRMED blocking finding to the chief as you find it (`bus post --to chief --kind info`); the maker has finished.

RETURN ≤ 80 words: status, verdict, fix-route, handoff path. The file holds the full reviewer handoff; `handoff_lint.py --role reviewer <file>` must accept it.
