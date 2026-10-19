# Multi-agent orchestration

How one **chief** (a Claude Code session running the [`orchestrate`](../../.claude/skills/orchestrate/SKILL.md) skill) splits large work across specialist agents. This file holds the roster, brief format, handoff contract, bus and workspace. Elsewhere:

- [`models.json`](models.json) + `scripts/agents/models.py route <role>`: which model plays which role, and which models are off. The only routing source.
- [`cheap-agents.md`](cheap-agents.md): dispatching T3 `delegate_task` children (DeepSeek makers, GPT reviewers).
- [`brief-checklist.md`](brief-checklist.md): what to check before every dispatch.
- [`prompts/`](prompts/): brief templates (`cmd-task.md`, `review-task.md`), the bus block (`bus-protocol.md`), the commit format (`commit-message.md`).
- [`research-swarm.md`](research-swarm.md): the strategy-factory lane.
- Agent definitions: [`.claude/agents/*.md`](../../.claude/agents/); each body is the agent's system prompt.
- [`history.md`](history.md): archived changelogs. Nothing loads it.

## Principles

1. **Maker-checker.** No agent grades its own output; a different agent, ideally on a different model family, can block each artifact.
2. **One voice.** Only the chief talks to the user, commits to `main`, pushes, closes beads and starts paper trading.
3. **Bounded briefs.** One task per dispatch, with a checkable done-when. "List every caller of X with file:line" is a brief; "investigate X" is not.
4. **Evidence over claims.** A claim without the command and output that prove it is UNVERIFIED. The chief re-runs the decisive command, or has a checker do it, before telling the user.
5. **Cheapest capable tier.** DeepSeek writes the code for every task with clear acceptance criteria; a stronger model reviews; Opus designs and gives money verdicts.
6. **Gates block.** A failed gate sends work back, escalates it or drops it. Lowering a threshold is never the fix.
7. **Checks over prose.** A recurring mistake gets a script, test or hook; prose is for judgement calls only (orchestrate skill § Self-improvement).

## Roster

| Agent | Lane | Writes | Job | Checked by |
|---|---|---|---|---|
| chief | both | ledger, merges, beads | decompose, dispatch, integrate, talk to the user | user gates |
| `architect` | eng | design doc | spec → design → task plan with acceptance criteria and contract table | user gate, design review |
| maker (`cmd`, T3) | eng | code + tests in a worktree | default maker for every task with clear acceptance criteria, test-first | reviewer + verifier |
| `implementer` | eng | code + tests in a worktree | fixer on `fix-route: sonnet`; takes over after a stall or 2 failed rounds; tasks too risky for the cheap maker | fresh reviewer + verifier |
| `operator` | eng | nothing; runs scripts/CLIs on scratch or read-only data | runs existing scripts and CLIs, reports numbers verbatim; never edits code or commits; stops and reports when a script fails closed | chief |
| `reviewer` | eng | nothing | two verdicts (spec, quality) + `fix-route` | chief |
| `reviewer-lite` | eng | nothing | first pass on mechanical diffs; escalates semantics | chief |
| `verifier` | eng | nothing | quality gates, exactness proofs, UI check | chief |
| `alpha-scout` | research | hypothesis tickets | sources → falsifiable hypotheses | strategy-author, overfit-auditor |
| `strategy-author` | research | DSL YAML, plugins | hypothesis → parser-valid strategy | overfit-auditor |
| `backtest-operator` | research | DB rows via API/CLI | runs screening/discovery/validation, reports raw numbers | overfit-auditor |
| `overfit-auditor` | research | nothing | PASS/REJECT against the repo's gates | risk-officer |
| `risk-officer` | research | nothing | veto before any paper/live step | user |

## Profiles

| Profile | Use when | Design |
|---|---|---|
| **lean** (default) | ≤ ~500 changed lines, one package, no NT engine lifecycle, fill/metric semantics or DB schema change | the chief writes the plan from the bead acceptance criteria |
| **full** | anything bigger, cross-package, or touching the areas above | `architect` → design review → user gate |

Build and check are the same in both: maker → gate → reviewer → fix round → verifier.

## Dispatch

| Runtime | How |
|---|---|
| Claude subagent | Agent tool, `subagent_type: "<name>"`, `model` from `models.py route`. Writers use a worktree from `scripts/agents/worktree.sh` (not `isolation: "worktree"`, which branches only from `main` and lacks market data). |
| T3 child (DeepSeek maker, GPT reviewer) | `delegate_task`, protocol in [`cheap-agents.md`](cheap-agents.md). |

- `scripts/agents/worktree.sh <slug> [base]` creates `../vq-<slug>` on `swarm/<slug>`, links `data/catalog` + `data/archive`, and prints the `run as` line (the editable install points at the main checkout, so worktree code runs with `PYTHONPATH=$PWD`).
- Shared state is single-writer: the state DB, the backend on :8000 and the beads store each have one owner at a time. Only `backtest-operator` launches jobs; only the chief mutates beads.
- WIP: at most 5 parallel writers. Readers fan out wider.

## Brief format

Briefs are rendered by `scripts/agents/mkbrief.py` from [`prompts/cmd-task.md`](prompts/cmd-task.md) [`prompts/review-task.md`](prompts/review-task.md), or, for a fix round, [`prompts/fix-task.md`](prompts/fix-task.md) (never hand-write one). Every brief has: role and id, TELEMETRY line, worktree, GOAL, SCOPE (the only files to edit), FACTS (pasted excerpts), STEPS, DONE WHEN (command + expected result), the gate command, the bus block, and RETURN = the handoff contract. The child sees nothing of the chief's conversation.

## Handoff contract

The last message of every agent:

```
## Handoff
status: done | blocked | failed | needs-decision
role: <agent>
task: <one line>
artifacts:
  - <path | run id | bead id | commit sha | worktree + branch>
evidence:
  - `<command>` → <decisive output line, verbatim numbers>
claims:
  - <claim> — VERIFIED (<how>) | UNVERIFIED (<why not>)
open:
  - <follow-up, suspected bug, decision needed; one line each>
comms: asked=<n> dms=<n> replies=<n>; board posts relied on: <ids | none>
improve: <one change to a script, check or brief template that would have saved a round, with evidence> | none
```

Reviewers add `verdict: PASS | SHOULD-FIX | BLOCK` and `fix-route: deepseek | sonnet | none`. `scripts/agents/handoff_lint.py` checks the block. `blocked` names the missing input; `needs-decision` gives options and a recommendation. Numbers are copied verbatim, never rounded.

## Tiered review

| Diff | Checker |
|---|---|
| test-only or tooling diffs ≤ ~100 lines outside money paths | chief diff-read + gate, no separate reviewer (user) |
| test-only nits, docs/config, dependency pins, renames, small UI tweaks, fix deltas that only apply a reviewer's exact spec | `reviewer-lite` (escalates when a hunk touches semantics) |
| fitness/metric/fill/funding/indicator math, look-ahead-sensitive code, data writes, codegen, NT lifecycle, schema, concurrency or process control, risk/paper/live, new scripts or modules, > ~150 hand-written non-test lines | `reviewer` |
| architect design | `design-review` route, `reviewer` in design mode |

- **Fix route:** `deepseek` (1–2 mechanical should-fix) → a new maker round with the findings verbatim, back to the same reviewer. `sonnet` (any blocking finding, ≥ 3 should-fix, a 2nd failed round, or a misunderstood task) → `implementer` with the findings verbatim, then a fresh reviewer.
- **Paper or live diffs:** the `risk-officer` reviews in parallel with the reviewer.
- **Nits never cost a round:** surviving mutants, test gaps outside money code and fail-closed findings in research tooling become follow-up beads. Mutants are capped at 5 per review and 3 per maker.

## Swarm bus

`scripts/agents/bus.py` (shim: `scripts/agents/bus`) gives each job a mailbox and message board under `data/swarm/<job>/bus/`.

- **Direct messages:** `post --to <agent|chief|all>`, `inbox`, `pending`, `ask` (blocks for the answer), `reply --ref`, `status` (the chief's ≤ 20-line overview).
- **Board:** `board post --topic <t>`, `board read`, `board search [--global] [--topic ...] [--path ...] [--grep ...]`, `board render`. Topics `#gotchas`, `#findings`, `#decisions` hold cross-job rules (`RULE / WHEN / EVIDENCE`) and are mirrored to the committed `docs/orchestration/board/messages.jsonl`; human view [`board/BOARD.md`](board/BOARD.md). Verdicts and status go to DMs or handoffs.
- **Workers** get the bus block in every brief. The gate fails a hand-back while a chief DM is unread.
- **Chief:** one durable tail as a background Bash call (`tail --to chief --durable --kinds question,blocker --exit-on-first`; task completions already notify); it keeps its own cursor, so a restart replays what it missed. Only one durable tail per (`--as`, `--to`) may run.
- **User decisions:** `bus.py ask-user`; only the user closes them (`needs-user answer --user`).

## Swarm tools

| Tool | What |
|---|---|
| `swarm-check.sh <wt> --scope 'g,g' [--base REF] [--bus B --bus-agent A] [--no-cache]` | pre-review gate: scope must cover every path in `REF...HEAD`, clean tree, bus inbox, ruff, mypy, targeted + importer tests, pinned-text, exactness, frontend. `--base` takes any ref (no `--scope` SKIPs the scope check); `mkbrief.py` pins the ref it was given to a SHA at render, so the rendered gate diffs a frozen commit (board/telemetry mirror files are never offenders). Caches PASS per HEAD+base+scope; writes a LOG per run; fails closed. |
| `mkbrief.py` | renders and validates a brief, writes it, prints the dispatch text |
| `review_pack.py <wt> --range B..H \| --merge-base P` | the review surface: log, stat, capped diff without lockfiles/generated files |
| `mutate.py --file F --old X --new Y -- <test>` | one safe mutant check (KILLED/SURVIVED) |
| `handoff_lint.py [--role reviewer] <file>` | checks the handoff block |
| `models.py route <role> \| set \| show` | model routing from `models.json` |
| `telemetry.py ingest-runs --job J \| review \| report` | cross-job task ledger `docs/orchestration/telemetry/tasks.jsonl` |

## Workspace

`data/swarm/<YYYYMMDD-slug>/` (gitignored): `brief.md` (goal, constraints, contract tables, rulings; append-only), `ledger.md` (one row per task), `prompts/` (rendered briefs), `handoffs/` (verbatim), `bus/`, `hypotheses/` (research). After a context compaction, re-read `brief.md`, `ledger.md` and the latest handoffs and continue. Durable outcomes go to beads, `bd remember`, `docs/discovery-journal.md` and git.

## Lanes

**Engineering:** brief → design (lean: chief; full: architect + design review) → user gate → beads → build (one maker per task, in its own worktree, test-first) → check (gate, reviewer, verifier) → land (one verbose commit per task) → report.

**Research:** scout → author → screen/discover → audit → validate → audit → risk → journal; see [`research-swarm.md`](research-swarm.md).

## Failure handling

| Symptom | Response |
|---|---|
| Same error 3 times | stop; re-brief with the error and a different approach, or escalate the tier |
| Agent edits out of scope | discard those hunks; file a bead for anything worth keeping |
| Handoff missing evidence | treat every claim as UNVERIFIED; dispatch `verifier` |
| Two agents need the same file | serialize them, or have `architect` re-cut the boundary |
| Process dies with no traceback | NT Rust abort, not an agent fault (`docs/claude/nautilus-gotchas.md`) |
| Results look too good | `overfit-auditor` before anyone celebrates |
