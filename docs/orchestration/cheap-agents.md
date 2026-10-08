# Cheap agents: the `cmd` CLI

[Command Code](https://commandcode.ai) (`cmd`, v1.74 installed globally via nvm, logged in) is a coding-agent CLI that serves open models (DeepSeek, Qwen, GLM, Kimi, MiniMax, …) for a fraction of Claude's price. Its models are **cheap and literal**: fast at well-specified mechanical work, weak at judgement, at keeping scope, and at noticing they're wrong. Brief them like a new contractor. Every `cmd` output is a maker artifact and needs a checker.

## Running a task

```bash
scripts/agents/cmd-task.sh [--tier fast|code|long | --model ID] [--max-turns N] \
                           [--cwd DIR] [--log-dir DIR] [--effort LEVEL] BRIEF.md   # or "-" for stdin
```

- Prints the agent's final text on stdout, with a status line (duration, tokens) and the NDJSON log path on stderr. The exit code is cmd's (`8` = hit `--max-turns`).
- Always headless with `--yolo`: the agent edits and runs shell commands without prompts. **Write tasks run with `--cwd` set to a disposable worktree** (`git worktree add ../vq-cmd-<slug> -b cmd/<slug>`), never the main checkout. The orchestrator reviews the diff and cherry-picks.
- Logs default to `data/swarm/cmd-logs/`. For a job, pass `--log-dir data/swarm/<job>/cmd-logs`.
- Fan-out: `ls prompts/*.md | xargs -P 6 -I{} scripts/agents/cmd-task.sh --tier fast {} > /dev/null` (outputs land in each brief's named output file, see the template).
- Raw CLI equivalent: `cmd -p "<brief>" -m <model> --yolo -t --skip-onboarding --no-session --max-turns N --output-format json`.

Each call carries ~17–20k input tokens of system-prompt overhead even for a one-word answer, so batch tiny items into one brief (e.g. 20 vault files per brief, not 1).

## Tiers

Verified with `cmd --list-models` and a smoke run on 2026-10-04. Re-run `cmd --list-models` when a model ID fails, since the catalog changes often.

| Tier | Default model | Alternates | Use for |
|---|---|---|---|
| `fast` | `deepseek/deepseek-v4.1-flash-fast` | `qwen/qwen3.8-flash`, `z-ai/glm-5.3-flash` | single-file mechanical edits, bulk classification/extraction to JSON, log summarisation |
| `code` | `deepseek/deepseek-v4.1-flash` at `--effort max` (script default for this tier; user pick 2026-10-08, replaced `zai-org/glm-5.3`) | `zai-org/glm-5.3`, `qwen/qwen3.8-max-0902`, `minimaxai/minimax-m3` | multi-file but fully specified changes, tests from a written spec, boilerplate |
| `pro` | `xiaomi/mimo-v2.6-pro` (fixed reasoning effort — passing `--effort` makes cmd fail; user 2026-10-08: "better than DeepSeek, ~Sonnet 5.5 level") | `xiaomi/mimo-v2.6-pro-ultraspeed` | harder multi-file / judgement-light coding a Claude implementer would otherwise do; fix rounds that need care |
| (trial) | `qwen/qwen3.8-max-0902` at `--effort xhigh` (supported: low, medium, xhigh — NOT max) | — | A/B vs MiMo 2026-10-08 (same brief): equivalent correct diff, 329s/702k in vs MiMo 315s/509k; small UI task 165s/232k, clean but kept the wrong (light-theme) Toaster variant → 1 fix round. Peer of MiMo; MiMo stays default (cheaper) |
| `long` | `moonshotai/kimi-k3` (1M ctx) | `z-ai/glm-5.3-flash` (1M), `xiaomi/mimo-v2.6-pro` | reading a lot to produce a short structured output |

## Routing

**Default implementer for well-specified work = `cmd --tier pro` (MiMo V2.6 Pro)**, not a Claude implementer: multi-file changes with named files, a written design, and deterministic tests (user, 2026-10-08). Track record (job 20261008-edge-hunt): UI selector + client regen, 3 strategy YAMLs, a funding event study — all clean first pass; one stall (37 min thinking, 0 edits) on an open-ended framework-design task → escalated to Sonnet. Watch for that failure mode: if a pro task shows no file edits after ~15 min (check the NDJSON log), stop it and escalate.

Claude implementers remain the default for: semantics-critical changes (fitness/metric math, fill/data correctness, look-ahead-sensitive code), open-ended design, NT engine lifecycle, and fix rounds on those.

Detached `cmd` runs don't notify the orchestrator — pair them with a background wait loop on their output files.


Send a task to `cmd` only when **all** of these hold:

1. The brief can name the exact files to touch or read and the exact output format.
2. Done is checkable by a command (test, parser, `jq` schema, `rg` count) or by a quick diff read.
3. A wrong answer is cheap: caught by the checker before it reaches `main`, the DB or a decision.

Good fits: Pine/FMZ spec triage to JSON, first-draft DSL YAML from a spec (the parser checks it), docstrings, renames across known files, fixture/table generation, test cases for a pure function given its spec, journal entry formatting from a handoff, summarising run logs.

Use Claude agents instead for: anything touching NT engine lifecycle, fill/metric semantics, the DB schema, concurrency, security/secrets, overfitting verdicts, risk, or any task where "figure out what's wrong" is the job.

**Escalation:** a `fast` task that fails its check twice moves to `code`. A `code` task that fails twice goes to a Claude `implementer` with the failure output in the brief. Don't retry the same brief a third time on the same tier.

## Writing the brief

Start from [`prompts/cmd-task.md`](prompts/cmd-task.md). Cheap models follow a brief literally, so:

- **Self-contained.** Paste the facts the model needs: file paths, the schema, 1–2 worked examples, the 2–3 gotchas that apply. Keep CLAUDE.md out of its reading list; at ~19KB it's mostly noise for a narrow task.
- **Positive scope.** "Edit only `a.py` and `tests/test_a.py`" rather than a list of forbidden things.
- **Exact verification command** the agent must run before it answers, plus the expected result.
- **Exact output.** Name the output file and give its format (JSON schema or a filled example). The final reply is one line: `DONE <path>` or `FAILED <reason>`.
- **Worktree Python.** If it runs repo Python in a worktree: `PYTHONPATH=$PWD /Users/verebelyin/projects/vibe-quant/.venv/bin/python`.
- No git commit/push and no secrets in its tasks. The orchestrator owns git.

## Checking cmd output

The `verifier` (or the orchestrator for tiny tasks) must:

1. `git -C <worktree> diff --stat`: only the files in scope changed.
2. Re-run the brief's verification command and check the result.
3. For bulk JSON: validate every file against the schema (`python3 -c` + `json.load`), and spot-check ~5% by hand against the source. Cheap models fabricate fields when a source is ambiguous.
