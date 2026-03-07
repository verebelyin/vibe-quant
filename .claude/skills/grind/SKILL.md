---
name: grind
description: Autonomous bead grinding loop — picks open beads one by one, validates relevance, implements, tests, commits, and pushes. No human interaction needed. Use when user says /grind, "grind beads", "work through issues", "implement beads", or "clear the backlog". Runs until all ready beads are done or context runs low.
---

# Grind

Autonomous loop: pick bead, validate, implement, test, commit, push. Repeat.

## Rules

1. **No subagents** — do everything in the main session
2. **No human interaction** — make decisions autonomously
3. **Scope discipline** — if you discover tech debt, unrelated bugs, or scope creep while working, DO NOT fix it. Instead: `bd create --title="..." --type=bug --priority=3` and move on
4. **Follow codebase style** — match existing patterns, imports, naming. Read files before modifying.
5. **Verify before closing** — run tests/linters, confirm code works. Never skip.
6. **Commit after each bead** — atomic commits, one bead per commit

## Loop

```
1. bd ready                          # get available beads
2. Pick highest-priority bead
3. bd show <id>                      # read full details
4. ANALYZE: Is this bead still relevant?
   - Check if the code/feature it references still exists
   - Check if another bead already fixed it
   - If irrelevant → bd close <id> --reason="no longer relevant" → goto 1
5. bd update <id> --status in_progress
6. IMPLEMENT:
   - Read relevant source files first
   - Follow existing code style and patterns
   - Add/update unit tests if the change is testable
   - Keep changes minimal and focused
7. VERIFY:
   - Run: pytest <relevant_test_files> (or full suite if unsure)
   - Run: ruff check <changed_files>
   - If tests/lint fail → fix → re-verify
   - Do NOT skip this step
8. COMMIT & PUSH:
   - git add <specific_files>
   - git commit -m "<type>: <description> (bd-<short_id>)"
   - bd close <id>
   - bd sync
   - git push
9. goto 1
```

## Stopping Conditions

Stop the loop when:
- `bd ready` returns no beads
- Context window is getting full (>80% used) — push remaining work and summarize
- A bead requires external input or decisions beyond scope — skip it, note in summary

## Bead Selection

Priority order:
1. Bugs (highest priority first)
2. Tasks
3. Features
4. Chores

Skip beads that:
- Require UI/frontend changes if you're unsure about the design
- Need external API keys or services not available
- Are epics (too large for a single grind iteration)

## Quality Gates

Before closing any bead:
- Changed Python files pass `ruff check`
- Relevant tests pass (run specific test files, not always full suite)
- No import errors in changed modules
- If you added a function/class, it has basic test coverage

## Commit Message Format

```
<type>: <concise description> (bd-<short_id>)
```

Types: `fix`, `feat`, `refactor`, `chore`, `test`, `docs`

Example: `fix: handle None in sharpe formatting (bd-k4ya)`

## Summary

After the loop ends, output a markdown summary:

```markdown
## Grind Summary

### Completed
| Bead | Title | Type | Changes |
|------|-------|------|---------|
| bd-xxx | Fix None crash | bug | vibe_quant/foo.py |

### Skipped
| Bead | Title | Reason |
|------|-------|--------|
| bd-yyy | Redesign UI | needs design input |

### New Issues Filed
| Bead | Title | Found While |
|------|-------|-------------|
| bd-zzz | Tech debt in X | implementing bd-xxx |

### Stats
- Completed: N
- Skipped: N
- New issues: N
- Commits pushed: N
```
