# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` to load workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

**Version:** bd 1.0.0+ — embedded Dolt backend, no daemon, no `bd sync`. Auto-commits locally on every mutation. Rules: use `bd` for ALL task tracking (no TodoWrite / markdown lists); use `bd remember` for persistent project knowledge.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

**Optional: memory durability.** `.beads/embeddeddolt/` is gitignored, so memories from `bd remember` are machine-local by default. If you want them backed up off-machine, run `bd dolt push` (writes to `refs/dolt/data` on the same remote). Not part of the mandatory flow — matches beads team's default.
