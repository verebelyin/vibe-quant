# Code Review: Project Process & Documentation

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** `.beads/` infrastructure, `README.md`, `CLAUDE.md`, `SPEC.md` -- project-level process, documentation accuracy, operational readiness

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `.beads/issues.jsonl` | ~305 | Issue tracker data (beads) |
| `.beads/README.md` | ~50 | Generic beads tool guide |
| `README.md` | 241 | Public-facing project documentation |
| `CLAUDE.md` | 132 | AI agent conventions and workflows |
| `SPEC.md` | ~2000+ | Authoritative implementation specification |
| **Total reviewed** | **~2400+** | |

---

## Findings

### MEDIUM (2)

#### M-1: Beads daemon startup unreliable in development environment
**File:** `.beads/` runtime behavior

`bd ready` and other CRUD operations intermittently fall back to direct mode with "Daemon took too long to start (>5s)" warnings. This adds 3-5 seconds latency per operation and risks workflow drift if developers bypass beads due to slowness.

`bd doctor` reports daemon running (PID 178, v0.49.3) with 4 warnings:
1. CLI version behind (0.49.3 vs 0.49.6)
2. Sync branch not configured
3. Claude plugin not installed
4. Uncommitted changes present

**Impact:** Slowed issue tracking operations. Developers may avoid creating/updating issues due to latency friction.

**Fix:**
1. Upgrade CLI: `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash`
2. Document team fallback workflow for direct mode (already in CLAUDE.md "Fallback" section)
3. Consider `bd migrate sync beads-sync` for multi-clone setups

#### M-2: Code review defects were not tracked as beads issues
**File:** `.beads/issues.jsonl`, project workflow

Prior code reviews (including `gpt-review.md`) identified P0/P1 defects that were not represented as beads issues. CLAUDE.md mandates beads as the single source of truth for all task tracking, yet critical defects existed only in markdown review files with no actionable tracking.

**Impact:** Defects discoverable only by reading review documents. No priority triage, no assignment, no status tracking.

**Fix:** Create beads issues for all findings from code reviews. *(Now resolved -- 67 beads created covering 195 findings across 19 review files.)*

**Status:** RESOLVED as of 2026-02-12.

### LOW (2)

#### L-1: Documentation authority split across 3 files with no clear hierarchy for newcomers
**Files:** `README.md`, `CLAUDE.md:1-10`, `SPEC.md`

- `SPEC.md` is authoritative for implementation (stated in CLAUDE.md line 30)
- `CLAUDE.md` is authoritative for conventions and workflows
- `README.md` is public-facing documentation
- `.beads/README.md` is a generic beads tool guide, not project-specific
- `docs/*.md` files are explicitly superseded (CLAUDE.md "Historical Documentation" section)

CLAUDE.md correctly establishes the precedence: `SPEC.md > CLAUDE.md > README.md > docs/*`. However, this hierarchy is only documented in CLAUDE.md, which new human contributors may not read first.

**Impact:** Low -- AI agents follow CLAUDE.md. Human contributors could make decisions based on outdated docs/ files.

**Fix:** Add a one-line note to README.md: "See SPEC.md for implementation details and CLAUDE.md for development conventions."

#### L-2: README.md marks all 8 phases complete but project has 12 CRITICAL blocking defects
**File:** `README.md:218-225`

All phases are checked `[x]` as complete:
```markdown
- [x] Phase 6: Paper Trading & Alerts
- [x] Phase 7: Ethereal DEX Integration
- [x] Phase 8: Automated Strategy Discovery
```

Current code state contradicts this:
- **Phase 6:** Paper module `_trading_node` is a placeholder dict (C-1). No actual trading occurs. No state persistence (C-2). No Telegram alerts (C-3).
- **Phase 7:** Ethereal has naive datetime (C-1) and nonce collision risk (C-2).
- **Phase 8:** Discovery has dual type system (C-1), STOCH param mismatch (C-2), invalid MACD ranges (C-3), and 100x SL/TP scale error (C-4).

Additionally, DB module has SQL injection (C-1) and FK violation (C-2), and validation is blocked by NT binary compatibility (C-1).

**Impact:** Misleads users and contributors about project maturity. Public README claims production readiness that doesn't exist.

**Fix:** Change completed phases to reflect actual operational state:
```markdown
- [x] Phase 1-4: Foundation, DSL, Validation, Overfitting (functional)
- [x] Phase 5: Dashboard (functional with import bugs)
- [ ] Phase 6: Paper Trading (scaffold only -- TradingNode not wired)
- [~] Phase 7: Ethereal DEX (functional with critical bugs)
- [~] Phase 8: Strategy Discovery (functional with type system bugs)
```

### INFO (2)

**I-1:** CLAUDE.md correctly establishes `SPEC.md` as authoritative, `docs/*` as historical only, and `bd` as mandatory issue tracker. This precedence model was used consistently throughout this review.

**I-2:** Beads infrastructure is healthy per `bd doctor`: 72 checks passed, 0 failures. 237 closed issues demonstrate active project history. Git hooks (pre-commit, post-merge, pre-push) are installed.

---

## Summary

| Severity | Count |
|----------|-------|
| MEDIUM | 2 |
| LOW | 2 |
| INFO | 2 |
| **Total** | **6** |

## Recommendations

**Priority 1:** Update README.md phase status to reflect actual operational readiness (L-2). This is the highest-visibility documentation error.

**Priority 2:** Upgrade beads CLI to 0.49.6 and document daemon fallback workflow (M-1).

**Priority 3:** Add documentation hierarchy note to README.md for human contributors (L-1).

**Resolved:** M-2 (defect tracking gap) addressed by creating 67 beads from code review findings.
