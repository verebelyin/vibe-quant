# Beads (bd) Reference

Extracted from CLAUDE.md (kept lean per CLAUDE.md best practices). Read this
when installing bd, debugging bd itself, or deciding where a fact belongs.
Day-to-day usage lives in CLAUDE.md § Issue Tracking.

## Installing `bd` (v1.0.0+)

**macOS (this machine):** Homebrew or direct binary download.

```bash
# Option A — Homebrew
brew install beads

# Option B — Direct binary (darwin_arm64 example)
VERSION=1.0.0
curl -sL -o /tmp/beads.tar.gz \
  "https://github.com/gastownhall/beads/releases/download/v${VERSION}/beads_${VERSION}_darwin_arm64.tar.gz"
tar xzf /tmp/beads.tar.gz -C /tmp
cp /tmp/beads_${VERSION}_darwin_arm64/bd ~/.local/bin/bd
chmod +x ~/.local/bin/bd
```

**Init in a new repo:**
```bash
bd init --non-interactive --role maintainer    # Fresh init
bd init --from-jsonl --non-interactive --role maintainer   # Import from existing .beads/issues.jsonl
chmod 700 .beads                                # bd warns if not 0700
```

## v1.0.0 architecture

- **Embedded Dolt** — no separate server, no daemon process. Each `bd` command opens the DB, runs, exits. Exclusive file lock means **one writer at a time**.
- **No `bd sync`, no `bd daemon`** — both removed. Beads auto-commits to Dolt on every mutation.
- **No SQLite fallback** — SQLite backend was deleted in v0.58. `--backend sqlite` only prints migration advice.
- **JSONL still exists** (`.beads/issues.jsonl`) as a plain-text export for git diffs, but is no longer the primary store.
- **`bd doctor` is not supported in embedded mode** — use `ls -la .beads/embeddeddolt/` and `bd info` instead.
- **Valid issue types:** `bug`, `feature`, `task`, `epic`, `chore`, `spike`, `story`, `milestone`, `merge-request`, `molecule`, `gate`, `agent`, `role`, `rig`, `convoy`, `event`. Use `feature` not `enhancement`.

## Memory key taxonomy (`bd remember`)

**Key naming:** `category:specific-item` with a colon separator. Categories in use:

- `architecture:` — system layout decisions (`architecture:backtest-tiers`, `architecture:data-archive`)
- `command:` — commonly-used dev commands (`command:backend-start`, `command:perf-profiling-playbook`)
- `design:` — contentious design decisions + their risks (`design:per-direction-sltp`)
- `discovery:` — discovery pipeline facts (`discovery:champions`, `discovery:fitness-formula`, `discovery:compiler-hash`)
- `gotcha:` — non-obvious pitfalls (`gotcha:nt-1230-upgrade`, `gotcha:dev-servers`, `gotcha:sqlite-state-db`)
- `indicator:` — indicator-specific facts (`indicator:rust-native-nt`)
- `ops:` — operational / rollback playbooks (`ops:bd-v1-rollback`)
- `pipeline:` — cross-component invariants (`pipeline:discovery-vs-validation`, `pipeline:determinism`)
- `policy:` — hard rules (`policy:licenses-secrets`, `policy:lint-zero`)
- `reference:` — pointers to canonical docs (`reference:spec`, `reference:research-diaries`)

**When to use `bd remember` vs alternatives:**

| Fact type | Put it in |
|---|---|
| Project-level facts, repo-specific gotchas, invariants | `bd remember` (shared across all agents on any machine) |
| Trackable work with status / priority / dependencies | `bd create` (issues, not memories) |
| User-level prefs ("I like concise commits", "use rg not grep") | Claude Code auto-memory (`~/.claude/projects/.../memory/`) |
| Architecture that's already documented in SPEC.md | Nothing — just reference SPEC.md |
| Conversation-local state ("we're halfway through refactor X") | Plans / tasks, not memory |

**Best practices (beads team + project conventions):**

1. **Lead with the fact, not the metadata.** Good: `"BacktestDataConfig data_cls must be the class object — a string disables bar-type narrowing"`. Bad: `"Remember that in the July session we found..."`.
2. **Update in place.** Passing `--key` to an existing key overwrites — don't create `discovery:champions-v2`.
3. **Keep memories evergreen.** If a fact is a dated snapshot, prefer a commit or a bead over a memory.
4. **Delete when stale.** `bd forget <key>` on facts that turn out wrong or get superseded.
5. **Don't duplicate CLAUDE.md / SPEC.md.** Memories are for facts that are more specific or would otherwise be lost.

**Durability:** memories live in gitignored `.beads/embeddeddolt/` (machine-local).
`bd dolt push` publishes them to `refs/dolt/data` on the GitHub remote — optional,
not part of mandatory session close.

## Performance notes

- **Cold start:** ~1–3s (embedded Dolt schema load); occasionally ~20s fully cold.
- **Warm queries:** 0.3–1.0s per command. ~5× slower than the old SQLite backend but sub-second.
- **Disk:** `.beads/embeddeddolt/` ~34 MB (Dolt stores full commit history).
- **Parallel agents:** exclusive writer lock; for concurrent writers switch to server mode via `bd init --server`.
