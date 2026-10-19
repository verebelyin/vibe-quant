# Beads (vibe-quant)

Track all work in `bd`, not TodoWrite or markdown lists. CLAUDE.md holds the project rules and wins over this text.

- Find: `bd ready`, `bd show <id>`; claim: `bd update <id> --claim`.
- Create: `bd create --title="..." --description="..." --type=task|bug|feature --priority=0-4` (`--parent=<epic>` for children).
- Notes: `bd update <id> --append-notes "..."` (a hook blocks `--notes`, which replaces them). `bd edit` opens an editor and blocks agents.
- Close: `bd close <id> [<id>...]`.
- Memories: session start lists their titles; read one with `bd recall <key>`, search with `bd memories <word>`.
- Finishing (CLAUDE.md § Session completion): file follow-ups, run the gates, close beads, commit, `git pull --rebase && git push`.
