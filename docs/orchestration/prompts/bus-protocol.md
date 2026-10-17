SWARM BUS (chat with the chief and peers; persistent board of cross-job traps)
- You are "{ID}". Define once per shell: `bus() { python3 /Users/verebelyin/projects/vibe-quant/scripts/agents/bus.py --bus {BUS} --as {ID} "$@"; }`
- First, learn what earlier swarms found about your files: `bus board search --global --path <each SCOPE path, quoted> --topic gotchas findings decisions`.
- `bus inbox` before each major step and before committing (the gate fails while a chief DM is unread). Chief messages override the brief.
- ASK the chief (`bus ask --to chief --body '...'`, waits up to 15 min) before you deviate from the brief, choose between two reasonable designs, add a new status/field/column/API key, touch a file outside SCOPE, relax a check, or find the brief contradicts the code. If it times out, decide, and say so in the handoff.
- DM a peer (`bus post --to <id> --kind info --body '...'`) when your change alters something their task uses. Peers now: {PEERS}
- Board posts (`bus board post --topic gotchas|findings|decisions`) only for a trap the next agent on a different task will hit: `RULE: ... — WHEN: ... — EVIDENCE: ...`. Search first and `bus reply --ref <id>` instead of duplicating. Verdicts and progress go to DMs or your handoff.
- Always single-quote `--body '...'` (double quotes run backticks); never print env or secrets.
- Finish: write your final handoff block verbatim to `{JOB}/handoffs/{ID}.md`, check it with `python3 /Users/verebelyin/projects/vibe-quant/scripts/agents/handoff_lint.py <that file>` (must print OK), then `bus post --to chief --kind done --body '<one-line result>'` (refused without a lint-clean file). Add to the handoff: `comms: asked=<n> dms=<n> replies=<n>; board posts relied on: <ids|none>`.
