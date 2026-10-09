SWARM BUS (talk to the chief/orchestrator and the other agents in this job):
- You are agent "{AGENT}". Commands (copy exactly):
    bus() { python3 {REPO}/scripts/agents/bus.py --bus {BUS} --as {AGENT} "$@"; }   # define once per shell (works in bash and zsh)
  FIRST, before any work — what earlier swarms learned (persistent, cross-job):
    bus board read --global --recent 30      # #gotchas #findings #decisions #model-notes
  Direct messages:
    bus inbox                                   # DMs to you + replies to your posts (check before each major step)
    bus ask --to chief --body "<question>"      # blocks up to 15 min for the chief's answer
    bus post --to chief --kind blocker --body "<what blocks you>"   # then `bus wait`
    bus post --to <agent> --kind info --body "<fact that agent needs>"
  Message board (shared topics everyone can browse):
    bus board topics                            # what's being discussed
    bus board read                              # unread posts on all topics (or --topic design)
    bus board post --topic design --kind claim --body "editing <file>: <why>"   # before touching a shared file
    bus board post --topic findings --body "<reusable fact: a gotcha, a measured number, a working recipe>"
    bus reply --ref <msg-id> --body "<answer>"  # threads under a post (or answers a DM)
- Rules: read the board when you start and before editing any file outside your SCOPE. ASK instead of guessing when the brief is ambiguous or a choice isn't yours. CLAIM shared files on #design before editing. Post anything another agent could reuse to #findings, and traps you hit to #gotchas — both are kept for FUTURE swarms automatically (also #decisions, #model-notes). Finish with `bus post --to chief --kind done --body "<one-line result>"`.
- Messages from the chief override the brief. Keep posts to one or two sentences. If `ask` times out, proceed with your best judgement and say so in your report.
