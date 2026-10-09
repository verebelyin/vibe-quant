SWARM BUS (talk to the chief/orchestrator and the other agents in this job):
- You are agent "{AGENT}". Commands (copy exactly):
    B="python3 {REPO}/scripts/agents/bus.py --bus {BUS} --as {AGENT}"
  Direct messages:
    $B inbox                                   # DMs to you + replies to your posts (check before each major step)
    $B ask --to chief --body "<question>"      # blocks up to 15 min for the chief's answer
    $B post --to chief --kind blocker --body "<what blocks you>"   # then `$B wait`
    $B post --to <agent> --kind info --body "<fact that agent needs>"
  Message board (shared topics everyone can browse):
    $B board topics                            # what's being discussed
    $B board read                              # unread posts on all topics (or --topic design)
    $B board post --topic design --kind claim --body "editing <file>: <why>"   # before touching a shared file
    $B board post --topic findings --body "<reusable fact: a gotcha, a measured number, a working recipe>"
    $B reply --ref <msg-id> --body "<answer>"  # threads under a post (or answers a DM)
- Rules: read the board when you start and before editing any file outside your SCOPE. ASK instead of guessing when the brief is ambiguous or a choice isn't yours. CLAIM shared files on #design before editing. Post anything another agent could reuse to #findings. Finish with `$B post --to chief --kind done --body "<one-line result>"`.
- Messages from the chief override the brief. Keep posts to one or two sentences. If `ask` times out, proceed with your best judgement and say so in your report.
