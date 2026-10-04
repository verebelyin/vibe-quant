<!--
Brief template for scripts/agents/cmd-task.sh. Copy to data/swarm/<job>/prompts/<slug>.md,
fill every <...>, delete these comments. Keep it self-contained: the cheap model sees only this.
-->
You are working in the directory you were started in: a disposable git worktree of a Python 3.13 project.

TASK: <one sentence: the exact change or extraction>

FILES YOU MAY EDIT OR CREATE: <exact paths>. Leave every other file unchanged.
FILES TO READ FIRST: <exact paths, the fewest that suffice>

FACTS YOU NEED:
- <pasted schema / signature / convention>
- <the 1–3 gotchas that apply, e.g. "run Python as: PYTHONPATH=$PWD /Users/verebelyin/projects/vibe-quant/.venv/bin/python">

EXAMPLE OF A CORRECT RESULT:
<a short worked example of the output or the edit>

STEPS:
1. <step>
2. <step>
3. Run this check: `<command>`. Expected: <result>. If it fails, fix and re-run, at most 3 times.

OUTPUT: <write results to <path> in <format>>.
Do not run git commit or git push.
Your final reply is exactly one line: `DONE <output path>` or `FAILED <one-line reason>`.
