<!--
cmd brief: first-draft DSL YAML from one triaged, dsl_feasible spec.
Tier: code (T3 delegate_task, commandcode_command, model deepseek/deepseek-v4.1-flash). T3 children have no cwd:
prepend the WORKSPACE block from prompts/cmd-task.md naming the disposable worktree (cd <worktree> && on every command).
Orchestrator fills: <SPEC PATH>, <TRIAGE JSON LINE>, <TEMPLATE PATHS>, <OUT YAML>, <INDICATORS>.
Checker: the parser command below (deterministic) + strategy-author reviews rule fidelity before any backtest.
-->
You are translating one trading-strategy description into our strategy YAML format. Create exactly one file: <OUT YAML>. Edit no other file.

READ FIRST:
- The strategy description: <SPEC PATH>
- Its triage summary: <TRIAGE JSON LINE>
- Two example strategies in our format (copy their structure exactly): <TEMPLATE PATHS, e.g. vibe_quant/strategies/templates/rsi_mean_reversion.yaml vibe_quant/strategies/templates/donchian_breakout.yaml>

FACTS YOU NEED:
- Allowed indicator `type` values (exact): <INDICATORS>
- Conditions are strings like "rsi < 30", "close > ema_filter", "fast crosses_above slow", using indicator keys you defined under `indicators:`. Use only operators that appear in the example files.
- Indicator values are from CLOSED bars only. A Pine `[1]` means the previous bar.
- Keep the source's default parameter values. If the source gives none, use the example files' values and list the field under the comment `# assumptions`.

FILE LAYOUT:
1. A comment block at the top, one line per rule in the description:
   `# <rule from the source> -> <condition you wrote>  [exact|approx: why|dropped: why]`
2. The YAML, structured like the examples, with `name: <OUT file name without .yaml>`.

CHECK (run it; it must print `OK`):
PYTHONPATH=$PWD /Users/verebelyin/projects/vibe-quant/.venv/bin/python -c "from vibe_quant.dsl.parser import parse_strategy; parse_strategy('<OUT YAML>'); print('OK')" 2>&1 | tail -3
If it prints an error, fix the YAML and re-run, at most 3 times.

Do not run git commands. Your final reply is exactly one line: `DONE <OUT YAML>` or `FAILED <one-line reason>`.
