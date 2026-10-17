# Dev servers & UI testing

Read before starting the backend/frontend, checking UI, or running the Playwright spec.

1. **Port 5173 may belong to another project.** vibe-quant's Vite can land on 5174/5175; screenshot or check the page title before testing (one session drove a Minecraft clone).
2. **Stop the backend with `pgrep -f "uvicorn vibe_quant" | xargs kill`.** `lsof -ti :8000 | xargs kill` also kills Vite, which holds proxy connections to :8000.
3. **The backend caches code at startup**: plugin registrations and endpoint changes show only after a restart. Discovery, validation and screening run as subprocesses and pick up new code from disk.
4. **Check ports before starting servers** (`lsof -i :8000`).
5. **Test a worktree's UI next to the user's servers** with `scripts/agents/ui-check.sh up|down <worktree>`; always run `down` afterwards.
6. **Run the whole Playwright suite with `scripts/agents/e2e.sh <worktree>`** — it boots the ui-check stack, runs the `chromium` project (every spec) and, when it started the stack, always tears it down. Filter with extra args (`... <worktree> e2e/settings.spec.ts` or `-g <scenario>`). It pins the API base to `127.0.0.1` (uvicorn is IPv4-only; `localhost` resolves to `::1` first) and caps workers (`E2E_WORKERS`, default 3) because the stack is a single uvicorn. If the ports are already busy (a live stack) it exits 3 without touching it.
7. **Wait for data before you screenshot or click.** Data pages render a skeleton first and then shift layout when tables load, which also makes agent-browser element refs stale; wait for a known loaded element, then OPEN every screenshot you cite (ver-e5-ingest-ui-1: one shot was only the loading skeleton).
