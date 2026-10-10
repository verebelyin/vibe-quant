# E2E tests (Playwright)

Specs in this directory drive the Vite app against a running backend.

## Run against the throwaway ui-check stack

`scripts/agents/ui-check.sh` boots backend `:8001` + Vite `:5188` on a **copy** of
the state DB (it never touches the real `data/state/vibe_quant.db` or `:8000/:5173`):

```bash
scripts/agents/ui-check.sh up <worktree>
cd <worktree>/frontend
E2E_BASE_URL=http://localhost:5188 \
  pnpm exec playwright test e2e/discovery-flow.spec.ts --project=chromium
scripts/agents/ui-check.sh down <worktree>   # always run this, pass or fail
```

- `E2E_BASE_URL` overrides `baseURL` and omits the `pnpm dev` `webServer`, so the
  specs reuse an already-running stack. Without it, the config keeps the default
  behaviour (`http://localhost:5173` + `pnpm dev` webServer).
- `E2E_API_URL` (default `http://localhost:8001`) is where specs read fixture ids
  (discovery runs with champions, completed validation runs) directly from the API.
- Node modules are symlinked from the main checkout by `ui-check.sh`; do **not**
  run `pnpm install`.

## Specs

- `discovery-flow.spec.ts` — discovery launch (Select All; UI + `/api/discovery/indicator-pool`
  names must equal the fixed GA pool list in the spec; tiny config; the run is then polled
  and must stay running-with-progress/completed, never `failed`) -> champion export ->
  validation launch of the exported strategy on BTCUSDT 1m / 6M (polled to `completed`,
  `error_message` is shown on failure) -> results render (headline metrics + equity chart).
  Takes ~1 min. Teeth: it fails if the pool names mismatch the GA (vibe-quant-rubp1) or the
  validation subprocess crashes after a 201 (vibe-quant-wrlea). Update `GA_INDICATOR_POOL`
  when an indicator is added to / removed from the GA pool.

`test-results/` and `playwright-report/` are gitignored. `e2e/discovery-flow.spec.ts` is the
only e2e file in biome's `files.includes`; the older specs are not linted.

The older specs date from 2026-02 and are not maintained.
