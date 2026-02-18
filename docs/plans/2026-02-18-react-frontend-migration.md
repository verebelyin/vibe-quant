# React Frontend Migration Plan

> **Status:** Draft — pending approval
> **Date:** 2026-02-18
> **Scope:** Replace the Streamlit dashboard (~8,310 LOC) with a React SPA + FastAPI backend
> **Approach:** Phased migration with parallel operation — Streamlit continues to work until React reaches feature parity

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Target Architecture](#3-target-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Phase 0: FastAPI API Layer](#phase-0-fastapi-api-layer-foundation)
6. [Phase 1: React Scaffold & Core Infrastructure](#phase-1-react-scaffold--core-infrastructure)
7. [Phase 2: Data Management & Settings](#phase-2-data-management--settings-pages)
8. [Phase 3: Strategy Management](#phase-3-strategy-management)
9. [Phase 4: Backtest Launch & Results Analysis](#phase-4-backtest-launch--results-analysis)
10. [Phase 5: Discovery & Paper Trading (Real-Time)](#phase-5-discovery--paper-trading-real-time)
11. [Phase 6: Polish, Testing & Streamlit Retirement](#phase-6-polish-testing--streamlit-retirement)
12. [Risk Assessment](#risk-assessment)
13. [Decision Log](#decision-log)

---

## 1. Executive Summary

### Why migrate?

The Streamlit dashboard has served well for prototyping but has accumulated **8 documented workarounds** for fundamental framework limitations: keyboard event hijacking, form state leakage, orphaned subprocess tracking, session state ordering bugs, and 5-second polling as the only "real-time" mechanism. These are not bugs we can fix — they are architectural constraints of Streamlit's top-to-bottom rerun model.

### What do we gain?

1. **Real-time WebSocket data** for paper trading (instant position/PnL updates vs 5s polling)
2. **A reusable API layer** (enables CLI tools, mobile, third-party integrations)
3. **Component-level rendering** (only changed elements update, not entire pages)
4. **Professional UI quality** (shadcn/ui provides 200+ polished components)
5. **Elimination of all 8 documented workarounds** in the current codebase
6. **End-to-end type safety** (Pydantic → OpenAPI → generated TypeScript types)

### What does it cost?

- ~6-10 weeks of development across 7 phases
- Introduction of TypeScript as a second language
- Two build systems (uv + Vite) instead of one
- Every new feature touches two layers (API + UI) instead of one

### Strategy

**Phase 0 (API layer) is valuable regardless of whether we complete the full migration.** It decouples the backend from any frontend and enables automation, testing, and future integrations. We can stop after any phase and still have a working system.

---

## 2. Current State Analysis

### Dashboard Inventory

| Page | LOC | Complexity | Key Features |
|------|-----|------------|-------------|
| Strategy Management | 638 + ~1,150 components | High | Visual/YAML/Split editor, condition builder, indicator catalog, templates, sweep config |
| Discovery | 573 | High | GA config, real-time fitness chart, top strategies table, export to strategy |
| Backtest Launch | 286 | Medium | Strategy selector, sweep config, overfitting toggles, job monitoring |
| Results Analysis | 740 | Very High | 15+ chart types, comparison view, 3D Pareto, trade log, CSV export |
| Paper Trading | 713 | High | Real-time status, positions table, OS signal controls, checkpoint timeline |
| Data Management | 760 | High | Ingest form, streaming progress, TradingView candlestick, data quality |
| Settings | ~200 | Low | Sizing/risk config CRUD, latency presets |
| App shell + utils | 192 + charts 529 + data_builders 318 | Medium | Navigation, keyboard hack, chart builders, data transforms |

**Total: ~8,310 LOC** across 8 pages, 15 component files, shared utilities, and chart builders.

### Backend API Surface (existing, no HTTP layer)

| Module | Methods | Purpose |
|--------|---------|---------|
| `StateManager` | 30+ methods | CRUD for strategies, configs, runs, results, trades, sweeps, jobs |
| `BacktestJobManager` | 10+ methods | Subprocess spawn, kill, heartbeat, status sync |
| `CatalogManager` | ~5 methods | Parquet data catalog queries |
| `RawDataArchive` | ~5 methods | SQLite kline archive queries |
| `DSL modules` | ~10 functions | Validation, indicator metadata, template loading |
| `Discovery` | ~5 functions | Genome-to-DSL conversion, fitness evaluation |
| `Paper Trading` | ~8 functions | Persistence, state recovery, config management |

### Documented Pain Points (eliminated by migration)

| # | Pain Point | File:Line | Root Cause |
|---|-----------|-----------|------------|
| 1 | Arrow key capture by `st.navigation()` | app.py:79-94 | Streamlit intercepts DOM events globally |
| 2 | Form state leakage on mode switch | strategy_management.py:214-240 | Session state persists across reruns |
| 3 | Page rendering race conditions | app.py:98-104 | `st.navigation()` + `st.rerun()` conflict |
| 4 | Orphaned subprocess on navigation | data_management.py:60-72 | No component unmount lifecycle |
| 5 | Stale job status in DB | job_status.py:29-31 | Polling can miss process death |
| 6 | Session state ordering dependency | strategy_management.py:204-206 | Widget execution order matters |
| 7 | Subprocess output buffering | data_management.py:380-398 | Python stdout default buffering |
| 8 | Optional dependency fallback | data_management.py:28-31 | No module system for optional features |

---

## 3. Target Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         REACT SPA (Vite)                             │
│                                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Strategy │ │Discovery │ │ Backtest │ │ Results  │ │  Paper   │  │
│  │ Mgmt     │ │          │ │ Launch   │ │ Analysis │ │ Trading  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       │             │            │             │            │         │
│  ┌────┴─────────────┴────────────┴─────────────┴────────────┴────┐   │
│  │                     Shared Infrastructure                      │   │
│  │  TanStack Query (cache) │ Zustand (client state) │ Router     │   │
│  │  WebSocket manager      │ Type-safe API client    │ Auth       │   │
│  └───────────────────────────┬───────────────────────────────────┘   │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
                    REST (HTTP) │ WebSocket │ SSE
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                       FastAPI Backend                                 │
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │  REST Routers  │  │  WS Handlers   │  │  SSE Endpoints         │  │
│  │                │  │                │  │                        │  │
│  │ /api/strategies│  │ /ws/jobs       │  │ /api/backtest/progress │  │
│  │ /api/backtest  │  │ /ws/trading    │  │ /api/data/progress     │  │
│  │ /api/results   │  │ /ws/discovery  │  │ /api/logs/stream       │  │
│  │ /api/discovery │  │                │  │                        │  │
│  │ /api/paper     │  │                │  │                        │  │
│  │ /api/data      │  │                │  │                        │  │
│  │ /api/settings  │  │                │  │                        │  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────────────┘  │
│          │                   │                    │                    │
│  ┌───────┴───────────────────┴────────────────────┴────────────────┐  │
│  │                    Service Layer                                 │  │
│  │  StateManager │ JobManager │ CatalogManager │ DiscoveryPipeline │  │
│  │  RawDataArchive │ PaperTrading │ DSL Validator                  │  │
│  └─────────────────────────┬───────────────────────────────────────┘  │
│                            │                                          │
│  ┌─────────────────────────┴───────────────────────────────────────┐  │
│  │  SQLite (WAL) │ ParquetDataCatalog │ Subprocesses (NT engine)  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

### Communication Protocols

| Protocol | Use Case | Direction |
|----------|----------|-----------|
| **REST** (TanStack Query) | CRUD operations, data queries, job launch | Request/Response |
| **WebSocket** (Zustand store) | Job status, paper trading positions, discovery progress | Bidirectional push |
| **SSE** (EventSource) | Backtest progress bars, data ingest log streaming | Server → Client |

---

## 4. Technology Stack

### Frontend

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Build/Dev | Vite | 6.x | Fastest DX; no SSR needed for internal dashboard |
| Framework | React | 19.x | Component model; largest ecosystem |
| Language | TypeScript | 5.x | Type safety; auto-generated from Pydantic |
| Styling | Tailwind CSS | 4.x | Utility-first; pairs with shadcn/ui |
| Components | shadcn/ui | latest | 200+ accessible components; copy-paste ownership |
| Routing | React Router | 7.x | File-based routing; nested layouts |
| Price Charts | TradingView Lightweight Charts | 4.x | Canvas-based; purpose-built for financial data |
| Charts | Recharts | 2.x | Simple JSX API for equity curves, histograms, heatmaps |
| Tables | TanStack Table (via shadcn) | 8.x | Headless; free; integrates with Tailwind |
| Server State | TanStack Query | 5.x | Caching, dedup, background refetch |
| Client State | Zustand | 5.x | WebSocket store; UI preferences; lightweight |
| WebSocket | react-use-websocket | 4.x | Connection lifecycle, reconnection, buffering |
| Forms | React Hook Form + Zod | latest | Performant forms; schema validation |
| Type Generation | openapi-typescript | latest | Auto-generate TS types from FastAPI OpenAPI spec |

### Backend (new layer)

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Framework | FastAPI | 0.115+ | Async; native WebSocket; auto OpenAPI; Pydantic |
| Server | uvicorn | 0.34+ | ASGI server with --workers for multiprocess |
| Validation | Pydantic v2 | 2.x | Already used by FastAPI; generates OpenAPI schema |
| WebSocket | Starlette WebSocket | (via FastAPI) | Built-in; clean async pattern |
| SSE | sse-starlette | 2.x | StreamingResponse for progress updates |
| Testing | httpx + pytest | latest | Async test client for FastAPI |

### Development Tools

| Tool | Purpose |
|------|---------|
| `uv` | Python package management (existing) |
| `pnpm` | Node.js package management (faster than npm) |
| `orval` or `openapi-typescript` | Auto-generate TypeScript API client from OpenAPI |
| `vitest` | Unit testing for React components |
| `Playwright` | E2E testing (replaces agent-browser for automated tests) |
| `Biome` | Linting + formatting for TypeScript (replaces ESLint + Prettier) |

---

## Phase 0: FastAPI API Layer (Foundation)

> **Goal:** Create a REST + WebSocket API that wraps all existing backend modules. Both Streamlit and React can consume it.
> **Estimated effort:** 2-3 weeks
> **Deliverable:** Fully documented OpenAPI spec at `/docs`; all existing dashboard functionality accessible via HTTP

### Directory Structure

```
vibe_quant/
├── api/                          # NEW - FastAPI application
│   ├── __init__.py
│   ├── app.py                    # FastAPI app factory, lifespan, CORS
│   ├── deps.py                   # Dependency injection (StateManager, JobManager, etc.)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── strategies.py         # /api/strategies/*
│   │   ├── backtest.py           # /api/backtest/*
│   │   ├── results.py            # /api/results/*
│   │   ├── discovery.py          # /api/discovery/*
│   │   ├── paper_trading.py      # /api/paper/*
│   │   ├── data.py               # /api/data/*
│   │   └── settings.py           # /api/settings/*
│   ├── schemas/                  # Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   ├── backtest.py
│   │   ├── result.py
│   │   ├── discovery.py
│   │   ├── paper_trading.py
│   │   ├── data.py
│   │   └── settings.py
│   ├── ws/                       # WebSocket handlers
│   │   ├── __init__.py
│   │   ├── jobs.py               # /ws/jobs (job status push)
│   │   ├── trading.py            # /ws/trading (live positions/PnL)
│   │   └── discovery.py          # /ws/discovery (generation progress)
│   └── sse/                      # Server-Sent Events
│       ├── __init__.py
│       └── progress.py           # Backtest/ingest progress streaming
```

### User Stories

#### US-0.1: Strategy CRUD API
> **As a** frontend application,
> **I want to** create, read, update, delete, and list strategies via REST endpoints,
> **so that** the UI is decoupled from direct database access.

**Acceptance Criteria:**
- `GET /api/strategies` — list all strategies (filter: `?active_only=true`)
- `GET /api/strategies/{id}` — get single strategy with full DSL
- `POST /api/strategies` — create strategy (validates DSL schema)
- `PUT /api/strategies/{id}` — update strategy fields and/or DSL
- `DELETE /api/strategies/{id}` — soft-delete (set inactive)
- `GET /api/strategies/templates` — list available strategy templates
- `POST /api/strategies/{id}/validate` — validate DSL without saving
- All endpoints return Pydantic-validated JSON with proper HTTP status codes
- OpenAPI schema generated automatically

#### US-0.2: Backtest Launch & Job Management API
> **As a** frontend application,
> **I want to** launch screening/validation backtests and monitor job status via API,
> **so that** job management is accessible from any client.

**Acceptance Criteria:**
- `POST /api/backtest/screening` — launch screening sweep (returns `run_id`)
- `POST /api/backtest/validation` — launch validation backtest (returns `run_id`)
- `GET /api/backtest/jobs` — list active jobs with status
- `GET /api/backtest/jobs/{run_id}` — get single job status + heartbeat info
- `DELETE /api/backtest/jobs/{run_id}` — kill a running job (SIGTERM)
- `POST /api/backtest/jobs/{run_id}/sync` — force sync stale job status
- Request bodies validated with Pydantic (strategy_id, symbols, timeframe, date_range, sweep_params, overfitting_filters, sizing_config, risk_config, latency_preset)

#### US-0.3: Results & Analytics API
> **As a** frontend application,
> **I want to** query backtest results, trades, sweep results, and Pareto fronts via API,
> **so that** the results page can render charts from structured JSON.

**Acceptance Criteria:**
- `GET /api/results/runs` — list all backtest runs (with optional filters: status, strategy_id, date range)
- `GET /api/results/runs/{run_id}` — get run summary + metrics
- `GET /api/results/runs/{run_id}/trades` — get trade log (filter: symbol, direction)
- `GET /api/results/runs/{run_id}/sweeps` — get sweep results (filter: pareto_only)
- `GET /api/results/runs/{run_id}/equity-curve` — computed equity curve data points
- `GET /api/results/runs/{run_id}/drawdown` — drawdown series + top-N periods
- `GET /api/results/runs/{run_id}/monthly-returns` — monthly returns matrix
- `PUT /api/results/runs/{run_id}/notes` — update notes on a run
- `GET /api/results/runs/{run_id}/export/csv` — export trades/sweeps as CSV
- `GET /api/results/compare?run_ids=1,2,3` — comparison data for up to 3 runs

#### US-0.4: Discovery API
> **As a** frontend application,
> **I want to** launch genetic algorithm discovery runs and monitor progress via API,
> **so that** discovery can be started/stopped from any client.

**Acceptance Criteria:**
- `POST /api/discovery/launch` — start discovery run (config: population, generations, mutation rate, symbols, timeframes, indicator pool)
- `GET /api/discovery/jobs` — list active discovery jobs
- `DELETE /api/discovery/jobs/{run_id}` — kill discovery job
- `GET /api/discovery/results/{run_id}` — get top strategies discovered
- `POST /api/discovery/results/{run_id}/export/{strategy_index}` — export discovered strategy to strategy library
- `GET /api/discovery/indicator-pool` — get available indicators with parameter ranges

#### US-0.5: Paper Trading API
> **As a** frontend application,
> **I want to** start/stop paper trading sessions and get real-time position data via API,
> **so that** paper trading monitoring is not limited to Streamlit session state.

**Acceptance Criteria:**
- `POST /api/paper/start` — start paper trading session (strategy_id, Binance API creds via headers, testnet toggle)
- `POST /api/paper/halt` — halt trading (SIGUSR1)
- `POST /api/paper/resume` — resume trading (SIGUSR2)
- `POST /api/paper/stop` — graceful shutdown (SIGTERM, close positions)
- `GET /api/paper/status` — current state (running/halted/stopped/error), PnL metrics, trades count
- `GET /api/paper/positions` — open positions
- `GET /api/paper/orders` — pending orders
- `GET /api/paper/checkpoints` — recent checkpoint timeline

#### US-0.6: Data Management API
> **As a** frontend application,
> **I want to** trigger data downloads, check catalog status, and browse data via API,
> **so that** data management is not coupled to Streamlit subprocess handling.

**Acceptance Criteria:**
- `GET /api/data/status` — storage metrics (archive size, catalog size, total)
- `GET /api/data/coverage` — data coverage table (symbols, date ranges, kline/bar counts)
- `POST /api/data/ingest` — start data ingest (symbols, date range)
- `POST /api/data/update` — update existing data
- `POST /api/data/rebuild` — rebuild catalog from archive
- `GET /api/data/browse/{symbol}?interval={interval}&start={date}&end={date}` — OHLCV data for browser
- `GET /api/data/quality/{symbol}` — data quality verification results
- `GET /api/data/symbols` — supported symbols list
- `GET /api/data/history` — download history

#### US-0.7: Settings API
> **As a** frontend application,
> **I want to** manage sizing configs, risk configs, and latency presets via API,
> **so that** configuration is not embedded in the UI layer.

**Acceptance Criteria:**
- `GET/POST/PUT/DELETE /api/settings/sizing` — sizing config CRUD
- `GET/POST/PUT/DELETE /api/settings/risk` — risk config CRUD
- `GET /api/settings/latency-presets` — list available latency presets
- `GET /api/settings/system-info` — system diagnostics (table row counts, versions)

#### US-0.8: WebSocket Channels
> **As a** frontend application,
> **I want to** receive real-time push updates for jobs, paper trading, and discovery,
> **so that** I don't need to poll endpoints every 5 seconds.

**Acceptance Criteria:**
- `WS /ws/jobs` — pushes job status changes (started, progress, completed, failed, killed)
- `WS /ws/trading` — pushes position updates, PnL changes, order fills, state transitions
- `WS /ws/discovery` — pushes generation progress (best/mean/worst fitness per generation)
- All channels support reconnection with last-event-id for catch-up
- Heartbeat ping/pong every 30s to detect stale connections

#### US-0.9: SSE Progress Streams
> **As a** frontend application,
> **I want to** stream backtest and data ingest progress as Server-Sent Events,
> **so that** long-running operations show real-time log output.

**Acceptance Criteria:**
- `GET /api/backtest/jobs/{run_id}/progress` — SSE stream of subprocess log lines
- `GET /api/data/ingest/{job_id}/progress` — SSE stream of download progress
- Streams auto-close when the underlying process completes
- Client can reconnect and resume from last event ID

#### US-0.10: OpenAPI Documentation & Type Export
> **As a** frontend developer,
> **I want to** generate TypeScript types from the FastAPI OpenAPI spec,
> **so that** the React frontend has compile-time type safety for all API calls.

**Acceptance Criteria:**
- `GET /api/openapi.json` — full OpenAPI 3.1 spec
- `GET /docs` — Swagger UI for interactive testing
- CI script: `pnpm run generate-api-types` → generates `frontend/src/api/types.ts`
- All request/response bodies have Pydantic models (no `dict[str, Any]` in API boundaries)

### Phase 0 Tests

- Unit tests for each router (httpx AsyncClient + pytest)
- Integration test: launch screening via API → poll status → verify completed
- WebSocket test: connect, receive job status push, disconnect gracefully
- OpenAPI spec snapshot test (detect breaking changes)

---

## Phase 1: React Scaffold & Core Infrastructure

> **Goal:** Set up the React project, core layout, authentication, API client, and shared components.
> **Estimated effort:** 1 week
> **Prerequisite:** Phase 0 complete
> **Deliverable:** Working React app with navigation shell, theme toggle, and type-safe API client

### Directory Structure

```
frontend/                         # NEW - React application
├── index.html
├── package.json
├── pnpm-lock.yaml
├── tsconfig.json
├── vite.config.ts
├── biome.json                    # Linting + formatting config
├── tailwind.config.ts
├── components.json               # shadcn/ui config
├── src/
│   ├── main.tsx                  # App entry point
│   ├── App.tsx                   # Root component + providers
│   ├── api/
│   │   ├── client.ts             # Base fetch/axios config
│   │   ├── types.ts              # Auto-generated from OpenAPI
│   │   ├── queries/              # TanStack Query hooks per domain
│   │   │   ├── strategies.ts
│   │   │   ├── backtest.ts
│   │   │   ├── results.ts
│   │   │   ├── discovery.ts
│   │   │   ├── paper-trading.ts
│   │   │   ├── data.ts
│   │   │   └── settings.ts
│   │   └── websocket/
│   │       ├── useJobsSocket.ts
│   │       ├── useTradingSocket.ts
│   │       └── useDiscoverySocket.ts
│   ├── stores/                   # Zustand stores
│   │   ├── ui.ts                 # Sidebar, theme, layout prefs
│   │   ├── trading.ts            # Live positions, PnL (from WS)
│   │   └── jobs.ts               # Active jobs (from WS)
│   ├── components/
│   │   ├── ui/                   # shadcn/ui components (auto-added)
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   └── PageLayout.tsx
│   │   ├── charts/               # Reusable chart components
│   │   │   ├── EquityCurve.tsx
│   │   │   ├── DrawdownChart.tsx
│   │   │   ├── CandlestickChart.tsx
│   │   │   ├── MonthlyReturnsHeatmap.tsx
│   │   │   ├── TradeDistribution.tsx
│   │   │   ├── ParetoScatter.tsx
│   │   │   ├── RadarProfile.tsx
│   │   │   ├── RollingMetric.tsx
│   │   │   └── FitnessEvolution.tsx
│   │   └── shared/               # Shared domain components
│   │       ├── JobStatusBadge.tsx
│   │       ├── MetricCard.tsx
│   │       ├── StrategyCard.tsx
│   │       └── DateRangePicker.tsx
│   ├── pages/
│   │   ├── StrategyManagement.tsx
│   │   ├── Discovery.tsx
│   │   ├── BacktestLaunch.tsx
│   │   ├── ResultsAnalysis.tsx
│   │   ├── PaperTrading.tsx
│   │   ├── DataManagement.tsx
│   │   └── Settings.tsx
│   └── lib/
│       ├── utils.ts              # Shared utilities (cn, formatters)
│       └── constants.ts          # API URLs, default values
├── public/
│   └── favicon.svg
└── tests/
    ├── setup.ts
    ├── components/               # Vitest component tests
    └── e2e/                      # Playwright E2E tests
```

### User Stories

#### US-1.1: React Project Scaffold
> **As a** developer,
> **I want to** scaffold the React project with Vite, TypeScript, Tailwind, and shadcn/ui,
> **so that** I have a working development environment.

**Acceptance Criteria:**
- `pnpm create vite frontend --template react-ts`
- Tailwind CSS 4 configured
- shadcn/ui initialized with "New York" style, dark mode default
- Biome configured for linting + formatting
- `pnpm dev` starts dev server at `localhost:3000`
- Proxy to FastAPI backend at `localhost:8000` configured in `vite.config.ts`

#### US-1.2: App Shell & Navigation
> **As a** user,
> **I want to** see a sidebar navigation with all pages organized into categories,
> **so that** I can navigate the dashboard like the current Streamlit app.

**Acceptance Criteria:**
- Collapsible sidebar with category groups: Strategies, Backtesting, Trading, System
- Pages: Strategy Management, Discovery, Backtest Launch, Results Analysis, Paper Trading, Data Management, Settings
- Active page highlighted in sidebar
- URL-based routing (e.g., `/strategies`, `/discovery`, `/backtest`, `/results`, `/paper-trading`, `/data`, `/settings`)
- Dark/light theme toggle in header
- Responsive: sidebar collapses to hamburger on narrow screens

#### US-1.3: Type-Safe API Client
> **As a** developer,
> **I want to** auto-generate TypeScript types from the FastAPI OpenAPI spec,
> **so that** every API call is type-checked at compile time.

**Acceptance Criteria:**
- `pnpm run generate-api` fetches `http://localhost:8000/api/openapi.json` and generates `src/api/types.ts`
- Base API client with:
  - Automatic JSON serialization/deserialization
  - Error handling (toast notifications for 4xx/5xx)
  - Request cancellation on component unmount
- TanStack Query wrapper for each endpoint domain (e.g., `useStrategies()`, `useBacktestRun(id)`)

#### US-1.4: WebSocket Infrastructure
> **As a** developer,
> **I want to** establish WebSocket connections that update Zustand stores,
> **so that** real-time data flows to components without manual polling.

**Acceptance Criteria:**
- `useJobsSocket()` hook: connects to `/ws/jobs`, updates `jobsStore` on message
- `useTradingSocket()` hook: connects to `/ws/trading`, updates `tradingStore` on message
- `useDiscoverySocket()` hook: connects to `/ws/discovery`, updates discovery state on message
- Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, max 30s)
- Connection status indicator in header (green/yellow/red dot)
- Heartbeat ping every 30s

#### US-1.5: Shared Chart Components
> **As a** developer,
> **I want to** create reusable chart components that accept typed data props,
> **so that** charts are consistent across all pages.

**Acceptance Criteria:**
- `<CandlestickChart data={bars} />` — TradingView Lightweight Charts wrapper
- `<EquityCurve data={points} />` — Recharts line chart with proper formatting
- `<DrawdownChart data={points} periods={topN} />` — area chart with period overlays
- `<MonthlyReturnsHeatmap data={matrix} />` — red-white-green diverging heatmap
- `<TradeDistribution trades={trades} />` — histogram with profit/loss split
- `<ParetoScatter points={sweeps} />` — Sharpe vs MaxDD scatter
- `<RadarProfile metrics={normalized} />` — radar/spider chart
- All charts support dark/light theme via CSS variables
- All charts are responsive (fill parent container)

---

## Phase 2: Data Management & Settings Pages

> **Goal:** Implement the simplest pages first to validate the stack end-to-end.
> **Estimated effort:** 1 week
> **Prerequisite:** Phase 1 complete
> **Deliverable:** Fully functional Data Management and Settings pages in React

### User Stories

#### US-2.1: Data Management — Status Dashboard
> **As a** user,
> **I want to** see storage metrics and data coverage in the React app,
> **so that** I can understand what data is available.

**Acceptance Criteria:**
- Three metric cards: Archive size, Catalog size, Total size
- Data coverage table: symbol, date range, kline count, bar count per interval
- Auto-refresh every 60 seconds (TanStack Query `refetchInterval`)
- Supported symbols list

#### US-2.2: Data Management — Ingest & Progress
> **As a** user,
> **I want to** trigger data downloads and see real-time progress in the React app,
> **so that** I know the status of long-running data operations.

**Acceptance Criteria:**
- Ingest form: date range picker, multi-symbol select, preview (archived vs new months)
- Three action buttons: Ingest, Update, Rebuild
- SSE-connected log viewer showing real-time subprocess output (last 50 lines)
- Progress indicator with elapsed time
- Cancel button (triggers SIGTERM via API)
- Download history table

#### US-2.3: Data Management — Data Browser
> **As a** user,
> **I want to** browse raw OHLCV data with a candlestick chart in the React app,
> **so that** I can visually inspect data quality.

**Acceptance Criteria:**
- Symbol selector → date range selector → interval buttons (1m, 5m, 15m, 1h, 4h)
- TradingView Lightweight Charts candlestick with volume
- Data quality verification panel (gaps, OHLC errors)
- OHLCV table view with formatted datetime (TanStack Table)

#### US-2.4: Settings — Sizing & Risk Config CRUD
> **As a** user,
> **I want to** manage sizing and risk configurations in the React app,
> **so that** I can create/edit/delete configs for backtesting.

**Acceptance Criteria:**
- Sizing config list with create/edit/delete
- Dynamic form: fields change based on method (fixed_fractional / kelly / atr)
- Risk config list with create/edit/delete
- Strategy-level and portfolio-level limit fields
- Latency preset selector (read-only display)
- System info diagnostics (table row counts)

---

## Phase 3: Strategy Management

> **Goal:** Implement the most complex page — the strategy editor with visual/YAML/split modes.
> **Estimated effort:** 1.5 weeks
> **Prerequisite:** Phase 1 complete
> **Deliverable:** Full strategy CRUD with visual editor, YAML editor, condition builder, templates

### User Stories

#### US-3.1: Strategy List & Navigation
> **As a** user,
> **I want to** see all my strategies in a searchable, filterable list,
> **so that** I can quickly find and manage strategies.

**Acceptance Criteria:**
- Strategy cards with name, description, version, status badge (active/inactive)
- Search by name
- Filter by status (active/inactive/all)
- Create New button → opens editor
- Click card → opens editor with strategy loaded
- Delete with confirmation dialog

#### US-3.2: Visual Strategy Editor
> **As a** user,
> **I want to** edit strategies using a visual form interface,
> **so that** I don't need to write YAML manually.

**Acceptance Criteria:**
- General tab: name, description, version, timeframe
- Indicators tab: add/remove/configure indicators (20+ types with type-specific params)
- Conditions tab: entry long/short, exit long/short — condition builder with operator/value/logic
- Risk tab: stop loss config (pct/atr/indicator), take profit config (pct/atr/indicator/rr)
- Time Filters tab: sessions, blocked days, funding avoidance
- Sweep tab: parameter sweep configuration
- Form state managed by React Hook Form + Zod validation
- Changes auto-sync to YAML representation

#### US-3.3: YAML Editor
> **As a** user,
> **I want to** edit strategy DSL as raw YAML with real-time validation,
> **so that** I have full control over the strategy definition.

**Acceptance Criteria:**
- Code editor with YAML syntax highlighting (Monaco Editor or CodeMirror)
- Real-time validation: errors displayed inline with line numbers
- Auto-format button
- Changes sync back to visual form (bidirectional)

#### US-3.4: Split View Editor
> **As a** user,
> **I want to** see the visual form and YAML side-by-side,
> **so that** I can see the effect of my form changes in real-time.

**Acceptance Criteria:**
- Resizable split panel (drag divider)
- Left: visual form, Right: YAML preview (read-only or editable, toggle)
- Changes in either panel reflect in the other instantly
- No state leakage when switching between editor modes (React component lifecycle handles cleanup)

#### US-3.5: Strategy Templates
> **As a** user,
> **I want to** create strategies from templates,
> **so that** I can start from proven patterns instead of blank.

**Acceptance Criteria:**
- Template selector dropdown with categorized templates
- "Load Template" fills the editor with template content
- Template preview before loading
- Indicator catalog: browseable list of all indicators with parameter docs

#### US-3.6: Condition Builder Component
> **As a** user,
> **I want to** build entry/exit conditions using a visual builder,
> **so that** I can compose complex logic without writing YAML.

**Acceptance Criteria:**
- Add/remove condition rows
- Each row: left operand (indicator/price) → operator (>, <, crosses_above, crosses_below) → right operand (value/indicator)
- AND/OR logic between rows
- Drag to reorder (optional — nice to have)
- Validates against available indicators

---

## Phase 4: Backtest Launch & Results Analysis

> **Goal:** Implement the backtesting workflow — launch, monitor, and analyze results.
> **Estimated effort:** 1.5 weeks
> **Prerequisite:** Phase 0 (API), Phase 1 (charts)
> **Deliverable:** Full backtest launch workflow + comprehensive results dashboard with 15+ chart types

### User Stories

#### US-4.1: Backtest Launch Form
> **As a** user,
> **I want to** configure and launch screening/validation backtests from the React app,
> **so that** I can start backtests without the Streamlit session state issues.

**Acceptance Criteria:**
- Strategy selector with summary card (name, indicators, conditions)
- Symbol multi-select (BTC, ETH, SOL, etc.)
- Timeframe selector
- Date range with presets (1M, 3M, 6M, 1Y, 2Y, custom)
- Auto-generated parameter sweep fields (from strategy DSL)
- Overfitting filter toggles (Deflated Sharpe, Walk-Forward, Purged K-Fold)
- Sizing config selector (dropdown from settings)
- Risk config selector (dropdown from settings)
- Latency preset selector (co-located, retail, etc.)
- Preflight summary (estimated combinations, time estimate)
- "Run Screening" / "Run Validation" buttons
- Button disabled while another job of same type is running

#### US-4.2: Active Jobs Panel
> **As a** user,
> **I want to** see all active backtest/discovery jobs with real-time status updates,
> **so that** I know what's running and can kill stale jobs.

**Acceptance Criteria:**
- Active jobs list (fed by WebSocket, not polling)
- Per job: type (screening/validation/discovery), status badge, elapsed time, heartbeat indicator
- Stale job detection (>120s no heartbeat) with warning icon
- Kill button per job (confirmation dialog)
- Recent completed/failed runs list

#### US-4.3: Results — Single Run Analysis
> **As a** user,
> **I want to** see comprehensive analysis of a single backtest run,
> **so that** I can evaluate strategy performance in detail.

**Acceptance Criteria:**
- Run selector dropdown (cached, sorted by date)
- Metrics panel: 20+ metrics across 5 categories (returns, risk, trades, costs, perpetual)
- Win/loss analysis (counts, P&L, ratio, consecutive streaks)
- Cost breakdown (fees, funding, slippage, drag %)
- Perpetual futures analytics (gross P&L breakdown pie chart)
- Long vs short performance split
- Liquidation event summary
- Overfitting filter results (PASS/FAIL badges with values)

#### US-4.4: Results — Charts
> **As a** user,
> **I want to** see interactive charts for equity, drawdown, returns, and trade analysis,
> **so that** I can visually assess strategy behavior.

**Acceptance Criteria:**
- Equity curve (line chart with hover tooltip)
- Drawdown chart (area chart with top-5 periods highlighted)
- Rolling Sharpe ratio (line with reference lines at 1.0, 2.0)
- Yearly/daily returns (bar charts, green/red)
- Monthly returns heatmap (year × month, diverging colorscale)
- Trade P&L distribution (histogram, profit green / loss red)
- Trade scatter plots (ROI vs duration, size vs PnL)
- All charts: responsive, dark/light theme, zoom/pan, export as PNG

#### US-4.5: Results — Sweep & Pareto Analysis
> **As a** user,
> **I want to** see parameter sweep results with Pareto front visualization,
> **so that** I can identify optimal parameter combinations.

**Acceptance Criteria:**
- Sweep results table (TanStack Table with sort/filter/pagination)
- 2D Pareto scatter (Sharpe vs MaxDD, diamond markers for optimal)
- 3D Pareto surface (Sharpe × MaxDD × PF) with rotation/zoom
- Filter: show only Pareto-optimal, filter by overfitting pass/fail

#### US-4.6: Results — Trade Log
> **As a** user,
> **I want to** see a detailed trade log with filtering and export,
> **so that** I can audit individual trade decisions.

**Acceptance Criteria:**
- Trade log table with all fields (entry/exit time, price, PnL, fees, funding, reason)
- Filter by symbol, direction (long/short)
- Liquidation trades highlighted in red
- Sort by any column
- Export to CSV button

#### US-4.7: Results — Comparison View
> **As a** user,
> **I want to** compare up to 3 runs side-by-side,
> **so that** I can evaluate which strategy/parameter set performs best.

**Acceptance Criteria:**
- Multi-select: choose 2-3 runs for comparison
- Side-by-side metrics table (columns = runs, rows = metrics)
- Overlaid equity curves (different colors per run)
- Overlaid drawdown charts
- Radar profile comparison (normalized metrics)

#### US-4.8: Results — Notes & Export
> **As a** user,
> **I want to** annotate runs with notes and export data,
> **so that** I can record observations and do further analysis externally.

**Acceptance Criteria:**
- Editable notes section (auto-saves on blur)
- Export buttons: trades CSV, sweep results CSV, tearsheet JSON
- Notes persisted via `PUT /api/results/runs/{id}/notes`

---

## Phase 5: Discovery & Paper Trading (Real-Time)

> **Goal:** Implement the two most real-time-intensive pages using WebSocket push.
> **Estimated effort:** 1.5 weeks
> **Prerequisite:** Phase 0 (WebSocket handlers), Phase 1 (WS infrastructure)
> **Deliverable:** Real-time discovery monitoring + paper trading dashboard with instant position updates

### User Stories

#### US-5.1: Discovery — Configuration & Launch
> **As a** user,
> **I want to** configure and launch genetic algorithm discovery runs,
> **so that** I can automatically find profitable strategy patterns.

**Acceptance Criteria:**
- Config form: population size, generations, mutation rate, elite count, tournament size, convergence detection
- Symbol/timeframe/date-range selectors
- Indicator pool explorer: expandable list of indicators with parameter ranges
- "Start Discovery" button
- Active discovery jobs list (from WebSocket)

#### US-5.2: Discovery — Real-Time Progress
> **As a** user,
> **I want to** see live fitness evolution as discovery runs,
> **so that** I know when convergence is reached.

**Acceptance Criteria:**
- Fitness evolution chart: best/mean/worst per generation (updates live via WebSocket)
- Generation counter: current / total
- Chart updates incrementally (append new data points, don't redraw)
- Kill button per active discovery job

#### US-5.3: Discovery — Results & Export
> **As a** user,
> **I want to** see top discovered strategies and export them to my strategy library,
> **so that** I can validate and trade discovered patterns.

**Acceptance Criteria:**
- Top strategies table: rank, Sharpe, MaxDD, PF, trades, genes, fitness score
- Expandable row: full strategy detail with DSL YAML
- "Export to Strategy" button → creates strategy in library via API
- Copy YAML to clipboard button

#### US-5.4: Paper Trading — Session Control
> **As a** user,
> **I want to** start, halt, resume, and stop paper trading sessions,
> **so that** I have full control over live trading.

**Acceptance Criteria:**
- Strategy selector: only shows validated strategies (passed overfitting filters)
- Binance API credential input (password-masked, sent via secure header, never stored in frontend)
- Testnet toggle
- Control buttons: START, HALT (amber), RESUME (green), STOP (red)
- Confirmation dialog for STOP (closes all positions)
- Button states match session state (can't halt if not running, etc.)

#### US-5.5: Paper Trading — Live Dashboard
> **As a** user,
> **I want to** see real-time position and P&L data as it changes,
> **so that** I can monitor trading performance live.

**Acceptance Criteria:**
- Status indicator: colored dot + label (running/paused/halted/stopped/error/initializing)
- P&L metrics: total balance, available margin, margin used, daily P&L, total P&L
- Metrics update instantly via WebSocket (no 5s polling)
- Trades today / consecutive losses counter
- Open positions table: symbol, side, qty, entry price, current price, unrealized P&L (live updates)
- Pending orders table: symbol, side, type, qty, price, status
- Recent checkpoint timeline: state transitions with timestamps and daily P&L

---

## Phase 6: Polish, Testing & Streamlit Retirement

> **Goal:** Ensure feature parity, write E2E tests, optimize performance, and retire Streamlit.
> **Estimated effort:** 1-2 weeks
> **Prerequisite:** All previous phases complete
> **Deliverable:** Production-ready React dashboard; Streamlit code archived

### User Stories

#### US-6.1: Feature Parity Audit
> **As a** developer,
> **I want to** systematically verify every Streamlit feature has a React equivalent,
> **so that** no functionality is lost in the migration.

**Acceptance Criteria:**
- Checklist of all features from Section 2 (Current State Analysis) verified in React
- Side-by-side comparison screenshots (Streamlit vs React) for each page
- Any intentionally dropped features documented with rationale

#### US-6.2: E2E Test Suite
> **As a** developer,
> **I want to** have Playwright E2E tests covering all critical workflows,
> **so that** regressions are caught automatically.

**Acceptance Criteria:**
- Test: Data Management → download data → verify coverage table updates
- Test: Strategy Management → create strategy from template → verify saved
- Test: Backtest Launch → configure and launch screening → verify job appears
- Test: Results Analysis → select run → verify charts render → export CSV
- Test: Paper Trading → start session → verify status updates → halt → stop
- Test: Discovery → launch → verify progress chart updates → export strategy
- Tests run in CI (GitHub Actions) with headed Chromium

#### US-6.3: Performance Optimization
> **As a** developer,
> **I want to** ensure the React app is performant with large datasets,
> **so that** it handles real-world data volumes smoothly.

**Acceptance Criteria:**
- Trade log table: smooth scrolling with 10K+ rows (virtual scrolling if needed)
- Candlestick chart: renders 25K candles without frame drops
- WebSocket: handles 100+ messages/sec without UI jank (batched state updates)
- Bundle size < 500KB gzipped (code-split by page/route)
- Lighthouse performance score > 90

#### US-6.4: Development Workflow Integration
> **As a** developer,
> **I want to** have a smooth development workflow for the full-stack app,
> **so that** working on features is efficient.

**Acceptance Criteria:**
- `make dev` — starts both FastAPI (port 8000) and Vite dev server (port 3000) concurrently
- `make build` — builds React SPA and configures FastAPI to serve it as static files
- `make test` — runs pytest (backend) + vitest (frontend) + playwright (E2E)
- `make generate-api` — regenerates TypeScript types from OpenAPI spec
- API changes detected by CI (OpenAPI spec diff check)

#### US-6.5: Streamlit Retirement
> **As a** developer,
> **I want to** cleanly archive the Streamlit code,
> **so that** the codebase has a single frontend.

**Acceptance Criteria:**
- `vibe_quant/dashboard/` moved to `archive/streamlit-dashboard/` (preserved for reference)
- SPEC.md Section 10 updated to reflect React architecture
- CLAUDE.md updated: dashboard start command, UI testing approach
- All Streamlit dependencies removed from `pyproject.toml`
- `streamlit` removed from production dependencies (keep in dev if needed for comparison)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Scope creep** — "while we're at it" feature additions | High | Medium | Strict feature parity scope. New features only after Phase 6. |
| **API design churn** — changing endpoints during frontend dev | Medium | High | Phase 0 delivers stable OpenAPI spec before Phase 1 starts. Spec snapshot tests catch breaking changes. |
| **TypeScript learning curve** — Python team unfamiliar with TS | Medium | Medium | shadcn/ui provides copy-paste components. TanStack Query/Zustand have excellent TS docs. Start with Phase 2 (simplest pages) to build confidence. |
| **WebSocket reliability** — connection drops, message ordering | Medium | High | Exponential backoff reconnection. Last-event-id for catch-up. Heartbeat protocol. Fall back to REST polling if WS fails. |
| **Streamlit users disrupted** — migration takes longer than expected | Low | High | Parallel operation: Streamlit at `:8501`, React at `:3000`, both hit same SQLite DB during migration. |
| **Performance regression** — React app slower than expected | Low | Medium | Lighthouse CI gate. Bundle size monitoring. Virtual scrolling for large tables. |
| **Two codebases permanently** — migration stalls at Phase 3 | Medium | High | Phase 0 (API layer) is valuable standalone. Each phase delivers a complete, working subset. Can pause after any phase. |

---

## Decision Log

| # | Decision | Rationale | Alternatives Considered |
|---|----------|-----------|------------------------|
| D1 | **Vite over Next.js** | Internal dashboard; no SSR/SEO needed. Simpler deployment (static files). | Next.js (adds unnecessary server complexity) |
| D2 | **shadcn/ui over MUI/Ant Design** | Copy-paste ownership; no dependency lock-in; native Tailwind. 200+ components including charts and data tables. | Material UI (heavy, opinionated), Ant Design (large bundle) |
| D3 | **TanStack Query + Zustand over Redux** | Split by state type: server state (Query) vs client state (Zustand). Less boilerplate than Redux Toolkit. | Redux Toolkit (overkill), Recoil (abandoned), Jotai (too granular) |
| D4 | **TradingView Lightweight Charts over Plotly.js** | Canvas-based (not SVG); purpose-built for financial data; 10x smaller bundle; native real-time update API. | Plotly.js (heavier, SVG-based), Highcharts (commercial license) |
| D5 | **Recharts for supplementary charts** | Simplest JSX API; good enough for equity curves, histograms, heatmaps. | visx (too low-level), Nivo (heavy), Chart.js (imperative API) |
| D6 | **FastAPI over Flask/Django** | Async-native; auto-OpenAPI; Pydantic validation; native WebSocket support. Already familiar from Python ecosystem. | Flask (no async), Django (too heavy), Litestar (less ecosystem) |
| D7 | **pnpm over npm/yarn** | Fastest installs; strictest dependency resolution; disk-efficient. | npm (slower), yarn (no significant advantage over pnpm) |
| D8 | **Biome over ESLint+Prettier** | Single tool for lint+format; 10-100x faster than ESLint; fewer config files. | ESLint+Prettier (two tools, slower, more config) |
| D9 | **React Router over TanStack Router** | More mature; larger community; simpler for SPA routing. | TanStack Router (newer, type-safe but less ecosystem) |
| D10 | **Parallel operation during migration** | Zero disruption risk. Both UIs work simultaneously. | Big-bang cutover (high risk), iframe embedding (hacky) |

---

## Summary Timeline

```
Week 1-3:  Phase 0 — FastAPI API Layer (REST + WS + SSE)
Week 4:    Phase 1 — React Scaffold & Infrastructure
Week 5:    Phase 2 — Data Management & Settings
Week 5-6:  Phase 3 — Strategy Management
Week 6-7:  Phase 4 — Backtest Launch & Results Analysis
Week 7-8:  Phase 5 — Discovery & Paper Trading (Real-Time)
Week 9-10: Phase 6 — Polish, Testing & Streamlit Retirement
```

**Total: ~10 weeks** (conservative, solo developer estimate)

**Safe stopping points:** After Phase 0 (API is valuable standalone), after Phase 2 (validates stack), after Phase 4 (covers core workflow). Migration can be paused and resumed at any phase boundary.
