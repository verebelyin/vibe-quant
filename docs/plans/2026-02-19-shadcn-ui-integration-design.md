# shadcn/ui Integration Design

**Date:** 2026-02-19
**Status:** Approved
**Style:** New York

## Problem

Frontend uses 41 hand-rolled components with raw Tailwind. Missing: focus traps, keyboard nav, ARIA attrs, consistent styling, animations. Biome flags 31 a11y warnings. Looks like a developer prototype, not a professional trading dashboard.

## Decision

Integrate shadcn/ui (New York style) with OKLCH theming on Tailwind v4. Incremental migration — replace hand-rolled primitives page by page, keep domain-specific composites.

## Context7 Findings

- shadcn/ui Tailwind v4: `@import "shadcn/tailwind.css"`, `@theme inline`, `@custom-variant dark`
- OKLCH replaces HSL for all CSS vars (perceptually uniform, more vibrant)
- `components.json` with `"style": "new-york"`, `"cssVariables": true`
- Components install via `npx shadcn@latest add <name>`
- Sidebar, Sonner, DataTable all have dedicated components

## Architecture

### Three-tier component structure

```
components/
  ui/           # Raw shadcn components (auto-added, minimal edits)
  primitives/   # Light wrappers (MetricCard, StatusBadge)
  [domain]/     # Product composites (backtest/, data/, strategies/, etc.)
```

### CSS Migration: HSL → OKLCH

**Before (current):**
```css
:root { --background: 222.2 84% 4.9%; }
/* Usage: style={{ color: "hsl(var(--foreground))" }} */
```

**After (shadcn/ui + Tailwind v4):**
```css
@import "tailwindcss";
@import "tw-animate-css";
@import "shadcn/tailwind.css";

@custom-variant dark (&:is(.dark *));

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  /* ... all tokens mapped */
}

:root {
  --radius: 0.625rem;
  --background: oklch(1 0 0);
  --foreground: oklch(0.145 0 0);
  /* ... */
}

.dark {
  --background: oklch(0.145 0 0);
  --foreground: oklch(0.985 0 0);
  /* ... */
}
```

**Usage becomes:** `className="bg-background text-foreground"` instead of inline `style` props.

### Dark Theme Color Palette (Trading-Optimized)

| Token | OKLCH Value | Purpose |
|-------|------------|---------|
| `--background` | `oklch(0.145 0 0)` | Deep dark bg |
| `--card` | `oklch(0.205 0 0)` | Elevated surfaces |
| `--primary` | `oklch(0.488 0.243 264.376)` | Blue accent (interactive) |
| `--destructive` | `oklch(0.704 0.191 22.216)` | Red/loss/danger |
| `--chart-1` | `oklch(0.488 0.243 264.376)` | Blue chart line |
| `--chart-2` | `oklch(0.696 0.17 162.48)` | Green/profit chart |
| `--chart-3` | `oklch(0.769 0.188 70.08)` | Amber chart |
| `--chart-4` | `oklch(0.627 0.265 303.9)` | Purple chart |
| `--chart-5` | `oklch(0.645 0.246 16.439)` | Rose chart |

Custom semantic tokens (added for trading):
```css
.dark {
  --profit: oklch(0.723 0.219 149.579);   /* green-500 */
  --loss: oklch(0.637 0.237 25.331);       /* red-500 */
  --warning: oklch(0.769 0.188 70.08);     /* amber-500 */
}
```

### Typography

- Body: system sans-serif (Tailwind default)
- Numbers: `font-variant-numeric: tabular-nums` globally
- Monospace: system monospace for run IDs, prices, timestamps

## Component Migration Map

### Phase 1: Foundation (init + base components)

```bash
npx shadcn@latest init        # New York, neutral base, OKLCH
npx shadcn@latest add button input label badge card tabs dialog
npx shadcn@latest add select checkbox switch toggle-group
npx shadcn@latest add tooltip skeleton scroll-area separator
npx shadcn@latest add sonner sidebar command
npx shadcn@latest add table    # + @tanstack/react-table
```

### Phase 2: Replace hand-rolled → shadcn

| Current | Replacement | Files affected |
|---------|------------|----------------|
| Raw `<button>` | `<Button variant="...">` | All pages |
| Raw `<select>` | `<Select>` | BacktestLaunchForm, IngestForm, RunSelector |
| Raw `<input>` | `<Input>` | BacktestLaunchForm, IngestForm, SizingTab, RiskTab |
| Raw `<label>` | `<Label>` | All forms (fixes 12 a11y warnings) |
| Backdrop div dialogs | `<Dialog>` | StrategyCreateDialog, StrategyDeleteDialog |
| Hand-rolled tabs | `<Tabs>` | Settings page |
| `JobStatusBadge` | `<Badge variant="...">` | ActiveJobsPanel |
| `LoadingSpinner` | `<Skeleton>` | All loading states |
| `MetricCard` | `<Card>` + content | DataStatusDashboard, MetricsPanel |
| `EmptyState` | Keep (no shadcn equiv) | — |
| CoverageTable | `<DataTable>` (TanStack) | Data page |
| Hand-rolled sidebar | `<Sidebar>` | Layout |
| No toasts | `<Sonner>` | After mutations |

### Phase 3: Polish

- Command palette (Cmd+K): symbol search, page nav, quick actions
- `tabular-nums` on all numeric displays
- Remove all inline `style={{ color: "hsl(var(--...))" }}` → Tailwind utilities
- Add `tw-animate-css` for dialog/sheet/popover animations

## What Stays

- All chart components (CandlestickChart, Recharts, Plotly) — no shadcn equivalent
- Domain composites (BacktestLaunchForm, ActiveJobsPanel, IngestForm, etc.) — just swap primitives inside
- WebSocket hooks, API layer, stores — untouched
- Route structure — untouched

## Migration Strategy

**Incremental, not big-bang.** Each page migrated independently, merged separately.

Order: Foundation → Layout (Sidebar) → Settings → Data → Strategies → Backtest → Results → Discovery/Paper Trading

## Unresolved Questions

None — Context7 docs confirm all APIs and approaches.
