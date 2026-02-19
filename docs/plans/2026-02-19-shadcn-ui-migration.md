# shadcn/ui Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hand-rolled UI primitives with shadcn/ui (New York style) + OKLCH theming for professional, accessible trading dashboard.

**Architecture:** Initialize shadcn/ui on existing Vite+React+TW4 project. Migrate CSS vars from HSL to OKLCH. Replace primitives incrementally page by page. Keep domain composites, swap their internals.

**Tech Stack:** shadcn/ui (New York), Tailwind CSS v4, OKLCH colors, Radix UI primitives, tw-animate-css, Sonner toasts, TanStack Table via shadcn DataTable.

---

### Task 1: Initialize shadcn/ui Foundation

**Files:**
- Create: `frontend/components.json`
- Create: `frontend/src/lib/utils.ts`
- Modify: `frontend/src/index.css` (full rewrite to OKLCH + shadcn imports)
- Modify: `frontend/package.json` (new deps)

**Step 1: Install shadcn CLI and initialize**

```bash
cd frontend
npx shadcn@latest init
```

Select: New York style, Neutral base color, CSS variables yes, `src/index.css`, `@/` aliases (already configured).

**Step 2: Install core components**

```bash
npx shadcn@latest add button input label badge card tabs dialog
npx shadcn@latest add select checkbox switch toggle-group
npx shadcn@latest add tooltip skeleton scroll-area separator
npx shadcn@latest add sonner sidebar command
npx shadcn@latest add table dropdown-menu popover sheet
```

**Step 3: Add trading-specific CSS tokens to index.css**

Add after the `.dark` block:
```css
/* Trading semantic colors */
@theme inline {
  --color-profit: var(--profit);
  --color-loss: var(--loss);
  --color-warning: var(--warning);
}
```

And in `:root` / `.dark`:
```css
:root {
  --profit: oklch(0.723 0.219 149.579);
  --loss: oklch(0.577 0.245 27.325);
  --warning: oklch(0.769 0.188 70.08);
}
.dark {
  --profit: oklch(0.723 0.219 149.579);
  --loss: oklch(0.704 0.191 22.216);
  --warning: oklch(0.828 0.189 84.429);
}
```

**Step 4: Add tabular-nums globally**

```css
@layer base {
  * {
    @apply border-border outline-ring/50;
    font-variant-numeric: tabular-nums;
  }
  body {
    @apply bg-background text-foreground;
  }
}
```

**Step 5: Verify build**

```bash
npx tsc --noEmit && npx vite build
```

**Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: init shadcn/ui New York + OKLCH theme"
```

---

### Task 2: Migrate Layout (Sidebar + Header)

**Files:**
- Rewrite: `frontend/src/components/layout/Sidebar.tsx` (use shadcn Sidebar)
- Rewrite: `frontend/src/components/layout/Header.tsx` (use shadcn Button, Tooltip)
- Modify: `frontend/src/components/layout/PageLayout.tsx` (wire SidebarProvider)
- Modify: `frontend/src/App.tsx` (add SidebarProvider + Toaster)

**Step 1:** Replace Sidebar with shadcn `<Sidebar>` component using `SidebarProvider`, `SidebarMenu`, `SidebarMenuItem`, `SidebarMenuButton`. Keep same nav items and route paths.

**Step 2:** Replace Header theme toggle with shadcn `<Button variant="ghost" size="icon">` + `<Tooltip>`. Add SVG `<title>` elements to fix a11y warnings.

**Step 3:** Add `<Toaster />` from Sonner to App.tsx root.

**Step 4:** Verify build + browser test.

**Step 5:** Commit.

---

### Task 3: Migrate Settings Page

**Files:**
- Modify: `frontend/src/routes/settings.tsx` (use shadcn Tabs)
- Modify: `frontend/src/components/settings/SizingTab.tsx` (Button, Input, Label, Card, Select)
- Modify: `frontend/src/components/settings/RiskTab.tsx` (same)
- Modify: `frontend/src/components/settings/LatencyTab.tsx` (Card, Badge)
- Modify: `frontend/src/components/settings/DatabaseTab.tsx` (Card, Input, Button)
- Modify: `frontend/src/components/settings/SystemTab.tsx` (Card, Badge)

**Step 1:** Replace settings tab nav with `<Tabs>` + `<TabsList>` + `<TabsTrigger>` + `<TabsContent>`.

**Step 2:** In each tab, replace raw `<input>` with `<Input>`, `<select>` with `<Select>`, `<button>` with `<Button>`, `<label>` with `<Label>`. Wrap config cards in `<Card>`.

**Step 3:** Remove all inline `style={{ color: "hsl(var(--...))" }}` — use Tailwind utilities (`text-foreground`, `text-muted-foreground`, `border-border`, `bg-card`).

**Step 4:** Verify, commit.

---

### Task 4: Migrate Data Management Page

**Files:**
- Modify: `frontend/src/components/data/DataStatusDashboard.tsx` (Card, Skeleton)
- Modify: `frontend/src/components/data/CoverageTable.tsx` (shadcn Table or DataTable)
- Modify: `frontend/src/components/data/IngestForm.tsx` (Button, Input, Label, Select, Checkbox)
- Modify: `frontend/src/components/data/DownloadProgress.tsx` (Card, Badge)
- Modify: `frontend/src/components/data/DownloadHistory.tsx` (Table)
- Modify: `frontend/src/components/data/DataBrowser.tsx` (Select, Card)
- Modify: `frontend/src/components/data/DataQualityPanel.tsx` (Card, Badge)

**Step 1:** Replace MetricCard inline component with shadcn `<Card>` + `<CardHeader>` + `<CardContent>`.

**Step 2:** Replace CoverageTable with shadcn `<Table>` + `<TableHeader>` + `<TableRow>` + `<TableCell>`.

**Step 3:** Migrate IngestForm: `<Label>` + `<Input>` + `<Button>` + `<Select>` + `<Checkbox>`.

**Step 4:** Add `toast()` calls from Sonner after download/update/rebuild mutations.

**Step 5:** Remove all inline `hsl(var(--...))` styles. Verify, commit.

---

### Task 5: Migrate Strategies Page

**Files:**
- Modify: `frontend/src/routes/strategies.tsx` (Button)
- Modify: `frontend/src/components/strategies/StrategyList.tsx` (Input, Select, Card)
- Modify: `frontend/src/components/strategies/StrategyCreateDialog.tsx` (Dialog, Button, Input, Label)
- Modify: `frontend/src/components/strategies/StrategyDeleteDialog.tsx` (Dialog, Button)
- Modify: `frontend/src/components/ui/StrategyCard.tsx` (Card)

**Step 1:** Replace backdrop-div modals with shadcn `<Dialog>` + `<DialogContent>` + `<DialogHeader>` + `<DialogFooter>`. This fixes focus trap a11y issue.

**Step 2:** Replace search input with `<Input>`, filter/sort selects with `<Select>`.

**Step 3:** Wrap StrategyCard in shadcn `<Card>`. Add toast on create/delete success.

**Step 4:** Verify, commit.

---

### Task 6: Migrate Backtest Page

**Files:**
- Modify: `frontend/src/components/backtest/BacktestLaunchForm.tsx` (Input, Label, Select, Button, Checkbox)
- Modify: `frontend/src/components/backtest/ActiveJobsPanel.tsx` (Card, Table, Badge, Button)
- Modify: `frontend/src/components/backtest/PreflightStatus.tsx` (Badge)

**Step 1:** Replace all 12 raw `<label>` elements with `<Label htmlFor="...">` and add `id` to corresponding inputs. This fixes all 12 `noLabelWithoutControl` biome warnings.

**Step 2:** Replace raw inputs/selects/buttons with shadcn components.

**Step 3:** Replace ActiveJobsPanel table with shadcn `<Table>`. Replace `JobStatusBadge` with shadcn `<Badge>` variants.

**Step 4:** Add toast on launch success/failure.

**Step 5:** Verify, commit.

---

### Task 7: Migrate Results Page

**Files:**
- Modify: `frontend/src/routes/results.tsx` (Select)
- Modify: `frontend/src/components/results/RunSelector.tsx` (Select, Badge)
- Modify: `frontend/src/components/results/MetricsPanel.tsx` (Card, Skeleton)
- Modify: `frontend/src/components/results/CostBreakdown.tsx` (Card)
- Modify: `frontend/src/components/results/OverfittingBadges.tsx` (Card, Badge)

**Step 1:** Replace RunSelector raw selects with shadcn `<Select>`.

**Step 2:** Wrap MetricsPanel cards in shadcn `<Card>`.

**Step 3:** Replace overfitting pass/fail indicators with `<Badge variant="default|destructive">`.

**Step 4:** Verify, commit.

---

### Task 8: Final Polish + Cleanup

**Files:**
- Delete: `frontend/src/components/ui/LoadingSpinner.tsx` (replaced by Skeleton)
- Delete: `frontend/src/components/ui/MetricCard.tsx` (replaced by Card)
- Delete: `frontend/src/components/ui/JobStatusBadge.tsx` (replaced by Badge)
- Modify: `frontend/src/components/ui/index.ts` (update exports)
- Modify: all remaining files with inline `hsl(var(--...))` styles

**Step 1:** Search for remaining `hsl(var(--` and replace with Tailwind utility classes.

**Step 2:** Search for remaining raw `<button>`, `<input>`, `<select>`, `<label>` and replace.

**Step 3:** Delete unused hand-rolled components.

**Step 4:** Run `npx biome check src/` — target 0 errors, <10 warnings.

**Step 5:** Run `npx tsc --noEmit && npx vite build`.

**Step 6:** Browser test all 7 pages in dark + light mode.

**Step 7:** Final commit + push.

---

## Execution Notes

- Each task is independently committable
- Tasks 2-7 can be parallelized with subagents (one per page)
- Task 1 (foundation) must complete first — all others depend on it
- Keep domain logic untouched — only swap UI primitives
