/**
 * C15-8 (vibe-quant-clmmc.15): UI code converts the RAW stored Sharpe/Sortino
 * through lib/metrics.ts; it must never read an API `_display` twin. Passing a
 * twin through the helper converts twice (0.96 → 1.16), which no type system
 * catches — this scan does, permanently.
 */
import { describe, expect, it } from "vitest";

// Every non-generated source file, as text; Vite resolves the glob at test time
// (no Node fs types are installed in this app tsconfig).
const SOURCES = import.meta.glob("/src/**/*.{ts,tsx}", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const FILES = Object.keys(SOURCES).filter(
  (path) => !path.startsWith("/src/api/generated/") && !/\.test\.tsx?$/.test(path),
);

describe("UI never reads an API *_display field (C15-8)", () => {
  it("scans the UI source tree, not an empty list", () => {
    expect(FILES.length).toBeGreaterThan(50);
    expect(FILES).toContain("/src/components/results/MetricsPanel.tsx");
  });

  it("has zero reads outside api/generated and tests", () => {
    const offenders = FILES.filter((path) => /_display\b/.test(SOURCES[path] ?? ""));
    expect(offenders).toEqual([]);
  });
});
