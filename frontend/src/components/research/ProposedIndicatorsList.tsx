import { Badge } from "@/components/ui/badge";

interface ProposedIndicator {
  name?: unknown;
  display_name?: unknown;
  description?: unknown;
  formula?: unknown;
  parameters?: unknown;
  output_range?: unknown;
  source_quote?: unknown;
}

interface Props {
  json: string | null | undefined;
}

function asString(v: unknown): string | null {
  return typeof v === "string" && v.trim().length > 0 ? v : null;
}

function parseProposals(json: string | null | undefined): ProposedIndicator[] {
  if (!json) return [];
  try {
    const parsed: unknown = JSON.parse(json);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x): x is ProposedIndicator => typeof x === "object" && x !== null);
  } catch {
    return [];
  }
}

export function ProposedIndicatorsList({ json }: Props) {
  const proposals = parseProposals(json);
  if (proposals.length === 0) return null;
  return (
    <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wide text-amber-300/90 font-medium">
          Proposed indicators ({proposals.length})
        </span>
        <span className="text-[10px] text-muted-foreground">
          not yet implemented · candidate for a plugin
        </span>
      </div>
      <ul className="flex flex-col gap-3">
        {proposals.map((p, idx) => {
          const name = asString(p.name);
          const display = asString(p.display_name);
          const desc = asString(p.description);
          const formula = asString(p.formula);
          const range = asString(p.output_range);
          const quote = asString(p.source_quote);
          const params =
            p.parameters && typeof p.parameters === "object"
              ? (p.parameters as Record<string, unknown>)
              : null;
          return (
            <li
              key={`${name ?? "anon"}-${idx}`}
              className="flex flex-col gap-2 text-xs border-t border-amber-500/20 first:border-t-0 first:pt-0 pt-3"
            >
              <div className="flex items-baseline flex-wrap gap-2">
                <code className="font-mono text-[11px] text-amber-300/95">
                  {name ?? "(unnamed)"}
                </code>
                {display && <span className="text-foreground/80">{display}</span>}
                {range && (
                  <Badge variant="outline" className="font-mono text-[10px] text-muted-foreground">
                    {range}
                  </Badge>
                )}
              </div>
              {desc && <p className="text-muted-foreground leading-relaxed">{desc}</p>}
              {formula && (
                <pre className="overflow-auto rounded border border-border/40 bg-muted/30 p-2 font-mono text-[10px] leading-relaxed text-foreground/85 whitespace-pre">
                  {formula}
                </pre>
              )}
              {params && Object.keys(params).length > 0 && (
                <div className="flex flex-col gap-0.5 text-[10px]">
                  <span className="uppercase tracking-wide text-muted-foreground">Parameters</span>
                  <ul className="font-mono text-foreground/80">
                    {Object.entries(params).map(([k, v]) => (
                      <li key={k}>
                        {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {quote && (
                <blockquote className="border-l-2 border-amber-500/30 pl-2 italic text-muted-foreground">
                  “{quote}”
                </blockquote>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
