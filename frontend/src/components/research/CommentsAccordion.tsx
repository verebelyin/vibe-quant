interface Comment {
  author?: string | null;
  body?: string | null;
  score?: number | null;
}

interface Props {
  comments: Comment[];
}

export function CommentsAccordion({ comments }: Props) {
  if (comments.length === 0) {
    return <p className="text-xs text-muted-foreground italic">No comments captured.</p>;
  }
  return (
    <details className="group rounded-md border border-border/40 bg-card/40">
      <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-foreground/80 hover:text-foreground select-none flex items-center justify-between">
        <span>Top comments ({comments.length})</span>
        <span className="text-muted-foreground transition-transform group-open:rotate-90">›</span>
      </summary>
      <ul className="divide-y divide-border/40 border-t border-border/40">
        {comments.map((c, idx) => (
          <li key={`${c.author ?? "anon"}-${idx}`} className="px-3 py-2 text-xs">
            <div className="mb-1 flex items-center gap-2 text-[10px] text-muted-foreground tabular-nums">
              <span className="font-mono">{c.author ?? "[deleted]"}</span>
              {typeof c.score === "number" && <span>· {c.score} ↑</span>}
            </div>
            <div className="whitespace-pre-wrap text-foreground/85">{c.body ?? ""}</div>
          </li>
        ))}
      </ul>
    </details>
  );
}
