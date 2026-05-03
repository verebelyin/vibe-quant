import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Comment {
  author?: string | null;
  body?: string | null;
  score?: number | null;
}

interface Props {
  comments: Comment[];
}

export function CommentsList({ comments }: Props) {
  if (comments.length === 0) {
    return <p className="text-xs text-muted-foreground italic">No comments captured.</p>;
  }
  return (
    <div className="rounded-md border border-border/40 bg-card/40">
      <div className="border-b border-border/40 px-3 py-2 text-xs font-medium text-foreground/80">
        Top comments ({comments.length})
      </div>
      <ul className="divide-y divide-border/40">
        {comments.map((c, idx) => (
          <li key={`${c.author ?? "anon"}-${idx}`} className="px-3 py-3 text-xs">
            <div className="mb-1.5 flex items-center gap-2 text-[10px] text-muted-foreground tabular-nums">
              <span className="font-mono text-foreground/70">{c.author ?? "[deleted]"}</span>
              {typeof c.score === "number" && <span>· {c.score} ↑</span>}
            </div>
            {c.body ? (
              <article className="prose prose-invert prose-sm max-w-none prose-p:my-1 prose-p:text-foreground/85 prose-li:text-foreground/85 prose-li:my-0.5 prose-headings:text-foreground prose-strong:text-foreground prose-a:text-primary prose-code:text-primary/90 prose-pre:bg-muted/40 prose-pre:border prose-pre:border-border/40 prose-blockquote:text-muted-foreground prose-blockquote:border-l-border/60">
                <Markdown remarkPlugins={[remarkGfm]}>{c.body}</Markdown>
              </article>
            ) : (
              <span className="text-muted-foreground italic">(empty)</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
