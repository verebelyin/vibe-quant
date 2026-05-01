import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CommentsAccordion } from "@/components/research/CommentsAccordion";

interface Comment {
  author?: string | null;
  body?: string | null;
  score?: number | null;
}

interface Props {
  body: string | null;
  comments: Comment[];
}

export function PostContent({ body, comments }: Props) {
  const hasBody = typeof body === "string" && body.trim().length > 0;
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-md border border-border/40 bg-card/40 p-4">
        {hasBody ? (
          <article className="prose prose-invert prose-sm max-w-none prose-headings:text-foreground prose-p:text-muted-foreground prose-li:text-muted-foreground prose-strong:text-foreground prose-a:text-primary prose-code:text-primary/90 prose-pre:bg-muted/40 prose-pre:border prose-pre:border-border/40">
            <Markdown remarkPlugins={[remarkGfm]}>{body}</Markdown>
          </article>
        ) : (
          <p className="text-xs text-muted-foreground italic">(no body)</p>
        )}
      </div>
      <CommentsAccordion comments={comments} />
    </div>
  );
}
