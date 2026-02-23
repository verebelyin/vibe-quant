import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function GuidePage() {
  const [content, setContent] = useState("");

  useEffect(() => {
    fetch("/guide.md")
      .then((r) => r.text())
      .then(setContent)
      .catch(() => setContent("Failed to load guide."));
  }, []);

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <article className="prose prose-invert prose-sm max-w-none prose-headings:text-foreground prose-p:text-muted-foreground prose-li:text-muted-foreground prose-strong:text-foreground prose-a:text-primary prose-th:text-foreground prose-td:text-muted-foreground prose-code:text-primary/90 prose-hr:border-border">
        <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
      </article>
    </div>
  );
}
