interface Props {
  yaml: string;
}

export function YamlViewer({ yaml }: Props) {
  return (
    <pre className="overflow-auto rounded-md border border-border/40 bg-muted/30 p-3 font-mono text-[11px] leading-relaxed text-foreground/90 whitespace-pre">
      {yaml}
    </pre>
  );
}
