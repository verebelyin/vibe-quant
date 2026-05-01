import { Link } from "@tanstack/react-router";
import { ChevronLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  itemId: number;
}

export function ResearchItemPage({ itemId }: Props) {
  return (
    <div className="flex flex-col h-full gap-4">
      <div>
        <Button asChild variant="ghost" size="sm" className="gap-1.5 h-8 text-xs">
          <Link to="/research">
            <ChevronLeft className="h-3 w-3" />
            Back to research
          </Link>
        </Button>
      </div>
      <div className="rounded-md border border-border/40 bg-card/40 p-6">
        <h2 className="text-sm font-semibold">Item #{itemId}</h2>
        <p className="mt-2 text-xs text-muted-foreground">Detail view coming soon (T4.3).</p>
      </div>
    </div>
  );
}
