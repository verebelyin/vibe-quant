import { useCallback, useEffect, useRef, useState } from "react";
import {
  useGetRunSummaryApiResultsRunsRunIdGet,
  useUpdateNotesApiResultsRunsRunIdNotesPut,
} from "@/api/generated/results/results";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface NotesPanelProps {
  runId: number;
}

const DEBOUNCE_MS = 1000;

export function NotesPanel({ runId }: NotesPanelProps) {
  const summaryQuery = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const mutation = useUpdateNotesApiResultsRunsRunIdNotesPut();

  const serverNotes = (summaryQuery.data?.data as Record<string, unknown> | undefined)?.notes as string ?? "";
  const [text, setText] = useState(serverNotes);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const initializedRef = useRef(false);
  const prevRunIdRef = useRef(runId);

  // Reset when runId changes
  if (prevRunIdRef.current !== runId) {
    prevRunIdRef.current = runId;
    initializedRef.current = false;
    setSaveState("idle");
    setLastSaved(null);
  }

  // Sync from server on first load
  useEffect(() => {
    if (!summaryQuery.isLoading && !initializedRef.current) {
      setText(serverNotes);
      initializedRef.current = true;
    }
  }, [summaryQuery.isLoading, serverNotes]);

  const saveNotes = useCallback(
    (value: string) => {
      setSaveState("saving");
      mutation.mutate(
        { runId, data: { notes: value } },
        {
          onSuccess: () => {
            setSaveState("saved");
            setLastSaved(new Date());
            setTimeout(() => setSaveState("idle"), 2000);
          },
          onError: () => {
            setSaveState("idle");
          },
        },
      );
    },
    [runId, mutation],
  );

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setText(value);

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => saveNotes(value), DEBOUNCE_MS);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Notes
        </CardTitle>
        <div className="ml-auto flex items-center gap-2">
          {saveState === "saving" && (
            <span className="text-xs text-muted-foreground">Saving...</span>
          )}
          {saveState === "saved" && <span className="text-xs text-green-500">Saved</span>}
          {lastSaved && saveState === "idle" && (
            <span className="text-xs text-muted-foreground">
              Last saved {lastSaved.toLocaleTimeString()}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <textarea
          className={cn(
            "min-h-[100px] w-full resize-y rounded-md border border-input bg-transparent px-3 py-2 text-sm",
            "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          )}
          placeholder="Add notes about this run..."
          value={text}
          onChange={handleChange}
        />
      </CardContent>
    </Card>
  );
}
