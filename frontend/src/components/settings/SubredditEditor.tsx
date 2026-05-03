import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  getGetSubredditsApiResearchSettingsSubredditsGetQueryKey,
  useGetSubredditsApiResearchSettingsSubredditsGet,
  useResetSubredditsApiResearchSettingsSubredditsDelete,
  useSetSubredditsApiResearchSettingsSubredditsPut,
} from "@/api/generated/research/research";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const SUBREDDIT_RE = /^[a-z0-9_]{3,21}$/;

function validate(name: string): string | null {
  if (!name) return "Empty";
  if (name.startsWith("r/")) return "Drop the leading 'r/'";
  if (name !== name.toLowerCase()) return "Must be lowercase";
  if (!SUBREDDIT_RE.test(name))
    return "Lowercase letters, digits, underscores; 3-21 chars";
  return null;
}

export function SubredditEditor() {
  const queryClient = useQueryClient();
  const queryKey = getGetSubredditsApiResearchSettingsSubredditsGetQueryKey();
  const query = useGetSubredditsApiResearchSettingsSubredditsGet();
  const setMut = useSetSubredditsApiResearchSettingsSubredditsPut();
  const resetMut = useResetSubredditsApiResearchSettingsSubredditsDelete();

  const serverList =
    query.data?.status === 200 ? query.data.data.subreddits : [];
  const usingDefault =
    query.data?.status === 200 ? query.data.data.using_default : true;

  const [list, setList] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [draftError, setDraftError] = useState<string | null>(null);

  useEffect(() => {
    setList(serverList);
  }, [serverList]);

  const dirty = JSON.stringify(list) !== JSON.stringify(serverList);

  const addDraft = () => {
    const trimmed = draft.trim();
    const err = validate(trimmed);
    if (err) {
      setDraftError(err);
      return;
    }
    if (list.includes(trimmed)) {
      setDraftError("Already in list");
      return;
    }
    setList([...list, trimmed]);
    setDraft("");
    setDraftError(null);
  };

  const remove = (name: string) => {
    setList(list.filter((s) => s !== name));
  };

  const save = () => {
    if (list.length === 0) {
      toast.error("List must have at least one subreddit");
      return;
    }
    setMut.mutate(
      { data: { subreddits: list } },
      {
        onSuccess: (resp) => {
          if (resp.status === 200) {
            toast.success("Subreddit list saved");
            queryClient.invalidateQueries({ queryKey });
          } else {
            const detail = (resp.data as { detail?: unknown })?.detail;
            toast.error(typeof detail === "string" ? detail : "Save failed");
          }
        },
        onError: () => toast.error("Save failed"),
      },
    );
  };

  const restoreDefault = () => {
    resetMut.mutate(undefined, {
      onSuccess: (resp) => {
        if (resp.status === 200) {
          toast.success("Restored to default");
          queryClient.invalidateQueries({ queryKey });
        } else {
          toast.error("Restore failed");
        }
      },
      onError: () => toast.error("Restore failed"),
    });
  };

  return (
    <Card className="py-4">
      <CardContent>
        <div className="mb-3 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Reddit Subreddits
          </p>
          <Badge variant={usingDefault ? "secondary" : "default"}>
            {usingDefault ? "Default" : "Custom"}
          </Badge>
        </div>

        {query.isLoading ? (
          <p className="text-xs text-muted-foreground">Loading…</p>
        ) : (
          <>
            <div className="mb-3 flex flex-wrap gap-1.5">
              {list.length === 0 ? (
                <span className="text-xs text-muted-foreground">No subreddits — add one below.</span>
              ) : (
                list.map((name) => (
                  <Badge
                    key={name}
                    variant="outline"
                    className="gap-1.5 font-mono text-[11px]"
                  >
                    r/{name}
                    <button
                      type="button"
                      onClick={() => remove(name)}
                      className="text-muted-foreground hover:text-destructive"
                      aria-label={`Remove ${name}`}
                    >
                      ×
                    </button>
                  </Badge>
                ))
              )}
            </div>

            <div className="flex gap-2">
              <Input
                value={draft}
                onChange={(e) => {
                  setDraft(e.target.value);
                  if (draftError) setDraftError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addDraft();
                  }
                }}
                placeholder="add subreddit (e.g. algotrading)"
                className="h-8 text-xs font-mono"
              />
              <Button size="sm" variant="outline" onClick={addDraft}>
                Add
              </Button>
            </div>
            {draftError && (
              <p className="mt-1 text-[11px] text-destructive">{draftError}</p>
            )}

            <div className="mt-3 flex items-center gap-2">
              <Button
                size="sm"
                onClick={save}
                disabled={!dirty || setMut.isPending || list.length === 0}
              >
                {setMut.isPending ? "Saving…" : "Save"}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={restoreDefault}
                disabled={resetMut.isPending || (usingDefault && !dirty)}
              >
                {resetMut.isPending ? "Restoring…" : "Restore default"}
              </Button>
            </div>

            <p className="mt-3 text-[10px] text-muted-foreground">
              Saved list takes effect on the next scrape — no backend restart needed.
              Restore default reverts to the <code className="font-mono">REDDIT_SUBREDDITS</code>{" "}
              env var (or <code className="font-mono">algotrading</code> when unset).
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
