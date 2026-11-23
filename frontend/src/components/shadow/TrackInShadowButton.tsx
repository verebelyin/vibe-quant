import { Link } from "@tanstack/react-router";
import { useState } from "react";
import { apiErrorDetail } from "@/api/client";
import type { ShadowStrategy } from "@/api/generated/models/shadowStrategy";
import {
  useGetFrozenDslApiResultsRunsRunIdFrozenDslGet,
  useGetRunMetaApiResultsRunsRunIdMetaGet,
} from "@/api/generated/results/results";
import {
  getListShadowApiShadowGetQueryKey,
  useEnrolShadowApiShadowEnrolPost,
  useListShadowApiShadowGet,
} from "@/api/generated/shadow/shadow";
import { queryClient } from "@/api/query-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import {
  buildEnrolRequest,
  defaultEnrolForm,
  type EnrolForm,
  PREREG_HINT,
  PREREG_MIN_CHARS,
  preRegistrationShortBy,
  shadowEligible,
  trackedRowForRun,
} from "./enrolRequest";

function ShadowLink({ row }: { row: Pick<ShadowStrategy, "id" | "name" | "state"> }) {
  return (
    <Link to="/shadow" className="text-sm text-primary underline-offset-4 hover:underline">
      Open Shadow tracker (#{row.id} {row.name}, {row.state})
    </Link>
  );
}

/** "Track in shadow" on a completed validation run: freezes the DSL the run executed. */
export function TrackInShadowButton({ runId }: { runId: number }) {
  const meta = useGetRunMetaApiResultsRunsRunIdMetaGet(runId);
  const run = meta.data?.status === 200 ? meta.data.data : null;
  const eligible = run != null && shadowEligible(run);
  const frozenQuery = useGetFrozenDslApiResultsRunsRunIdFrozenDslGet(runId, {
    query: { enabled: eligible, retry: false },
  });
  const frozen = frozenQuery.data?.status === 200 ? frozenQuery.data.data : null;
  const listQuery = useListShadowApiShadowGet({ query: { enabled: eligible } });
  const rows = listQuery.data?.status === 200 ? listQuery.data.data.strategies : [];
  const [form, setForm] = useState<EnrolForm | null>(null);
  const [refusal, setRefusal] = useState<string | null>(null);
  const enrol = useEnrolShadowApiShadowEnrolPost({
    mutation: {
      onSuccess: () => {
        setForm(null);
        queryClient.invalidateQueries({ queryKey: getListShadowApiShadowGetQueryKey() });
      },
      onError: (err) => setRefusal(apiErrorDetail(err) ?? String(err)),
    },
  });

  if (!run || !eligible) return null;
  const tracked = trackedRowForRun(rows, runId);
  if (tracked) {
    return (
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <span>Already tracked in shadow as #{tracked.id}.</span>
        <ShadowLink row={tracked} />
      </div>
    );
  }
  const blocked = frozenQuery.isError ? apiErrorDetail(frozenQuery.error) : undefined;
  const set = (patch: Partial<EnrolForm>) => form && setForm({ ...form, ...patch });
  const shortBy = form ? preRegistrationShortBy(form.preRegistration) : 0;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <Button
          size="sm"
          variant="outline"
          disabled={!frozen || form != null}
          onClick={() => {
            setRefusal(null);
            setForm(defaultEnrolForm(frozen?.strategy_name ?? "", runId));
          }}
        >
          Track in shadow
        </Button>
        {blocked && <span className="text-sm text-destructive">{blocked}</span>}
      </div>
      {form && frozen && (
        <Card>
          <CardContent className="space-y-4 pt-4">
            <div className="space-y-1">
              <Label htmlFor="shadow-name">Name</Label>
              <Input
                id="shadow-name"
                value={form.name}
                onChange={(e) => set({ name: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="shadow-prereg">Pre-registration</Label>
              <textarea
                id="shadow-prereg"
                placeholder={PREREG_HINT}
                value={form.preRegistration}
                onChange={(e) => set({ preRegistration: e.target.value })}
                className={cn(
                  "min-h-[90px] w-full resize-y rounded-md border border-input bg-transparent px-3 py-2 text-sm",
                  "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                )}
              />
              <p className="text-xs text-muted-foreground">
                {PREREG_HINT}{" "}
                {shortBy > 0
                  ? `At least ${PREREG_MIN_CHARS} characters (${shortBy} more).`
                  : "Length OK."}
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1">
                <Label htmlFor="shadow-dd">Max drawdown limit (%)</Label>
                <Input
                  id="shadow-dd"
                  type="number"
                  value={form.maxDdPercent}
                  onChange={(e) => set({ maxDdPercent: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="shadow-warmup">Warm-up (days)</Label>
                <Input
                  id="shadow-warmup"
                  type="number"
                  value={form.warmupDays}
                  onChange={(e) => set({ warmupDays: e.target.value })}
                />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Freezes the DSL this run validated ({run.symbols.join(", ")}, DSL{" "}
              {frozen.dsl_sha256.slice(0, 12)}) with the run's parameters.
            </p>
            {refusal && <p className="text-sm text-destructive">{refusal}</p>}
            <div className="flex gap-2">
              <Button
                size="sm"
                disabled={enrol.isPending}
                onClick={() => {
                  setRefusal(null);
                  enrol.mutate({ data: buildEnrolRequest(run, frozen, form) });
                }}
              >
                {enrol.isPending ? "Enrolling…" : "Enrol"}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setForm(null)}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
      {enrol.data?.status === 201 && <ShadowLink row={enrol.data.data} />}
    </div>
  );
}
