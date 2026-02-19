import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { SizingConfigResponse } from "@/api/generated/models";
import {
  getListSizingConfigsApiSettingsSizingGetQueryKey,
  useCreateSizingConfigApiSettingsSizingPost,
  useDeleteSizingConfigApiSettingsSizingConfigIdDelete,
  useListSizingConfigsApiSettingsSizingGet,
  useUpdateSizingConfigApiSettingsSizingConfigIdPut,
} from "@/api/generated/settings/settings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/input";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type SizingMethod = "fixed_fractional" | "kelly" | "atr";

interface SizingFormState {
  name: string;
  method: SizingMethod;
  max_leverage: string;
  max_position_pct: string;
  risk_per_trade: string;
  win_rate: string;
  avg_win: string;
  avg_loss: string;
  kelly_fraction: string;
  atr_multiplier: string;
}

const EMPTY_FORM: SizingFormState = {
  name: "",
  method: "fixed_fractional",
  max_leverage: "10",
  max_position_pct: "0.1",
  risk_per_trade: "0.02",
  win_rate: "0.55",
  avg_win: "0.03",
  avg_loss: "0.02",
  kelly_fraction: "0.5",
  atr_multiplier: "2.0",
};

function buildConfig(form: SizingFormState): Record<string, unknown> {
  const base: Record<string, unknown> = {
    max_leverage: Number(form.max_leverage),
    max_position_pct: Number(form.max_position_pct),
  };
  if (form.method === "fixed_fractional" || form.method === "atr") {
    base.risk_per_trade = Number(form.risk_per_trade);
  }
  if (form.method === "kelly") {
    base.win_rate = Number(form.win_rate);
    base.avg_win = Number(form.avg_win);
    base.avg_loss = Number(form.avg_loss);
    base.kelly_fraction = Number(form.kelly_fraction);
  }
  if (form.method === "atr") {
    base.atr_multiplier = Number(form.atr_multiplier);
  }
  return base;
}

function formFromConfig(cfg: SizingConfigResponse): SizingFormState {
  const c = cfg.config as Record<string, unknown>;
  return {
    name: cfg.name,
    method: cfg.method as SizingMethod,
    max_leverage: String(c.max_leverage ?? "10"),
    max_position_pct: String(c.max_position_pct ?? "0.1"),
    risk_per_trade: String(c.risk_per_trade ?? "0.02"),
    win_rate: String(c.win_rate ?? "0.55"),
    avg_win: String(c.avg_win ?? "0.03"),
    avg_loss: String(c.avg_loss ?? "0.02"),
    kelly_fraction: String(c.kelly_fraction ?? "0.5"),
    atr_multiplier: String(c.atr_multiplier ?? "2.0"),
  };
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <Input type="number" step="any" value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function SizingForm({
  initial,
  onSubmit,
  onCancel,
  isPending,
  submitLabel,
}: {
  initial: SizingFormState;
  onSubmit: (form: SizingFormState) => void;
  onCancel: () => void;
  isPending: boolean;
  submitLabel: string;
}) {
  const [form, setForm] = useState<SizingFormState>(initial);
  const set = (k: keyof SizingFormState, v: string) => setForm((prev) => ({ ...prev, [k]: v }));

  return (
    <Card className="py-4">
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1 sm:col-span-2">
            <Label className="text-xs text-muted-foreground">Name</Label>
            <Input type="text" value={form.name} onChange={(e) => set("name", e.target.value)} />
          </div>

          <div className="space-y-1 sm:col-span-2">
            <Label className="text-xs text-muted-foreground">Method</Label>
            <Select value={form.method} onValueChange={(v) => set("method", v)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="fixed_fractional">Fixed Fractional</SelectItem>
                <SelectItem value="kelly">Kelly Criterion</SelectItem>
                <SelectItem value="atr">ATR-based</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <NumberField
            label="Max Leverage"
            value={form.max_leverage}
            onChange={(v) => set("max_leverage", v)}
          />
          <NumberField
            label="Max Position %"
            value={form.max_position_pct}
            onChange={(v) => set("max_position_pct", v)}
          />

          {(form.method === "fixed_fractional" || form.method === "atr") && (
            <NumberField
              label="Risk per Trade"
              value={form.risk_per_trade}
              onChange={(v) => set("risk_per_trade", v)}
            />
          )}

          {form.method === "kelly" && (
            <>
              <NumberField
                label="Win Rate"
                value={form.win_rate}
                onChange={(v) => set("win_rate", v)}
              />
              <NumberField
                label="Avg Win"
                value={form.avg_win}
                onChange={(v) => set("avg_win", v)}
              />
              <NumberField
                label="Avg Loss"
                value={form.avg_loss}
                onChange={(v) => set("avg_loss", v)}
              />
              <NumberField
                label="Kelly Fraction"
                value={form.kelly_fraction}
                onChange={(v) => set("kelly_fraction", v)}
              />
            </>
          )}

          {form.method === "atr" && (
            <NumberField
              label="ATR Multiplier"
              value={form.atr_multiplier}
              onChange={(v) => set("atr_multiplier", v)}
            />
          )}
        </div>

        <div className="mt-4 flex gap-2">
          <Button
            size="sm"
            disabled={isPending || !form.name.trim()}
            onClick={() => onSubmit(form)}
          >
            {isPending ? "Saving..." : submitLabel}
          </Button>
          <Button variant="outline" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export function SizingTab() {
  const qc = useQueryClient();
  const query = useListSizingConfigsApiSettingsSizingGet();
  const createMut = useCreateSizingConfigApiSettingsSizingPost();
  const updateMut = useUpdateSizingConfigApiSettingsSizingConfigIdPut();
  const deleteMut = useDeleteSizingConfigApiSettingsSizingConfigIdDelete();

  const [showCreate, setShowCreate] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);

  const invalidate = () =>
    qc.invalidateQueries({
      queryKey: getListSizingConfigsApiSettingsSizingGetQueryKey(),
    });

  const configs = query.data?.data ?? [];

  if (query.isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (query.isError) {
    return (
      <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-destructive">
        <p className="font-medium">Failed to load sizing configs</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {configs.length} config{configs.length !== 1 ? "s" : ""}
        </p>
        {!showCreate && (
          <Button size="sm" onClick={() => setShowCreate(true)}>
            New Config
          </Button>
        )}
      </div>

      {showCreate && (
        <SizingForm
          initial={EMPTY_FORM}
          submitLabel="Create"
          isPending={createMut.isPending}
          onCancel={() => setShowCreate(false)}
          onSubmit={(form) => {
            createMut.mutate(
              {
                data: {
                  name: form.name,
                  method: form.method,
                  config: buildConfig(form),
                },
              },
              {
                onSuccess: () => {
                  invalidate();
                  setShowCreate(false);
                },
              },
            );
          }}
        />
      )}

      {configs.length === 0 && !showCreate && (
        <EmptyState
          title="No sizing configs"
          description="Create your first position sizing configuration."
          action={{ label: "New Config", onClick: () => setShowCreate(true) }}
        />
      )}

      {configs.map((cfg) =>
        editingId === cfg.id ? (
          <SizingForm
            key={cfg.id}
            initial={formFromConfig(cfg)}
            submitLabel="Save"
            isPending={updateMut.isPending}
            onCancel={() => setEditingId(null)}
            onSubmit={(form) => {
              updateMut.mutate(
                {
                  configId: cfg.id,
                  data: {
                    name: form.name,
                    method: form.method,
                    config: buildConfig(form),
                  },
                },
                {
                  onSuccess: () => {
                    invalidate();
                    setEditingId(null);
                  },
                },
              );
            }}
          />
        ) : (
          <Card key={cfg.id} className="py-4">
            <CardContent className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-foreground">{cfg.name}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {cfg.method} | Created {new Date(cfg.created_at).toLocaleDateString()}
                </p>
                <div className="mt-1 flex flex-wrap gap-2">
                  {Object.entries(cfg.config as Record<string, unknown>).map(([k, v]) => (
                    <Badge key={k} variant="secondary" className="text-[10px]">
                      {k}: {String(v)}
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="flex gap-1.5">
                <Button variant="outline" size="xs" onClick={() => setEditingId(cfg.id)}>
                  Edit
                </Button>
                <Button
                  variant="destructive"
                  size="xs"
                  onClick={() => {
                    deleteMut.mutate({ configId: cfg.id }, { onSuccess: invalidate });
                  }}
                >
                  Delete
                </Button>
              </div>
            </CardContent>
          </Card>
        ),
      )}
    </div>
  );
}
