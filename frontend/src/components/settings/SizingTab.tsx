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
import { EmptyState } from "@/components/ui/EmptyState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

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
    <label className="block">
      <span
        className="mb-1 block text-xs font-medium"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        {label}
      </span>
      <input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border px-3 py-1.5 text-sm outline-none transition-colors focus:ring-1"
        style={{
          backgroundColor: "hsl(var(--background))",
          borderColor: "hsl(var(--border))",
          color: "hsl(var(--foreground))",
        }}
      />
    </label>
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
    <div
      className="rounded-lg border p-4"
      style={{
        backgroundColor: "hsl(var(--card))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block sm:col-span-2">
          <span
            className="mb-1 block text-xs font-medium"
            style={{ color: "hsl(var(--muted-foreground))" }}
          >
            Name
          </span>
          <input
            type="text"
            value={form.name}
            onChange={(e) => set("name", e.target.value)}
            className="w-full rounded-md border px-3 py-1.5 text-sm outline-none transition-colors focus:ring-1"
            style={{
              backgroundColor: "hsl(var(--background))",
              borderColor: "hsl(var(--border))",
              color: "hsl(var(--foreground))",
            }}
          />
        </label>

        <label className="block sm:col-span-2">
          <span
            className="mb-1 block text-xs font-medium"
            style={{ color: "hsl(var(--muted-foreground))" }}
          >
            Method
          </span>
          <select
            value={form.method}
            onChange={(e) => set("method", e.target.value)}
            className="w-full rounded-md border px-3 py-1.5 text-sm outline-none"
            style={{
              backgroundColor: "hsl(var(--background))",
              borderColor: "hsl(var(--border))",
              color: "hsl(var(--foreground))",
            }}
          >
            <option value="fixed_fractional">Fixed Fractional</option>
            <option value="kelly">Kelly Criterion</option>
            <option value="atr">ATR-based</option>
          </select>
        </label>

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
            <NumberField label="Avg Win" value={form.avg_win} onChange={(v) => set("avg_win", v)} />
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
        <button
          type="button"
          disabled={isPending || !form.name.trim()}
          onClick={() => onSubmit(form)}
          className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors hover:brightness-90 disabled:opacity-50"
          style={{
            backgroundColor: "hsl(var(--primary))",
            color: "hsl(var(--primary-foreground))",
          }}
        >
          {isPending ? "Saving..." : submitLabel}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors hover:brightness-90"
          style={{
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--foreground))",
          }}
        >
          Cancel
        </button>
      </div>
    </div>
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
      <div
        className="rounded-lg border p-4"
        style={{
          borderColor: "hsl(0 84% 60%)",
          backgroundColor: "hsl(0 84% 60% / 0.1)",
          color: "hsl(0 84% 60%)",
        }}
      >
        <p className="font-medium">Failed to load sizing configs</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>
          {configs.length} config{configs.length !== 1 ? "s" : ""}
        </p>
        {!showCreate && (
          <button
            type="button"
            onClick={() => setShowCreate(true)}
            className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors hover:brightness-90"
            style={{
              backgroundColor: "hsl(var(--primary))",
              color: "hsl(var(--primary-foreground))",
            }}
          >
            New Config
          </button>
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
          <div
            key={cfg.id}
            className="flex items-center justify-between rounded-lg border p-4"
            style={{
              backgroundColor: "hsl(var(--card))",
              borderColor: "hsl(var(--border))",
            }}
          >
            <div>
              <p className="text-sm font-semibold" style={{ color: "hsl(var(--foreground))" }}>
                {cfg.name}
              </p>
              <p className="mt-0.5 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                {cfg.method} | Created {new Date(cfg.created_at).toLocaleDateString()}
              </p>
              <div className="mt-1 flex flex-wrap gap-2">
                {Object.entries(cfg.config as Record<string, unknown>).map(([k, v]) => (
                  <span
                    key={k}
                    className="rounded px-1.5 py-0.5 text-[10px]"
                    style={{
                      backgroundColor: "hsl(var(--muted))",
                      color: "hsl(var(--muted-foreground))",
                    }}
                  >
                    {k}: {String(v)}
                  </span>
                ))}
              </div>
            </div>
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={() => setEditingId(cfg.id)}
                className="rounded border px-2 py-1 text-[10px] font-medium transition-colors hover:brightness-90"
                style={{
                  borderColor: "hsl(var(--border))",
                  color: "hsl(var(--foreground))",
                }}
              >
                Edit
              </button>
              <button
                type="button"
                onClick={() => {
                  deleteMut.mutate({ configId: cfg.id }, { onSuccess: invalidate });
                }}
                className="rounded border px-2 py-1 text-[10px] font-medium transition-colors hover:brightness-90"
                style={{
                  borderColor: "hsl(0 84% 60% / 0.3)",
                  color: "hsl(0 84% 60%)",
                }}
              >
                Delete
              </button>
            </div>
          </div>
        ),
      )}
    </div>
  );
}
