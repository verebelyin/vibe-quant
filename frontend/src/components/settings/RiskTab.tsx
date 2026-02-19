import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { RiskConfigResponse } from "@/api/generated/models";
import {
  getListRiskConfigsApiSettingsRiskGetQueryKey,
  useCreateRiskConfigApiSettingsRiskPost,
  useDeleteRiskConfigApiSettingsRiskConfigIdDelete,
  useListRiskConfigsApiSettingsRiskGet,
  useUpdateRiskConfigApiSettingsRiskConfigIdPut,
} from "@/api/generated/settings/settings";
import { EmptyState } from "@/components/ui/EmptyState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

interface RiskFormState {
  name: string;
  max_position_size: string;
  max_drawdown_pct: string;
  stop_loss_pct: string;
  max_total_exposure: string;
  max_correlated_positions: string;
  daily_loss_limit_pct: string;
}

const EMPTY_FORM: RiskFormState = {
  name: "",
  max_position_size: "0.1",
  max_drawdown_pct: "0.15",
  stop_loss_pct: "0.05",
  max_total_exposure: "1.0",
  max_correlated_positions: "3",
  daily_loss_limit_pct: "0.05",
};

function formFromConfig(cfg: RiskConfigResponse): RiskFormState {
  const sl = cfg.strategy_level as Record<string, unknown>;
  const pl = cfg.portfolio_level as Record<string, unknown>;
  return {
    name: cfg.name,
    max_position_size: String(sl.max_position_size ?? "0.1"),
    max_drawdown_pct: String(sl.max_drawdown_pct ?? "0.15"),
    stop_loss_pct: String(sl.stop_loss_pct ?? "0.05"),
    max_total_exposure: String(pl.max_total_exposure ?? "1.0"),
    max_correlated_positions: String(pl.max_correlated_positions ?? "3"),
    daily_loss_limit_pct: String(pl.daily_loss_limit_pct ?? "0.05"),
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

function RiskForm({
  initial,
  onSubmit,
  onCancel,
  isPending,
  submitLabel,
}: {
  initial: RiskFormState;
  onSubmit: (form: RiskFormState) => void;
  onCancel: () => void;
  isPending: boolean;
  submitLabel: string;
}) {
  const [form, setForm] = useState<RiskFormState>(initial);
  const set = (k: keyof RiskFormState, v: string) => setForm((prev) => ({ ...prev, [k]: v }));

  return (
    <div
      className="rounded-lg border p-4"
      style={{
        backgroundColor: "hsl(var(--card))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <label className="mb-3 block">
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

      <p
        className="mb-2 text-xs font-semibold uppercase tracking-wider"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        Strategy Level
      </p>
      <div className="mb-4 grid gap-3 sm:grid-cols-3">
        <NumberField
          label="Max Position Size"
          value={form.max_position_size}
          onChange={(v) => set("max_position_size", v)}
        />
        <NumberField
          label="Max Drawdown %"
          value={form.max_drawdown_pct}
          onChange={(v) => set("max_drawdown_pct", v)}
        />
        <NumberField
          label="Stop Loss %"
          value={form.stop_loss_pct}
          onChange={(v) => set("stop_loss_pct", v)}
        />
      </div>

      <p
        className="mb-2 text-xs font-semibold uppercase tracking-wider"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        Portfolio Level
      </p>
      <div className="mb-4 grid gap-3 sm:grid-cols-3">
        <NumberField
          label="Max Total Exposure"
          value={form.max_total_exposure}
          onChange={(v) => set("max_total_exposure", v)}
        />
        <NumberField
          label="Max Correlated Positions"
          value={form.max_correlated_positions}
          onChange={(v) => set("max_correlated_positions", v)}
        />
        <NumberField
          label="Daily Loss Limit %"
          value={form.daily_loss_limit_pct}
          onChange={(v) => set("daily_loss_limit_pct", v)}
        />
      </div>

      <div className="flex gap-2">
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

function formToPayload(form: RiskFormState) {
  return {
    name: form.name,
    strategy_level: {
      max_position_size: Number(form.max_position_size),
      max_drawdown_pct: Number(form.max_drawdown_pct),
      stop_loss_pct: Number(form.stop_loss_pct),
    },
    portfolio_level: {
      max_total_exposure: Number(form.max_total_exposure),
      max_correlated_positions: Number(form.max_correlated_positions),
      daily_loss_limit_pct: Number(form.daily_loss_limit_pct),
    },
  };
}

export function RiskTab() {
  const qc = useQueryClient();
  const query = useListRiskConfigsApiSettingsRiskGet();
  const createMut = useCreateRiskConfigApiSettingsRiskPost();
  const updateMut = useUpdateRiskConfigApiSettingsRiskConfigIdPut();
  const deleteMut = useDeleteRiskConfigApiSettingsRiskConfigIdDelete();

  const [showCreate, setShowCreate] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);

  const invalidate = () =>
    qc.invalidateQueries({
      queryKey: getListRiskConfigsApiSettingsRiskGetQueryKey(),
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
        <p className="font-medium">Failed to load risk configs</p>
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
        <RiskForm
          initial={EMPTY_FORM}
          submitLabel="Create"
          isPending={createMut.isPending}
          onCancel={() => setShowCreate(false)}
          onSubmit={(form) => {
            createMut.mutate(
              { data: formToPayload(form) },
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
          title="No risk configs"
          description="Create your first risk management configuration."
          action={{ label: "New Config", onClick: () => setShowCreate(true) }}
        />
      )}

      {configs.map((cfg) =>
        editingId === cfg.id ? (
          <RiskForm
            key={cfg.id}
            initial={formFromConfig(cfg)}
            submitLabel="Save"
            isPending={updateMut.isPending}
            onCancel={() => setEditingId(null)}
            onSubmit={(form) => {
              updateMut.mutate(
                { configId: cfg.id, data: formToPayload(form) },
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
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold" style={{ color: "hsl(var(--foreground))" }}>
                {cfg.name}
              </p>
              <p className="mt-0.5 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                Created {new Date(cfg.created_at).toLocaleDateString()}
              </p>
              <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
                <span className="text-[10px]" style={{ color: "hsl(var(--muted-foreground))" }}>
                  Strategy:{" "}
                  {Object.entries(cfg.strategy_level as Record<string, unknown>)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(", ")}
                </span>
                <span className="text-[10px]" style={{ color: "hsl(var(--muted-foreground))" }}>
                  Portfolio:{" "}
                  {Object.entries(cfg.portfolio_level as Record<string, unknown>)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(", ")}
                </span>
              </div>
            </div>
            <div className="ml-4 flex shrink-0 gap-1.5">
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
