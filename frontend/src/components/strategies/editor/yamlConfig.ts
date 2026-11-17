/** YAML text <-> editor model, shared by the YAML tab and its tests. */
import * as yaml from "js-yaml";
import { DIRECTIONAL_RISK_KEYS } from "./canonical";
import type { DslConfig } from "./types";
import { parseDslConfig } from "./types";

export interface YamlError {
  message: string;
  line?: number | undefined;
}

export function serializeConfig(config: DslConfig): string {
  return yaml.dump(config, { indent: 2, lineWidth: 120, noRefs: true });
}

/** Reject a mistyped port field: a dropped-by-reader value would silently delete the key. */
function portFieldError(raw: Record<string, unknown>): string | null {
  const { position_mode: mode, refresh_on_signal: refresh } = raw;
  if (mode !== undefined && mode !== null && mode !== "reverse")
    return "position_mode must be 'reverse' or absent";
  if (refresh !== undefined && refresh !== null && typeof refresh !== "boolean")
    return "refresh_on_signal must be true or false";
  for (const key of DIRECTIONAL_RISK_KEYS) {
    const leg = raw[key];
    if (leg === undefined || leg === null) continue;
    const legError = riskLegError(key, leg);
    if (legError) return legError;
  }
  return null;
}

const LEG_NUMBER_FIELDS = ["percent", "atr_multiplier", "risk_reward_ratio"];

/** A leg the reader would shrink (wrong type, unknown field) is an error, not a silent drop. */
function riskLegError(key: string, leg: unknown): string | null {
  if (typeof leg !== "object" || leg === null || Array.isArray(leg))
    return `${key} must be a mapping`;
  const fields = leg as Record<string, unknown>;
  if (typeof fields.type !== "string") return `${key}.type must be a string`;
  for (const [field, value] of Object.entries(fields)) {
    if (field === "type" || value === null) continue;
    if (field === "indicator") {
      if (typeof value !== "string") return `${key}.indicator must be a string`;
    } else if (LEG_NUMBER_FIELDS.includes(field)) {
      if (typeof value !== "number") return `${key}.${field} must be a number`;
    } else return `${key} has unknown field '${field}'`;
  }
  return null;
}

export function parseYamlToDsl(text: string): { config: DslConfig; error: YamlError | null } {
  try {
    // js-yaml 5 throws on empty input; keep the "must be a mapping" message for it.
    const parsed = text.trim() === "" ? null : yaml.load(text);
    if (parsed === null || parsed === undefined || typeof parsed !== "object") {
      return {
        config: {} as DslConfig,
        error: { message: "YAML must be a mapping (object), not a scalar or array" },
      };
    }
    const raw = parsed as Record<string, unknown>;
    const portError = portFieldError(raw);
    if (portError) return { config: {} as DslConfig, error: { message: portError } };
    const config = parseDslConfig(raw);
    return { config, error: null };
  } catch (e: unknown) {
    if (e instanceof yaml.YAMLException) {
      return {
        config: {} as DslConfig,
        error: {
          message: e.message,
          line: e.mark?.line !== undefined ? e.mark.line + 1 : undefined,
        },
      };
    }
    const msg = e instanceof Error ? e.message : "Unknown parse error";
    return { config: {} as DslConfig, error: { message: msg } };
  }
}
