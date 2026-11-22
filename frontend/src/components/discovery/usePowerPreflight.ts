/**
 * The launch form's power preflight (bead vibe-quant-clmmc.1): which Sharpe the run
 * could certify at all, asked from POST /api/discovery/preflight with the exact body
 * the launch would send. Reporting only: it never blocks a launch.
 */
import { useEffect, useState } from "react";
import { preflightDiscoveryApiDiscoveryPreflightPost } from "@/api/generated/discovery/discovery";
import type {
  DiscoveryLaunchRequest,
  DiscoveryPreflightResponse,
  FullHistoryStartApiDiscoveryFullHistoryStartGetParams,
} from "@/api/generated/models";
import { formatApiErrorDetail } from "./apiErrorDetail";

/** Wait this long after the last form edit before asking (one request per pause). */
const DEBOUNCE_MS = 400;

export type PowerPreflightState =
  | { kind: "idle" }
  | { kind: "ok"; data: DiscoveryPreflightResponse; body: DiscoveryLaunchRequest }
  | { kind: "error"; message: string };

/** A settled answer and the body (JSON key) it answers. */
export interface SettledPreflight {
  key: string;
  result: PowerPreflightState;
}

/**
 * What the form shows for the body `key`: a settled answer only when it answers this very
 * body; after an edit the older answer is hidden until the new one arrives (a launch in
 * between sends the new body, so the old bars would describe a run that is not launched).
 */
export function visiblePreflight(
  settled: SettledPreflight | null,
  key: string,
): PowerPreflightState {
  if (key === "" || settled === null || settled.key !== key) return { kind: "idle" };
  return settled.result;
}

/**
 * The preflight of `body` (null = the form cannot build a body yet), refreshed on every
 * change after a short pause; an answer to an older body is never shown.
 */
export function usePowerPreflight(body: DiscoveryLaunchRequest | null): PowerPreflightState {
  const [settled, setSettled] = useState<SettledPreflight | null>(null);
  const key = body === null ? "" : JSON.stringify(body);

  useEffect(() => {
    if (key === "") return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      const sent = JSON.parse(key) as DiscoveryLaunchRequest;
      preflightDiscoveryApiDiscoveryPreflightPost(sent, { signal: controller.signal })
        .then((resp) => {
          if (controller.signal.aborted) return;
          if (resp.status === 200) {
            setSettled({ key, result: { kind: "ok", data: resp.data, body: sent } });
          }
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const message = formatApiErrorDetail(err) || "Preflight failed";
          setSettled({ key, result: { kind: "error", message } });
        });
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key]);

  return visiblePreflight(settled, key);
}

/**
 * The "Use full history" query for `body`: its symbols and whether MARK coverage clamps the
 * start (liquidation on, or a bankroll grid, which is always liquidation-on). Never the
 * dates, so a range the preflight refuses can still be widened. Null = nothing to ask.
 */
export function fullHistoryParams(
  body: DiscoveryLaunchRequest | null,
): FullHistoryStartApiDiscoveryFullHistoryStartGetParams | null {
  const symbols = body?.symbols ?? [];
  if (body === null || symbols.length === 0) return null;
  const mark = (body.liquidation_mode ?? "off") !== "off" || body.bankroll_grid != null;
  return { symbols: symbols.join(","), mark };
}
