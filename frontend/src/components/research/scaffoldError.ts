import { isApiError } from "@/api/client";

/** Structured failure payload the scaffold endpoint puts in the HTTP error
 *  body: `raise HTTPException(status, detail={status, suggested_name, ...})`
 *  → customInstance attaches `body = {detail: {...}}` to the thrown Error. */
export interface ScaffoldErrorDetail {
  status?: string;
  name?: string | null;
  suggested_name?: string | null;
  error?: string | null;
  test_output?: string | null;
}

/** Narrow a thrown mutation error into its HTTP status + scaffold detail.
 *  Returns status 0 and an empty detail when the error isn't a recognised
 *  ApiError (e.g. a network failure with no JSON body). */
export function scaffoldErrorDetail(error: unknown): {
  status: number;
  detail: ScaffoldErrorDetail;
} {
  if (!isApiError(error)) return { status: 0, detail: {} };
  const body = error.body;
  if (typeof body === "object" && body !== null && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "object" && d !== null) {
      return { status: error.status, detail: d as ScaffoldErrorDetail };
    }
  }
  return { status: error.status, detail: {} };
}
