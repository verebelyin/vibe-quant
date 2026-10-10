import { isApiError } from "@/api/client";

interface PydanticErrorItem {
  msg: string;
  loc?: Array<string | number>;
}

function isPydanticErrorItem(value: unknown): value is PydanticErrorItem {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { msg?: unknown }).msg === "string"
  );
}

/**
 * Extract a human-readable message from an error thrown by the API client.
 *
 * `customInstance` throws an {@link ApiError} whose `body` is FastAPI's parsed
 * response body. The `detail` may be a string (our own config check) or a
 * pydantic list of `{loc, msg, type}` items. Returns an empty string when no
 * detail can be recovered, so callers can fall back to a generic message.
 */
export function formatApiErrorDetail(err: unknown): string {
  if (!isApiError(err)) return "";
  const body = err.body as { detail?: unknown } | undefined;
  const detail = body?.detail;

  if (typeof detail === "string" && detail.trim() !== "") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const parts = detail.filter(isPydanticErrorItem).map((item) => {
      const loc = item.loc;
      if (Array.isArray(loc) && loc.length > 0) {
        return `${String(loc[loc.length - 1])}: ${item.msg}`;
      }
      return item.msg;
    });
    if (parts.length > 0) {
      return parts.join("; ");
    }
  }

  return "";
}
