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

const VALUE_ERROR_PREFIX = "Value error, ";

function stripValueErrorPrefix(msg: string): string {
  return msg.startsWith(VALUE_ERROR_PREFIX) ? msg.slice(VALUE_ERROR_PREFIX.length) : msg;
}

/**
 * Render one pydantic item as `<field>: <msg>`.
 *
 * `loc` is the path to the failing value. FastAPI wraps the request body under
 * a leading `"body"` segment, which carries no information for the caller, so
 * it is never used as the field label. A list item error (e.g.
 * `["body", "symbols", 0]`) therefore reads `symbols: ...` — the array index is
 * kept as a `[0]` suffix rather than shown as the field name. When no field
 * segment remains (`["body"]`, i.e. a model-level validator) only the message
 * is shown; the pydantic `"Value error, "` boilerplate is stripped either way.
 */
function formatPydanticItem(item: PydanticErrorItem): string {
  const msg = stripValueErrorPrefix(item.msg);
  const loc = Array.isArray(item.loc) ? item.loc : [];

  let fieldIndex = -1;
  for (let i = 0; i < loc.length; i += 1) {
    const segment = loc[i];
    if (typeof segment === "string" && segment !== "body") {
      fieldIndex = i;
    }
  }

  if (fieldIndex === -1) return msg;

  const indices = loc
    .slice(fieldIndex + 1)
    .filter((segment) => typeof segment === "number")
    .map((segment) => `[${String(segment)}]`)
    .join("");

  return `${String(loc[fieldIndex])}${indices}: ${msg}`;
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
    const parts = detail.filter(isPydanticErrorItem).map(formatPydanticItem);
    if (parts.length > 0) {
      return parts.join("; ");
    }
  }

  return "";
}
