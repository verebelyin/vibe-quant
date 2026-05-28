const BASE_URL = ""; // Proxy handles /api prefix

/**
 * Error thrown by {@link customInstance} on any non-2xx response. Carries the
 * numeric HTTP status and the parsed response body so callers can render
 * structured failure data (e.g. scaffold's suggested_name / test_output)
 * instead of just the bare status line in `message`.
 */
export interface ApiError extends Error {
  status: number;
  body: unknown;
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof Error && "status" in error && "body" in error;
}

/**
 * Custom fetch instance for orval 8 generated hooks.
 * Orval 8 calls this as customInstance<T>(url, init).
 */
export const customInstance = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(`${BASE_URL}${url}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    // Preserve the response body (FastAPI's {"detail": ...}) on the thrown
    // Error so onError handlers can recover structured failure data.
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = undefined;
    }
    const err = new Error(
      `API error: ${response.status} ${response.statusText}`,
    ) as ApiError;
    err.status = response.status;
    err.body = body;
    throw err;
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return { data: undefined, status: 204 } as T;
  }

  const data = await response.json();
  return { data, status: response.status } as T;
};

export default customInstance;
