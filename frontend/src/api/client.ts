const BASE_URL = ""; // Proxy handles /api prefix

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
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
};

export default customInstance;
