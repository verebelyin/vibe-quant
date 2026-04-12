import { useQuery } from "@tanstack/react-query";
import { customInstance } from "@/api/client";

export interface ApiIndicatorEntry {
  type_name: string;
  display_name: string;
  description: string;
  category: string;
  popular: boolean;
  chart_placement: string;
  default_params: Record<string, number>;
  param_schema: Record<string, string>;
  output_names: string[];
  requires_high_low: boolean;
  requires_volume: boolean;
}

interface CatalogResponse {
  data: {
    indicators: ApiIndicatorEntry[];
    categories: string[];
  };
  status: number;
}

/**
 * Fetches the indicator catalog from GET /api/indicators/catalog.
 * staleTime=Infinity: the catalog doesn't change during a session
 * (plugins are loaded once at startup).
 */
export function useIndicatorCatalog() {
  return useQuery({
    queryKey: ["indicators", "catalog"],
    queryFn: () =>
      customInstance<CatalogResponse>("/api/indicators/catalog", {
        method: "GET",
      }),
    staleTime: Infinity,
  });
}
