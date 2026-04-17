import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { customInstance } from "./client";

export type SystemStatus = {
  kill_switch: boolean;
  reason: string | null;
  killed_at: string | null;
  killed_by: string | null;
  updated_at: string | null;
};

const QUERY_KEY = ["system", "status"] as const;

type ApiResponse<T> = { data: T; status: number };

export function useSystemStatus() {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: async () => {
      const res = await customInstance<ApiResponse<SystemStatus>>("/api/system/status");
      return res.data;
    },
    refetchInterval: 5000,
    refetchOnWindowFocus: true,
  });
}

export function useKillSystem() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: { reason: string; killed_by?: string }) => {
      const res = await customInstance<ApiResponse<SystemStatus>>("/api/system/kill", {
        method: "POST",
        body: JSON.stringify(body),
      });
      return res.data;
    },
    onSuccess: (data) => {
      qc.setQueryData(QUERY_KEY, data);
    },
  });
}

export function useUnlockSystem() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: { acknowledge: boolean; cleared_by?: string }) => {
      const res = await customInstance<ApiResponse<SystemStatus>>("/api/system/unlock", {
        method: "POST",
        body: JSON.stringify(body),
      });
      return res.data;
    },
    onSuccess: (data) => {
      qc.setQueryData(QUERY_KEY, data);
    },
  });
}
