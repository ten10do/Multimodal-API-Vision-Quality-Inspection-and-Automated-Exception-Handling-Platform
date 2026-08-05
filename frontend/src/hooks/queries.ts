import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function useRealtimeStatus(refreshMs = 3000) {
  return useQuery({
    queryKey: ["realtime-status"],
    queryFn: api.realtimeStatus,
    refetchInterval: refreshMs,
    retry: 2,
  });
}

export function useRecentInspections(limit = 300, refreshMs = 5000) {
  return useQuery({
    queryKey: ["inspections", "recent", limit],
    queryFn: () => api.listInspections({ limit }),
    refetchInterval: refreshMs,
    retry: 2,
  });
}

export function useInspection(id: string | null) {
  return useQuery({
    queryKey: ["inspection", id],
    queryFn: () => api.getInspection(id as string),
    enabled: id !== null,
    retry: 1,
  });
}

export function useProductHistory(productId: string | null) {
  return useQuery({
    queryKey: ["product", productId, "history"],
    queryFn: () => api.productInspections(productId as string),
    enabled: productId !== null,
    retry: 1,
  });
}
