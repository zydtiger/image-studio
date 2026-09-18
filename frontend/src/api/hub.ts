import { apiFetch } from "./client";
import type {
  CompatibilityReport,
  HubModelDetail,
  HubModelSummary,
} from "./types";

export function searchHubModels(
  query: string,
  limit = 20,
  signal?: AbortSignal,
): Promise<HubModelSummary[]> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return apiFetch<{ results: HubModelSummary[] }>(
    `/hub/models?${params.toString()}`,
    { signal },
  ).then((response) => response.results);
}

export function getHubModelDetail(
  repoId: string,
  signal?: AbortSignal,
): Promise<HubModelDetail> {
  return apiFetch<HubModelDetail>(`/hub/models/${encodeURIComponent(repoId)}`, {
    signal,
  });
}

export function getCompatibility(
  repoId: string,
  revision?: string | null,
  signal?: AbortSignal,
): Promise<CompatibilityReport> {
  const params = new URLSearchParams();
  if (revision) params.set("revision", revision);
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return apiFetch<CompatibilityReport>(
    `/hub/models/${encodeURIComponent(repoId)}/compatibility${suffix}`,
    { signal },
  );
}
