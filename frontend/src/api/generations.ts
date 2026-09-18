import { apiFetch } from "./client";
import type {
  FavoriteUpdate,
  GenerationRequest,
  QueueState,
  RunDetail,
  RunStatus,
  RunSummary,
} from "./types";

export interface RunListParams {
  q?: string;
  status?: RunStatus;
  model?: string;
  favorite?: boolean;
  /** "only" selects the Trash view; otherwise non-trashed runs. */
  trashed?: "only";
  limit?: number;
  offset?: number;
}

export interface RunListResult {
  runs: RunSummary[];
  total: number;
}

export function listRuns(
  params: RunListParams,
  signal?: AbortSignal,
): Promise<RunListResult> {
  const search = new URLSearchParams();
  if (params.q) search.set("q", params.q);
  if (params.status) search.set("status", params.status);
  if (params.model) search.set("model", params.model);
  if (params.favorite) search.set("favorite", "true");
  if (params.trashed) search.set("trashed", params.trashed);
  search.set("limit", String(params.limit ?? 24));
  search.set("offset", String(params.offset ?? 0));
  return apiFetch<RunListResult>(`/generations?${search.toString()}`, {
    signal,
  });
}

export function getQueue(signal?: AbortSignal): Promise<QueueState> {
  return apiFetch<QueueState>("/generations/queue", { signal });
}

/** Resumes the paused queue in original FIFO order. */
export function resumeQueue(signal?: AbortSignal): Promise<QueueState> {
  return apiFetch<QueueState>("/generations/queue/resume", {
    method: "POST",
    signal,
  });
}

export function getRun(
  runId: string,
  signal?: AbortSignal,
): Promise<RunDetail> {
  return apiFetch<RunDetail>(`/generations/${encodeURIComponent(runId)}`, {
    signal,
  });
}

/** Submission freezes all settings and returns the frozen run. */
export function submitGeneration(
  body: GenerationRequest,
  signal?: AbortSignal,
): Promise<RunDetail> {
  return apiFetch<RunDetail>("/generations", { method: "POST", body, signal });
}

/** Queued runs cancel immediately; running runs request cooperative cancel. */
export function cancelRun(
  runId: string,
  signal?: AbortSignal,
): Promise<RunDetail> {
  return apiFetch<RunDetail>(
    `/generations/${encodeURIComponent(runId)}/cancel`,
    { method: "POST", signal },
  );
}

export function setFavorite(
  runId: string,
  favorite: boolean,
  signal?: AbortSignal,
): Promise<RunSummary> {
  const body: FavoriteUpdate = { favorite };
  return apiFetch<RunSummary>(`/generations/${encodeURIComponent(runId)}`, {
    method: "PATCH",
    body,
    signal,
  });
}

/** Recoverable deletion; records and files are retained. */
export function trashRun(
  runId: string,
  signal?: AbortSignal,
): Promise<RunSummary> {
  return apiFetch<RunSummary>(
    `/generations/${encodeURIComponent(runId)}/trash`,
    { method: "POST", signal },
  );
}

export function restoreRun(
  runId: string,
  signal?: AbortSignal,
): Promise<RunSummary> {
  return apiFetch<RunSummary>(
    `/generations/${encodeURIComponent(runId)}/restore`,
    { method: "POST", signal },
  );
}

export function artifactUrl(
  runId: string,
  artifactId: string,
  download = false,
): string {
  const base = `/api/generations/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}`;
  return download ? `${base}?download=1` : base;
}

export function thumbnailUrl(runId: string, artifactId: string): string {
  return `/api/generations/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}/thumbnail`;
}

/** Serves the run's metadata.json as a download. */
export function metadataUrl(runId: string): string {
  return `/api/generations/${encodeURIComponent(runId)}/metadata`;
}
