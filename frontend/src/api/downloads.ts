import { apiFetch } from "./client";
import type { DownloadCreate, DownloadJob } from "./types";

export function listDownloads(signal?: AbortSignal): Promise<DownloadJob[]> {
  return apiFetch<{ jobs: DownloadJob[] }>("/downloads", { signal }).then(
    (response) => response.jobs,
  );
}

/** Enqueue a download; the revision resolves to a commit before transfer. */
export function createDownload(
  body: DownloadCreate,
  signal?: AbortSignal,
): Promise<DownloadJob> {
  return apiFetch<DownloadJob>("/downloads", { method: "POST", body, signal });
}

/** Retry a failed or cancelled job; reuses already cached files. */
export function retryDownload(
  id: string,
  signal?: AbortSignal,
): Promise<DownloadJob> {
  return apiFetch<DownloadJob>(`/downloads/${encodeURIComponent(id)}/retry`, {
    method: "POST",
    signal,
  });
}

/** Cancel a queued job; a running job answers 409. */
export function cancelDownload(
  id: string,
  signal?: AbortSignal,
): Promise<DownloadJob> {
  return apiFetch<DownloadJob>(`/downloads/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
    signal,
  });
}
