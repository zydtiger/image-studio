import { apiFetch } from "./client";
import type { CachedRepo } from "./types";

/** All cached repositories, including unsupported and incomplete ones. */
export function listCachedRepos(signal?: AbortSignal): Promise<CachedRepo[]> {
  return apiFetch<{ repos: CachedRepo[] }>("/cache/models", { signal }).then(
    (response) => response.repos,
  );
}
