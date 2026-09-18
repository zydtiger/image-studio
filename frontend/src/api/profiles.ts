import { apiFetch } from "./client";
import type { ProfileSpec } from "./types";

/** Capability metadata: defaults, bounds, and fixed/hidden fields. */
export function getProfiles(signal?: AbortSignal): Promise<ProfileSpec[]> {
  return apiFetch<{ profiles: ProfileSpec[] }>("/profiles", { signal }).then(
    (response) => response.profiles,
  );
}
