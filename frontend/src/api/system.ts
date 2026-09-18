import { apiFetch } from "./client";
import type { SystemInfo } from "./types";

/** Effective configuration, GPUs, HF login state, development flags. */
export function getSystem(signal?: AbortSignal): Promise<SystemInfo> {
  return apiFetch<SystemInfo>("/system", { signal });
}
