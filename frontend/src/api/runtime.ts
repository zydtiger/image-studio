import { apiFetch } from "./client";
import type { RuntimeStatus } from "./types";

/** Resident model, worker state, current run, queue depth. */
export function getRuntime(signal?: AbortSignal): Promise<RuntimeStatus> {
  return apiFetch<RuntimeStatus>("/runtime", { signal });
}

/** Eject the idle worker; the server answers 409 while busy. */
export function ejectRuntime(signal?: AbortSignal): Promise<void> {
  return apiFetch<void>("/runtime/eject", { method: "POST", signal });
}
