import { apiFetch } from "./client";
import type {
  ModelRegistration,
  RegistrationCreate,
  RegistrationUpdate,
} from "./types";

export function listRegistrations(
  signal?: AbortSignal,
): Promise<ModelRegistration[]> {
  return apiFetch<{ registrations: ModelRegistration[] }>("/models", {
    signal,
  }).then((response) => response.registrations);
}

/** Register an existing snapshot; never downloads. 422 cache_incomplete. */
export function createRegistration(
  body: RegistrationCreate,
  signal?: AbortSignal,
): Promise<ModelRegistration> {
  return apiFetch<ModelRegistration>("/models", {
    method: "POST",
    body,
    signal,
  });
}

/** Profile changes conflict (409) while resident or referenced. */
export function updateRegistration(
  id: string,
  body: RegistrationUpdate,
  signal?: AbortSignal,
): Promise<ModelRegistration> {
  return apiFetch<ModelRegistration>(`/models/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body,
    signal,
  });
}

/** Removes the registration only; shared cache files are never deleted. */
export function deleteRegistration(
  id: string,
  signal?: AbortSignal,
): Promise<void> {
  return apiFetch<void>(`/models/${encodeURIComponent(id)}`, {
    method: "DELETE",
    signal,
  });
}
