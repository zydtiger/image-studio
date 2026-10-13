import { apiFetch } from "./client";
import type { InputImage } from "./types";

export function uploadInput(
  file: File,
  signal?: AbortSignal,
): Promise<InputImage> {
  const body = new FormData();
  body.append("file", file);
  return apiFetch("/inputs", { method: "POST", body, signal });
}

export function inputFromArtifact(
  runId: string,
  artifactId: string,
): Promise<InputImage> {
  return apiFetch("/inputs/from-artifact", {
    method: "POST",
    body: { run_id: runId, artifact_id: artifactId },
  });
}
