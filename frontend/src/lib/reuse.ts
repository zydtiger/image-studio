import type { RunDetail } from "../api/types";

/**
 * Payload carried through router state when reusing a historical run's
 * parameters on the Generate page.
 */
export interface ReusePayload {
  registrationId: string | null;
  repoId: string;
  prompt: string;
  negativePrompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number | null;
  seed: number;
  count: number;
  gpuUuid: string | null;
  gpuName: string | null;
  profile: RunDetail["profile"];
}

export function buildReusePayload(run: RunDetail): ReusePayload {
  return {
    registrationId: run.registration_id,
    repoId: run.repo_id,
    prompt: run.prompt,
    negativePrompt: run.negative_prompt ?? null,
    width: run.width,
    height: run.height,
    steps: run.steps,
    guidance: run.guidance,
    seed: run.initial_seed,
    count: run.image_count,
    gpuUuid: run.gpu?.uuid ?? null,
    gpuName: run.gpu?.name ?? null,
    profile: run.profile,
  };
}
