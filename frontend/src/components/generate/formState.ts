import type {
  GenerationRequest,
  GpuInfo,
  ModelRegistration,
  ProfileSpec,
} from "../../api/types";
import type { ReusePayload } from "../../lib/reuse";
import {
  parseSeed,
  validateDimension,
  validateImageCount,
  validateSeedWindow,
} from "../../lib/validation";

/**
 * Generation form state and its pure transformations. All capability
 * rules come from the API's ProfileSpec; nothing here duplicates the
 * backend's defaults table.
 */

export interface GenerationFormState {
  registrationId: string;
  gpuUuid: string;
  prompt: string;
  negativePrompt: string;
  width: string;
  height: string;
  steps: string;
  guidance: string;
  seed: string;
  count: string;
}

export const EMPTY_FORM: GenerationFormState = {
  registrationId: "",
  gpuUuid: "",
  prompt: "",
  negativePrompt: "",
  width: "",
  height: "",
  steps: "",
  guidance: "",
  seed: "",
  count: "1",
};

export type FormErrors = Partial<Record<keyof GenerationFormState, string>>;

export interface ParsedFormValues {
  registrationId: string;
  gpuUuid: string;
  prompt: string;
  negativePrompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number | null;
  seed: number | null;
  count: number;
}

/** Applies profile defaults to capability-dependent fields. */
export function applyProfileDefaults(
  state: GenerationFormState,
  spec: ProfileSpec,
): GenerationFormState {
  return {
    ...state,
    steps: String(spec.default_steps),
    guidance: spec.guidance_fixed === null ? String(spec.guidance_default) : "",
    width: String(spec.default_width),
    height: String(spec.default_height),
    negativePrompt: spec.negative_prompt_supported ? state.negativePrompt : "",
  };
}

/**
 * Prefills the form from a historical run. Profile-dependent fields are
 * only applied when the run's registration still resolves; an unavailable
 * historical GPU forces an explicit new selection.
 */
export function applyReuse(
  state: GenerationFormState,
  reuse: ReusePayload,
  registrations: ModelRegistration[],
  profiles: ProfileSpec[],
  gpus: GpuInfo[],
): { state: GenerationFormState; notices: string[]; gpuNeedsChoice: boolean } {
  const notices: string[] = [];
  let next: GenerationFormState = {
    ...state,
    prompt: reuse.prompt,
    width: String(reuse.width),
    height: String(reuse.height),
    seed: String(reuse.seed),
    count: String(reuse.count),
    negativePrompt: "",
  };

  const registration = registrations.find(
    (entry) => entry.id === reuse.registrationId,
  );
  // The registration's profile is the actual target; the run's recorded
  // profile is only a fallback when it still matches.
  const spec =
    profiles.find((entry) => entry.profile_id === registration?.profile) ??
    (registration !== undefined && registration.profile === reuse.profile
      ? profiles.find((entry) => entry.profile_id === reuse.profile)
      : undefined);

  if (registration === undefined || spec === undefined) {
    notices.push(
      `The model used by this run (${reuse.repoId}) is not registered anymore. Choose a model.`,
    );
  } else {
    next = { ...next, registrationId: registration.id };
    next.steps = String(
      Math.min(spec.max_steps, Math.max(spec.min_steps, reuse.steps)),
    );
    if (spec.guidance_fixed === null && reuse.guidance !== null) {
      next.guidance = String(reuse.guidance);
    }
    if (spec.negative_prompt_supported && reuse.negativePrompt !== null) {
      next.negativePrompt = reuse.negativePrompt;
    }
  }

  let gpuNeedsChoice = false;
  const gpu = gpus.find((entry) => entry.uuid === reuse.gpuUuid);
  if (gpu !== undefined) {
    next.gpuUuid = gpu.uuid;
  } else {
    // The historical GPU is gone: clear whatever default is present so
    // submission is blocked until an explicit new selection — never a
    // silent fallback.
    next.gpuUuid = "";
    gpuNeedsChoice = true;
    notices.push(
      `The GPU used by this run (${reuse.gpuName ?? reuse.gpuUuid ?? "unknown"}) is not available. Choose a GPU.`,
    );
  }

  return { state: next, notices, gpuNeedsChoice };
}

function parsePositiveInt(
  raw: string,
  label: string,
): { value?: number; error?: string } {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return { error: `Enter ${label}.` };
  }
  if (!/^\d+$/.test(trimmed)) {
    return { error: `${label} must be a whole number.` };
  }
  return { value: Number(trimmed) };
}

/** Validates the form against the selected profile's capability spec. */
export function validateForm(
  state: GenerationFormState,
  spec: ProfileSpec | undefined,
): { values: ParsedFormValues; errors: FormErrors } {
  const errors: FormErrors = {};

  if (state.registrationId === "") {
    errors.registrationId = "Select a model.";
  }
  if (state.gpuUuid === "") {
    errors.gpuUuid = "Select a GPU.";
  }
  if (state.prompt.trim() === "") {
    errors.prompt = "Enter a prompt.";
  }

  const width = parsePositiveInt(state.width, "a width");
  if (width.error !== undefined || width.value === undefined) {
    errors.width = width.error;
  } else {
    const check = validateDimension(
      width.value,
      "Width",
      spec?.dimension_multiple,
    );
    if (!check.ok) errors.width = check.message;
  }
  const height = parsePositiveInt(state.height, "a height");
  if (height.error !== undefined || height.value === undefined) {
    errors.height = height.error;
  } else {
    const check = validateDimension(
      height.value,
      "Height",
      spec?.dimension_multiple,
    );
    if (!check.ok) errors.height = check.message;
  }

  const count = parsePositiveInt(state.count, "an image count");
  let countValue = 0;
  if (count.error !== undefined || count.value === undefined) {
    errors.count = count.error;
  } else {
    const check = validateImageCount(count.value);
    if (!check.ok) errors.count = check.message;
    else countValue = count.value;
  }

  let stepsValue = 0;
  if (spec === undefined) {
    errors.steps = "Select a model first.";
  } else {
    const steps = parsePositiveInt(state.steps, "steps");
    if (steps.error !== undefined || steps.value === undefined) {
      errors.steps = steps.error;
    } else if (steps.value < spec.min_steps || steps.value > spec.max_steps) {
      errors.steps = `Steps must be ${spec.min_steps}-${spec.max_steps} for ${spec.label}.`;
    } else {
      stepsValue = steps.value;
    }
  }

  let guidanceValue: number | null = null;
  if (spec !== undefined && spec.guidance_fixed === null) {
    const trimmed = state.guidance.trim();
    if (trimmed === "") {
      errors.guidance = "Enter guidance.";
    } else {
      const parsed = Number(trimmed);
      if (!Number.isFinite(parsed) || parsed < 0) {
        errors.guidance = "Guidance must be zero or greater.";
      } else {
        guidanceValue = parsed;
      }
    }
  }

  let seedValue: number | null = null;
  const seed = parseSeed(state.seed);
  if (seed.kind === "invalid") {
    errors.seed = seed.message;
  } else if (seed.kind === "value") {
    if (countValue > 0) {
      const check = validateSeedWindow(seed.value, countValue);
      if (!check.ok) errors.seed = check.message;
      else seedValue = seed.value;
    } else {
      seedValue = seed.value;
    }
  }

  const widthOk = errors.width === undefined && width.value !== undefined;
  const heightOk = errors.height === undefined && height.value !== undefined;

  return {
    values: {
      registrationId: state.registrationId,
      gpuUuid: state.gpuUuid,
      prompt: state.prompt,
      negativePrompt:
        spec?.negative_prompt_supported === true &&
        state.negativePrompt.trim() !== ""
          ? state.negativePrompt
          : null,
      width: widthOk ? (width.value as number) : 0,
      height: heightOk ? (height.value as number) : 0,
      steps: stepsValue,
      guidance: guidanceValue,
      seed: seedValue,
      count: countValue,
    },
    errors,
  };
}

/** Builds the frozen submission body; capability-fixed fields are omitted. */
export function buildRequest(
  values: ParsedFormValues,
  spec: ProfileSpec,
): GenerationRequest {
  return {
    registration_id: values.registrationId,
    gpu_uuid: values.gpuUuid,
    prompt: values.prompt,
    negative_prompt: values.negativePrompt,
    width: values.width,
    height: values.height,
    steps: values.steps,
    guidance: spec.guidance_fixed === null ? values.guidance : null,
    seed: values.seed,
    count: values.count,
  };
}
