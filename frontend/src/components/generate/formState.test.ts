import { describe, expect, it } from "vitest";

import type { GpuInfo, ModelRegistration, ProfileSpec } from "../../api/types";
import type { ReusePayload } from "../../lib/reuse";
import {
  applyProfileDefaults,
  applyReuse,
  buildRequest,
  validateForm,
  EMPTY_FORM,
  type GenerationFormState,
} from "./formState";

const Z_IMAGE: ProfileSpec = {
  profile_id: "z-image",
  label: "Z-Image",
  default_steps: 50,
  min_steps: 1,
  max_steps: 100,
  guidance_default: 4,
  guidance_fixed: null,
  negative_prompt_supported: true,
  default_width: 1024,
  default_height: 1024,
  dtype: "bfloat16",
};

const Z_IMAGE_TURBO: ProfileSpec = {
  ...Z_IMAGE,
  profile_id: "z-image-turbo",
  label: "Z-Image-Turbo",
  default_steps: 9,
  guidance_default: 0,
  guidance_fixed: 0,
  negative_prompt_supported: false,
};

const REGISTRATION_BASE: ModelRegistration = {
  id: "reg-base",
  repo_id: "Tongyi-MAI/Z-Image",
  commit_sha: "abc123def456",
  profile: "z-image",
  display_name: null,
  status: "ready",
  missing_files: [],
  snapshot_path: "/cache/snap",
  created_at: "2026-09-16T00:00:00Z",
  last_used_at: null,
};

const REGISTRATION_TURBO: ModelRegistration = {
  ...REGISTRATION_BASE,
  id: "reg-turbo",
  repo_id: "Tongyi-MAI/Z-Image-Turbo",
  profile: "z-image-turbo",
};

const GPUS: GpuInfo[] = [
  { uuid: "gpu-0", name: "RTX A", index: 0, memory_total_bytes: null },
  { uuid: "gpu-1", name: "RTX B", index: 1, memory_total_bytes: null },
];

describe("applyProfileDefaults", () => {
  it("applies capability defaults and drops unsupported fields", () => {
    const state: GenerationFormState = {
      ...EMPTY_FORM,
      steps: "50",
      guidance: "7.5",
      negativePrompt: "blurry",
      width: "768",
      height: "768",
    };

    const next = applyProfileDefaults(state, Z_IMAGE_TURBO);

    expect(next.steps).toBe("9");
    expect(next.guidance).toBe("");
    expect(next.negativePrompt).toBe("");
    expect(next.width).toBe("1024");
    expect(next.height).toBe("1024");
  });

  it("keeps an adjustable guidance default for the base profile", () => {
    const next = applyProfileDefaults(EMPTY_FORM, Z_IMAGE);
    expect(next.steps).toBe("50");
    expect(next.guidance).toBe("4");
  });
});

describe("applyReuse", () => {
  const reuse: ReusePayload = {
    registrationId: "reg-base",
    repoId: "Tongyi-MAI/Z-Image",
    prompt: "a quiet harbor",
    negativePrompt: "text",
    width: 1280,
    height: 768,
    steps: 60,
    guidance: 5,
    seed: 1234,
    count: 2,
    gpuUuid: "gpu-1",
    gpuName: "RTX B",
    profile: "z-image",
  };

  it("prefills everything when the registration and GPU resolve", () => {
    const { state, notices, gpuNeedsChoice } = applyReuse(
      EMPTY_FORM,
      reuse,
      [REGISTRATION_BASE],
      [Z_IMAGE],
      GPUS,
    );

    expect(state.registrationId).toBe("reg-base");
    expect(state.prompt).toBe("a quiet harbor");
    expect(state.negativePrompt).toBe("text");
    expect(state.width).toBe("1280");
    expect(state.steps).toBe("60");
    expect(state.guidance).toBe("5");
    expect(state.seed).toBe("1234");
    expect(state.count).toBe("2");
    expect(state.gpuUuid).toBe("gpu-1");
    expect(notices).toEqual([]);
    expect(gpuNeedsChoice).toBe(false);
  });

  it("forces an explicit GPU choice when the historical GPU is gone", () => {
    const { state, notices, gpuNeedsChoice } = applyReuse(
      EMPTY_FORM,
      { ...reuse, gpuUuid: "gpu-gone", gpuName: "Old GPU" },
      [REGISTRATION_BASE],
      [Z_IMAGE],
      GPUS,
    );

    expect(state.gpuUuid).toBe("");
    expect(gpuNeedsChoice).toBe(true);
    expect(notices[0]).toContain("Old GPU");
  });

  it("clears an already-filled default GPU when the historical GPU is gone", () => {
    // Response-order race: the system list arrived first and the default
    // effect already filled gpu-0; reuse must not keep that silent
    // fallback.
    const preFilled: GenerationFormState = {
      ...EMPTY_FORM,
      gpuUuid: "gpu-0",
    };
    const { state, gpuNeedsChoice } = applyReuse(
      preFilled,
      { ...reuse, gpuUuid: "gpu-gone", gpuName: "Old GPU" },
      [REGISTRATION_BASE],
      [Z_IMAGE],
      GPUS,
    );

    expect(state.gpuUuid).toBe("");
    expect(gpuNeedsChoice).toBe(true);
  });

  it("drops profile-dependent fields and notifies when the model is gone", () => {
    const { state, notices } = applyReuse(
      EMPTY_FORM,
      reuse,
      [],
      [Z_IMAGE, Z_IMAGE_TURBO],
      GPUS,
    );

    expect(state.registrationId).toBe("");
    expect(state.negativePrompt).toBe("");
    expect(state.guidance).toBe("");
    expect(notices[0]).toContain("not registered");
  });

  it("omits the negative prompt when reusing into a Turbo registration", () => {
    // The run was recorded with the base profile, but the registration
    // it resolves to now uses Turbo.
    const { state } = applyReuse(
      EMPTY_FORM,
      { ...reuse, registrationId: "reg-turbo" },
      [REGISTRATION_TURBO],
      [Z_IMAGE, Z_IMAGE_TURBO],
      GPUS,
    );

    expect(state.registrationId).toBe("reg-turbo");
    expect(state.negativePrompt).toBe("");
    expect(state.guidance).toBe("");
  });
});

describe("validateForm", () => {
  it("reports required selections and prompt", () => {
    const { errors } = validateForm(EMPTY_FORM, Z_IMAGE);
    expect(errors.registrationId).toBe("Select a model.");
    expect(errors.gpuUuid).toBe("Select a GPU.");
    expect(errors.prompt).toBe("Enter a prompt.");
  });

  it("rejects dimensions outside the contract bounds", () => {
    const { errors } = validateForm(
      {
        ...EMPTY_FORM,
        registrationId: "reg-base",
        gpuUuid: "gpu-0",
        prompt: "x",
        width: "1000",
        height: "2048",
        steps: "50",
        guidance: "4",
        count: "1",
      },
      Z_IMAGE,
    );
    expect(errors.width).toContain("multiple of 16");
    expect(errors.height).toBeUndefined();
  });

  it("enforces profile step bounds and seed window", () => {
    const { errors } = validateForm(
      {
        ...EMPTY_FORM,
        registrationId: "reg-turbo",
        gpuUuid: "gpu-0",
        prompt: "x",
        width: "1024",
        height: "1024",
        steps: "9",
        count: "2",
        seed: "4294967295",
      },
      Z_IMAGE_TURBO,
    );
    expect(errors.seed).toContain("never wrap");
    expect(errors.steps).toBeUndefined();
    expect(errors.guidance).toBeUndefined();
  });
});

describe("buildRequest", () => {
  it("omits capability-fixed fields for Turbo", () => {
    const { values, errors } = validateForm(
      {
        ...EMPTY_FORM,
        registrationId: "reg-turbo",
        gpuUuid: "gpu-0",
        prompt: "a fox",
        width: "1024",
        height: "1024",
        steps: "9",
        count: "2",
        seed: "99",
      },
      Z_IMAGE_TURBO,
    );
    expect(errors).toEqual({});

    const request = buildRequest(values, Z_IMAGE_TURBO);
    expect(request).toEqual({
      registration_id: "reg-turbo",
      gpu_uuid: "gpu-0",
      prompt: "a fox",
      negative_prompt: null,
      width: 1024,
      height: 1024,
      steps: 9,
      guidance: null,
      seed: 99,
      count: 2,
    });
  });

  it("submits explicit values for the base profile", () => {
    const { values, errors } = validateForm(
      {
        ...EMPTY_FORM,
        registrationId: "reg-base",
        gpuUuid: "gpu-0",
        prompt: "a fox",
        negativePrompt: "blurry",
        width: "1280",
        height: "768",
        steps: "60",
        guidance: "4.5",
        count: "1",
        seed: "",
      },
      Z_IMAGE,
    );
    expect(errors).toEqual({});

    const request = buildRequest(values, Z_IMAGE);
    expect(request.negative_prompt).toBe("blurry");
    expect(request.guidance).toBe(4.5);
    expect(request.seed).toBeNull();
    expect(request.steps).toBe(60);
  });
});
