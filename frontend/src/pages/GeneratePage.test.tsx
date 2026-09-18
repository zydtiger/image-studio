import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  ModelRegistration,
  ProfileSpec,
  RunDetail,
  SystemInfo,
} from "../api/types";

const profilesApi = vi.hoisted(() => ({ getProfiles: vi.fn() }));
const modelsApi = vi.hoisted(() => ({
  listRegistrations: vi.fn(),
  createRegistration: vi.fn(),
  updateRegistration: vi.fn(),
  deleteRegistration: vi.fn(),
}));
const systemApi = vi.hoisted(() => ({ getSystem: vi.fn() }));
const runtimeApi = vi.hoisted(() => ({
  getRuntime: vi.fn(),
  ejectRuntime: vi.fn(),
}));
const generationsApi = vi.hoisted(() => ({
  submitGeneration: vi.fn(),
  getRun: vi.fn(),
  getQueue: vi.fn().mockResolvedValue({ paused: false, pending: [] }),
  listRuns: vi.fn(),
  cancelRun: vi.fn(),
  resumeQueue: vi.fn(),
  setFavorite: vi.fn(),
  trashRun: vi.fn(),
  restoreRun: vi.fn(),
  artifactUrl: (runId: string, artifactId: string, download = false) =>
    `/api/generations/${runId}/artifacts/${artifactId}${download ? "?download=1" : ""}`,
  thumbnailUrl: (runId: string, artifactId: string) =>
    `/api/generations/${runId}/artifacts/${artifactId}/thumbnail`,
  metadataUrl: (runId: string) => `/api/generations/${runId}/metadata`,
}));

vi.mock("../api/profiles", () => profilesApi);
vi.mock("../api/models", () => modelsApi);
vi.mock("../api/system", () => systemApi);
vi.mock("../api/runtime", () => runtimeApi);
vi.mock("../api/generations", () => generationsApi);

import GeneratePage from "./GeneratePage";
import { ApiError } from "../api/client";
import { ToastProvider } from "../state/toast/ToastProvider";
import { RuntimeProvider } from "../state/runtime/RuntimeProvider";

const PROFILES: ProfileSpec[] = [
  {
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
  },
  {
    profile_id: "z-image-turbo",
    label: "Z-Image-Turbo",
    default_steps: 9,
    min_steps: 1,
    max_steps: 100,
    guidance_default: 0,
    guidance_fixed: 0,
    negative_prompt_supported: false,
    default_width: 1024,
    default_height: 1024,
    dtype: "bfloat16",
  },
];

const REGISTRATIONS: ModelRegistration[] = [
  {
    id: "reg-base",
    repo_id: "Tongyi-MAI/Z-Image",
    commit_sha: "aaaa1111",
    profile: "z-image",
    display_name: null,
    status: "ready",
    missing_files: [],
    snapshot_path: "/cache/a",
    created_at: "2026-09-16T00:00:00Z",
    last_used_at: null,
  },
  {
    id: "reg-turbo",
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    commit_sha: "bbbb2222",
    profile: "z-image-turbo",
    display_name: null,
    status: "ready",
    missing_files: [],
    snapshot_path: "/cache/b",
    created_at: "2026-09-16T00:00:00Z",
    last_used_at: null,
  },
  {
    id: "reg-broken",
    repo_id: "someone/other",
    commit_sha: "cccc3333",
    profile: "z-image",
    display_name: null,
    status: "missing_files",
    missing_files: ["model_index.json"],
    snapshot_path: "/cache/c",
    created_at: "2026-09-16T00:00:00Z",
    last_used_at: null,
  },
];

const SYSTEM: SystemInfo = {
  host: "127.0.0.1",
  port: 7860,
  paths: {
    config_file: "/cfg",
    data_dir: "/data",
    database_file: "/data/app.sqlite",
    outputs_dir: "/data/outputs",
    trash_dir: "/data/trash",
    thumbnails_dir: "/cache/thumbnails",
    log_file: "/state/app.log",
    hub_cache_dir: "/cache/hub",
  },
  hf_logged_in: false,
  hf_username: null,
  gpus: [
    { uuid: "gpu-0", name: "RTX A", index: 0, memory_total_bytes: null },
    { uuid: "gpu-1", name: "RTX B", index: 1, memory_total_bytes: null },
  ],
  development: { fake_runtime: false, fake_hub: false },
};

function residentRuntime(registrationId: string, gpuUuid = "gpu-0") {
  return {
    implementation: "real" as const,
    state: "idle" as const,
    resident: {
      registration_id: registrationId,
      repo_id:
        REGISTRATIONS.find((entry) => entry.id === registrationId)?.repo_id ??
        "",
      commit_sha: "aaaa1111",
      profile: "z-image" as const,
      dtype: "bfloat16",
      gpu: { uuid: gpuUuid, name: gpuUuid === "gpu-0" ? "RTX A" : "RTX B" },
    },
    current_run_id: null,
    queue_depth: 0,
    last_error: null,
  };
}

function renderPage(runtime = residentRuntime("reg-base")) {
  profilesApi.getProfiles.mockResolvedValue(PROFILES);
  modelsApi.listRegistrations.mockResolvedValue(REGISTRATIONS);
  systemApi.getSystem.mockResolvedValue(SYSTEM);
  runtimeApi.getRuntime.mockResolvedValue(runtime);
  generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });
  return render(
    <ToastProvider>
      <RuntimeProvider>
        <MemoryRouter>
          <GeneratePage />
        </MemoryRouter>
      </RuntimeProvider>
    </ToastProvider>,
  );
}

async function waitForForm() {
  await screen.findByLabelText("Prompt");
  // Wait until the async defaults (model and GPU) have been applied.
  await waitFor(() => {
    expect(
      (screen.getByLabelText("Model") as HTMLSelectElement).value,
    ).not.toBe("");
    expect((screen.getByLabelText("GPU") as HTMLSelectElement).value).not.toBe(
      "",
    );
  });
}

beforeEach(() => {
  profilesApi.getProfiles.mockReset();
  modelsApi.listRegistrations.mockReset();
  systemApi.getSystem.mockReset();
  runtimeApi.getRuntime.mockReset();
  generationsApi.submitGeneration.mockReset();
  generationsApi.getRun.mockReset();
});

describe("GeneratePage", () => {
  it("defaults to the first ready model and the resident GPU", async () => {
    renderPage();

    await waitForForm();
    const model = screen.getByLabelText("Model") as HTMLSelectElement;
    expect(model.value).toBe("reg-base");
    const gpu = screen.getByLabelText("GPU") as HTMLSelectElement;
    expect(gpu.value).toBe("gpu-0");
    expect((screen.getByLabelText("Steps") as HTMLInputElement).value).toBe(
      "50",
    );
  });

  it("renders capability-driven fields for the base profile", async () => {
    renderPage();

    await waitForForm();
    expect(screen.getByLabelText("Negative prompt")).toBeInTheDocument();
    expect(screen.getByLabelText("Guidance")).toBeInTheDocument();
    expect(
      screen.getByText(/Reuses the resident model on its current GPU/i),
    ).toBeInTheDocument();
  });

  it("hides fixed-capability fields when a Turbo model is selected", async () => {
    renderPage();

    await waitForForm();
    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "reg-turbo" },
    });

    expect(screen.queryByLabelText("Negative prompt")).toBeNull();
    expect(screen.queryByLabelText("Guidance")).toBeNull();
    expect((screen.getByLabelText("Steps") as HTMLInputElement).value).toBe(
      "9",
    );
    expect(
      screen.getByText(/Guidance is fixed at 0 for Z-Image-Turbo/i),
    ).toBeInTheDocument();
  });

  it("explains a model or GPU replacement before submission", async () => {
    renderPage();

    await waitForForm();
    fireEvent.change(screen.getByLabelText("GPU"), {
      target: { value: "gpu-1" },
    });

    expect(
      await screen.findByText(/is fully unloaded and .* loads on RTX B/i),
    ).toBeInTheDocument();
  });

  it("submits a frozen request and follows the returned run", async () => {
    const detail: RunDetail = {
      run_id: "run-new1",
      created_at: "2026-09-16T00:00:00Z",
      started_at: null,
      finished_at: null,
      status: "queued",
      favorite: false,
      trashed: false,
      registration_id: "reg-base",
      repo_id: "Tongyi-MAI/Z-Image",
      commit_sha: "aaaa1111",
      profile: "z-image",
      dtype: "bfloat16",
      gpu: { uuid: "gpu-0", name: "RTX A" },
      prompt: "a quiet harbor, 你好",
      negative_prompt: null,
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 4,
      initial_seed: 7,
      image_count: 2,
      pipeline_class: null,
      dependency_versions: {},
      runtime_meta: {},
      queue_position: 1,
      progress: null,
      error: null,
      images: [],
    };
    generationsApi.submitGeneration.mockResolvedValue(detail);
    generationsApi.getRun.mockResolvedValue(detail);
    renderPage();

    await waitForForm();
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "a quiet harbor, 你好" },
    });
    fireEvent.change(screen.getByLabelText("Images"), {
      target: { value: "2" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() =>
      expect(generationsApi.submitGeneration).toHaveBeenCalledOnce(),
    );
    const request = generationsApi.submitGeneration.mock.calls[0][0];
    expect(request).toMatchObject({
      registration_id: "reg-base",
      gpu_uuid: "gpu-0",
      prompt: "a quiet harbor, 你好",
      steps: 50,
      guidance: 4,
      seed: null,
      count: 2,
    });
    expect(await screen.findByText(/run run-new1/i)).toBeInTheDocument();
  });

  it("blocks submission on validation errors without calling the API", async () => {
    renderPage();

    await waitForForm();
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByText("Enter a prompt.")).toBeInTheDocument();
    expect(generationsApi.submitGeneration).not.toHaveBeenCalled();
  });

  it("surfaces server validation failures inline", async () => {
    generationsApi.submitGeneration.mockRejectedValue(
      new ApiError(
        "guidance is fixed at 0.0 for z-image-turbo; submit without guidance",
        { kind: "http", status: 422, code: "validation" },
      ),
    );
    renderPage();

    await waitForForm();
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "hello" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    expect(
      await screen.findByText(/guidance is fixed at 0.0/i),
    ).toBeInTheDocument();
  });

  it("requires an explicit GPU after reuse even when system answered first", async () => {
    // Controlled response order: system (GPU list) resolves immediately
    // so the default GPU fills, then registrations resolve and the reuse
    // payload (historical GPU gone) must clear that default.
    const reusePayload = {
      registrationId: "reg-base",
      repoId: "Tongyi-MAI/Z-Image",
      prompt: "reused prompt",
      negativePrompt: null,
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 4,
      seed: 7,
      count: 1,
      gpuUuid: "gpu-retired",
      gpuName: "Retired GPU",
      profile: "z-image" as const,
    };
    let resolveModels!: (registrations: ModelRegistration[]) => void;
    modelsApi.listRegistrations.mockImplementation(
      () =>
        new Promise<ModelRegistration[]>((resolve) => {
          resolveModels = resolve;
        }),
    );
    profilesApi.getProfiles.mockResolvedValue(PROFILES);
    systemApi.getSystem.mockResolvedValue(SYSTEM);
    runtimeApi.getRuntime.mockResolvedValue(residentRuntime("reg-base"));
    generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });

    render(
      <ToastProvider>
        <RuntimeProvider>
          <MemoryRouter
            initialEntries={[
              { pathname: "/generate", state: { reuse: reusePayload } },
            ]}
          >
            <GeneratePage />
          </MemoryRouter>
        </RuntimeProvider>
      </ToastProvider>,
    );

    // The default GPU fills while models are still pending. (Do not use
    // waitForForm here: the model default cannot apply yet by design.)
    await screen.findByLabelText("Prompt");
    await waitFor(() =>
      expect(screen.getByLabelText("GPU")).toHaveValue("gpu-0"),
    );

    // Registrations arrive; reuse applies and clears the silent default.
    resolveModels(REGISTRATIONS);

    await waitFor(() =>
      expect(screen.getByLabelText("Prompt")).toHaveValue("reused prompt"),
    );
    await waitFor(() => expect(screen.getByLabelText("GPU")).toHaveValue(""));
    expect(
      screen.getByText(/Retired GPU.*is not available/),
    ).toBeInTheDocument();

    // Submission stays blocked until the user picks a GPU.
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    expect(await screen.findByText("Select a GPU.")).toBeInTheDocument();
    expect(generationsApi.submitGeneration).not.toHaveBeenCalled();

    // A user GPU choice clears the hint and unblocks submission.
    fireEvent.change(screen.getByLabelText("GPU"), {
      target: { value: "gpu-1" },
    });
    await waitFor(() =>
      expect(screen.queryByText(/Choose a GPU to continue/)).toBeNull(),
    );
  });

  it("waits for profiles before consuming the reuse payload", async () => {
    const reusePayload = {
      registrationId: "reg-base",
      repoId: "Tongyi-MAI/Z-Image",
      prompt: "late profiles prefill",
      negativePrompt: null,
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 4,
      seed: 9,
      count: 1,
      gpuUuid: "gpu-1",
      gpuName: "RTX B",
      profile: "z-image" as const,
    };
    let resolveProfiles!: (specs: typeof PROFILES) => void;
    profilesApi.getProfiles.mockImplementationOnce(
      () =>
        new Promise<typeof PROFILES>((resolve) => {
          resolveProfiles = resolve;
        }),
    );
    modelsApi.listRegistrations.mockResolvedValue(REGISTRATIONS);
    systemApi.getSystem.mockResolvedValue(SYSTEM);
    runtimeApi.getRuntime.mockResolvedValue(residentRuntime("reg-base"));
    generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });

    render(
      <ToastProvider>
        <RuntimeProvider>
          <MemoryRouter
            initialEntries={[
              { pathname: "/generate", state: { reuse: reusePayload } },
            ]}
          >
            <GeneratePage />
          </MemoryRouter>
        </RuntimeProvider>
      </ToastProvider>,
    );

    // Models and GPUs are in, but profiles are still pending: the one-shot
    // prefill must not fire (no prompt, no misleading not-registered hint).
    await waitFor(() => expect(screen.getByLabelText("Model")).toHaveValue(""));
    expect(screen.getByLabelText("Prompt")).toHaveValue("");
    expect(screen.queryByText(/not registered anymore/)).toBeNull();

    resolveProfiles(PROFILES);

    await waitFor(() =>
      expect(screen.getByLabelText("Prompt")).toHaveValue(
        "late profiles prefill",
      ),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Model")).toHaveValue("reg-base"),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("GPU")).toHaveValue("gpu-1"),
    );
  });

  it("recovers the reuse prefill after a failed models query is retried", async () => {
    const reusePayload = {
      registrationId: "reg-base",
      repoId: "Tongyi-MAI/Z-Image",
      prompt: "retry recovers prefill",
      negativePrompt: null,
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 4,
      seed: 3,
      count: 1,
      gpuUuid: "gpu-0",
      gpuName: "RTX A",
      profile: "z-image" as const,
    };
    profilesApi.getProfiles.mockResolvedValue(PROFILES);
    systemApi.getSystem.mockResolvedValue(SYSTEM);
    runtimeApi.getRuntime.mockResolvedValue(residentRuntime("reg-base"));
    generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });
    modelsApi.listRegistrations
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue(REGISTRATIONS);

    render(
      <ToastProvider>
        <RuntimeProvider>
          <MemoryRouter
            initialEntries={[
              { pathname: "/generate", state: { reuse: reusePayload } },
            ]}
          >
            <GeneratePage />
          </MemoryRouter>
        </RuntimeProvider>
      </ToastProvider>,
    );

    // The failed query surfaces its error and the prefill stays pending.
    expect(
      await screen.findByText("Models could not be loaded"),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByLabelText("Prompt")).toHaveValue(""),
    );
    expect(screen.queryByText(/not registered anymore/)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() =>
      expect(screen.getByLabelText("Prompt")).toHaveValue(
        "retry recovers prefill",
      ),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Model")).toHaveValue("reg-base"),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("GPU")).toHaveValue("gpu-0"),
    );
  });

  it("applies the resident GPU default once the runtime answers late", async () => {
    const lateRuntime = residentRuntime("reg-base", "gpu-1");
    let resolveRuntime!: (status: typeof lateRuntime) => void;
    runtimeApi.getRuntime.mockImplementation(
      () =>
        new Promise<typeof lateRuntime>((resolve) => {
          resolveRuntime = resolve;
        }),
    );
    modelsApi.listRegistrations.mockResolvedValue(REGISTRATIONS);
    systemApi.getSystem.mockResolvedValue(SYSTEM);
    profilesApi.getProfiles.mockResolvedValue(PROFILES);
    generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });

    render(
      <ToastProvider>
        <RuntimeProvider>
          <MemoryRouter>
            <GeneratePage />
          </MemoryRouter>
        </RuntimeProvider>
      </ToastProvider>,
    );

    await screen.findByLabelText("Prompt");
    // System answered but the runtime has not: no provisional default.
    await waitFor(() =>
      expect(screen.getByLabelText("Model")).toHaveValue("reg-base"),
    );
    expect(screen.getByLabelText("GPU")).toHaveValue("");

    // The resident worker runs on the SECOND GPU; the default follows it.
    resolveRuntime(lateRuntime);
    await waitFor(() =>
      expect(screen.getByLabelText("GPU")).toHaveValue("gpu-1"),
    );
  });

  it("never overwrites an explicit user GPU choice with a later resident default", async () => {
    const lateRuntime = residentRuntime("reg-base", "gpu-1");
    let resolveRuntime!: (status: typeof lateRuntime) => void;
    runtimeApi.getRuntime.mockImplementation(
      () =>
        new Promise<typeof lateRuntime>((resolve) => {
          resolveRuntime = resolve;
        }),
    );
    modelsApi.listRegistrations.mockResolvedValue(REGISTRATIONS);
    systemApi.getSystem.mockResolvedValue(SYSTEM);
    profilesApi.getProfiles.mockResolvedValue(PROFILES);
    generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });

    render(
      <ToastProvider>
        <RuntimeProvider>
          <MemoryRouter>
            <GeneratePage />
          </MemoryRouter>
        </RuntimeProvider>
      </ToastProvider>,
    );

    await screen.findByLabelText("Prompt");
    // The user explicitly picks the first GPU before the runtime answers.
    fireEvent.change(screen.getByLabelText("GPU"), {
      target: { value: "gpu-0" },
    });

    // The late runtime status arrives (resident on the SECOND GPU); the
    // explicit user choice must survive it.
    resolveRuntime(lateRuntime);
    await waitFor(() => expect(screen.getByText("RTX B")).toBeInTheDocument());
    expect(screen.getByLabelText("GPU")).toHaveValue("gpu-0");
  });
});
