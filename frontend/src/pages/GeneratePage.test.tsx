import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  ModelRegistration,
  ProfileSpec,
  RunDetail,
  RunSummary,
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

function renderPage(
  runtime = residentRuntime("reg-base"),
  profiles = PROFILES,
  registrations = REGISTRATIONS,
) {
  profilesApi.getProfiles.mockResolvedValue(profiles);
  modelsApi.listRegistrations.mockResolvedValue(registrations);
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

function runSummary(overrides: Partial<RunSummary>): RunSummary {
  return {
    run_id: "run-aaaa0000",
    created_at: new Date(Date.now() - 60_000).toISOString(),
    status: "completed",
    favorite: false,
    trashed: false,
    prompt: "a quiet harbor",
    negative_prompt: null,
    repo_id: "Tongyi-MAI/Z-Image",
    profile: "z-image",
    image_count: 2,
    completed_count: 2,
    preview_artifact_id: "image-001",
    ...overrides,
  };
}

function runDetail(overrides: Partial<RunDetail>): RunDetail {
  return {
    run_id: "run-aaaa0000",
    created_at: new Date(Date.now() - 60_000).toISOString(),
    started_at: null,
    finished_at: null,
    status: "completed",
    favorite: false,
    trashed: false,
    registration_id: "reg-base",
    repo_id: "Tongyi-MAI/Z-Image",
    commit_sha: "aaaa1111",
    profile: "z-image",
    dtype: "bfloat16",
    gpu: { uuid: "gpu-0", name: "RTX A" },
    prompt: "a quiet harbor",
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
    queue_position: null,
    progress: null,
    error: null,
    images: [],
    ...overrides,
  };
}

/** Newest-first in-memory backend for the selected-model run list. */
function mockListModelRuns(history: RunSummary[]) {
  generationsApi.listRuns.mockImplementation(
    (params: { model?: string; limit?: number; offset?: number }) => {
      const forModel = history.filter(
        (entry) => entry.repo_id === params.model,
      );
      const offset = params.offset ?? 0;
      return Promise.resolve({
        runs: forModel.slice(offset, offset + (params.limit ?? 24)),
        total: forModel.length,
      });
    },
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
  window.localStorage.clear();
  profilesApi.getProfiles.mockReset();
  modelsApi.listRegistrations.mockReset();
  systemApi.getSystem.mockReset();
  runtimeApi.getRuntime.mockReset();
  generationsApi.submitGeneration.mockReset();
  generationsApi.getRun.mockReset();
  generationsApi.listRuns.mockReset();
  generationsApi.listRuns.mockResolvedValue({ runs: [], total: 0 });
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

  it("propagates a terminal queue cancel to the followed run's results", async () => {
    const queued: RunDetail = {
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
      prompt: "cancel me",
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
    const cancelled: RunDetail = {
      ...queued,
      status: "cancelled",
      finished_at: "2026-09-16T00:00:20Z",
      error: null,
    };
    generationsApi.submitGeneration.mockResolvedValue(queued);
    // The Results view's first read resolves; every later read hangs so
    // only the propagated cancel response can update the view.
    generationsApi.getRun.mockResolvedValueOnce(queued);
    generationsApi.getRun.mockReturnValue(
      new Promise<RunDetail>(() => undefined),
    );
    generationsApi.cancelRun.mockResolvedValue(cancelled);
    renderPage();
    // renderPage resets getQueue to an empty queue; the next 1 s poll
    // carries the pending row.
    generationsApi.getQueue.mockResolvedValue({
      paused: false,
      pending: [
        {
          run_id: "run-new1",
          created_at: queued.created_at,
          status: "queued",
          favorite: false,
          trashed: false,
          prompt: "cancel me",
          negative_prompt: null,
          repo_id: "Tongyi-MAI/Z-Image",
          profile: "z-image",
          image_count: 2,
          completed_count: 0,
          preview_artifact_id: null,
        },
      ],
    });

    await waitForForm();
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "cancel me" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    expect(await screen.findByText(/run run-new1/i)).toBeInTheDocument();

    const queuePanel = screen.getByLabelText("Generation queue");
    const cancelButton = await within(queuePanel).findByRole(
      "button",
      {
        name: "Cancel",
      },
      { timeout: 4_000 },
    );
    fireEvent.click(cancelButton);

    // Queue row and followed-run Results both reflect the authoritative
    // terminal response while the Results poll is still stalled.
    await waitFor(() =>
      expect(within(queuePanel).queryByText("cancel me")).toBeNull(),
    );
    expect(await screen.findByText("Cancelled")).toBeInTheDocument();
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

  it("lists only the selected model's runs and follows the newest", async () => {
    mockListModelRuns([
      runSummary({ run_id: "run-new1", prompt: "newest base run" }),
      runSummary({
        run_id: "run-old1",
        prompt: "older base run",
        created_at: new Date(Date.now() - 3_600_000).toISOString(),
      }),
      runSummary({
        run_id: "run-turbo1",
        prompt: "turbo run",
        repo_id: "Tongyi-MAI/Z-Image-Turbo",
        profile: "z-image-turbo",
      }),
    ]);
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(runDetail({ run_id: runId })),
    );
    renderPage();

    await waitForForm();
    await waitFor(() =>
      expect(generationsApi.listRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "Tongyi-MAI/Z-Image",
          has_images: true,
          limit: 8,
        }),
        expect.anything(),
      ),
    );
    const results = screen.getByLabelText("Run results");
    expect(
      await within(results).findByTitle("newest base run"),
    ).toBeInTheDocument();
    expect(within(results).getByTitle("older base run")).toBeInTheDocument();
    expect(within(results).queryByTitle("turbo run")).toBeNull();
    expect(within(results).getAllByRole("listitem")).toHaveLength(2);

    // The newest run of the selected model is followed by default.
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-new1",
        expect.anything(),
      ),
    );
    expect(
      await within(results).findByText(/run run-new1/i),
    ).toBeInTheDocument();
  });

  it("restores the persisted model and follows its newest run after a reload", async () => {
    window.localStorage.setItem(
      "image-studio.generate.registration-id",
      "reg-turbo",
    );
    mockListModelRuns([
      runSummary({
        run_id: "run-turbo1",
        prompt: "turbo newest",
        repo_id: "Tongyi-MAI/Z-Image-Turbo",
        profile: "z-image-turbo",
      }),
    ]);
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(
        runDetail({
          run_id: runId,
          repo_id: "Tongyi-MAI/Z-Image-Turbo",
          profile: "z-image-turbo",
        }),
      ),
    );
    renderPage();

    await waitForForm();
    expect(screen.getByLabelText("Model")).toHaveValue("reg-turbo");
    await waitFor(() =>
      expect(generationsApi.listRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "Tongyi-MAI/Z-Image-Turbo",
          has_images: true,
        }),
        expect.anything(),
      ),
    );
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-turbo1",
        expect.anything(),
      ),
    );
    expect(
      await within(screen.getByLabelText("Run results")).findByTitle(
        "turbo newest",
      ),
    ).toBeInTheDocument();
  });

  it("falls back to the default model when the stored registration is gone", async () => {
    window.localStorage.setItem(
      "image-studio.generate.registration-id",
      "reg-removed",
    );
    renderPage();

    await waitForForm();
    expect(screen.getByLabelText("Model")).toHaveValue("reg-base");
  });

  it("switching models resets the selection to the new model's newest run", async () => {
    mockListModelRuns([
      runSummary({ run_id: "run-new1", prompt: "newest base run" }),
      runSummary({
        run_id: "run-turbo1",
        prompt: "turbo newest",
        repo_id: "Tongyi-MAI/Z-Image-Turbo",
        profile: "z-image-turbo",
      }),
    ]);
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(runDetail({ run_id: runId })),
    );
    renderPage();

    await waitForForm();
    const results = screen.getByLabelText("Run results");
    await within(results).findByTitle("newest base run");
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-new1",
        expect.anything(),
      ),
    );

    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "reg-turbo" },
    });

    expect(
      await within(results).findByTitle("turbo newest"),
    ).toBeInTheDocument();
    expect(within(results).queryByTitle("newest base run")).toBeNull();
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-turbo1",
        expect.anything(),
      ),
    );
    // The persisted preference follows the switch.
    expect(
      window.localStorage.getItem("image-studio.generate.registration-id"),
    ).toBe("reg-turbo");
  });

  it("ignores a delayed run list for a previous model after switching", async () => {
    const turboRuns = [
      runSummary({
        run_id: "run-turbo1",
        prompt: "turbo prompt",
        repo_id: "Tongyi-MAI/Z-Image-Turbo",
        profile: "z-image-turbo",
      }),
    ];
    let resolveBase!: (result: { runs: RunSummary[]; total: number }) => void;
    generationsApi.listRuns.mockImplementation((params: { model?: string }) => {
      if (params.model === "Tongyi-MAI/Z-Image-Turbo") {
        return Promise.resolve({ runs: turboRuns, total: turboRuns.length });
      }
      return new Promise<{ runs: RunSummary[]; total: number }>((resolve) => {
        resolveBase = resolve;
      });
    });
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(runDetail({ run_id: runId })),
    );
    renderPage();

    await waitForForm();
    const results = screen.getByLabelText("Run results");
    expect(within(results).getByText("Loading runs")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "reg-turbo" },
    });
    expect(
      await within(results).findByTitle("turbo prompt"),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-turbo1",
        expect.anything(),
      ),
    );

    // The delayed base-model response lands after the switch: it must not
    // repopulate the list or steal the selection back.
    resolveBase({
      runs: [runSummary({ run_id: "run-base1", prompt: "base prompt" })],
      total: 1,
    });
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(within(results).queryByTitle("base prompt")).toBeNull();
    expect(within(results).getByTitle("turbo prompt")).toBeInTheDocument();
    expect(generationsApi.getRun).not.toHaveBeenCalledWith(
      "run-base1",
      expect.anything(),
    );
  });

  it("shows an older run's detail when picked from the list", async () => {
    mockListModelRuns([
      runSummary({ run_id: "run-new1", prompt: "newest run" }),
      runSummary({
        run_id: "run-old1",
        prompt: "older run",
        created_at: new Date(Date.now() - 3_600_000).toISOString(),
      }),
    ]);
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(
        runDetail({
          run_id: runId,
          prompt: runId === "run-old1" ? "older run" : "newest run",
          initial_seed: runId === "run-old1" ? 41 : 7,
        }),
      ),
    );
    renderPage();

    await waitForForm();
    const results = screen.getByLabelText("Run results");
    await within(results).findByTitle("older run");
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-new1",
        expect.anything(),
      ),
    );

    fireEvent.click(within(results).getByTitle("older run"));

    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-old1",
        expect.anything(),
      ),
    );
    expect(await within(results).findByText("seed 41")).toBeInTheDocument();
    expect(within(results).getByTitle("older run")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(within(results).getByTitle("newest run")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("refreshes the run list when a generation finishes", async () => {
    const history: RunSummary[] = [];
    mockListModelRuns(history);
    generationsApi.submitGeneration.mockResolvedValue(
      runDetail({ run_id: "run-new1", prompt: "fresh run", status: "queued" }),
    );
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(runDetail({ run_id: runId, prompt: "fresh run" })),
    );
    renderPage();

    await waitForForm();
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "fresh run" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() =>
      expect(generationsApi.submitGeneration).toHaveBeenCalledOnce(),
    );

    // The finished run is now persisted; the next runtime poll still
    // reports it as the worker's current run before the following one
    // reports it cleared — that transition must refresh the list.
    history.push(runSummary({ run_id: "run-new1", prompt: "fresh run" }));
    runtimeApi.getRuntime.mockResolvedValueOnce({
      ...residentRuntime("reg-base"),
      state: "generating",
      current_run_id: "run-new1",
    });

    const results = screen.getByLabelText("Run results");
    const chip = await within(results).findByTitle(
      "fresh run",
      {},
      { timeout: 5_000 },
    );
    expect(within(chip).getByText("Completed")).toBeInTheDocument();
    // The submitted run stays followed.
    expect(
      await within(results).findByText(/run run-new1/i),
    ).toBeInTheDocument();
  });

  it("loads older runs with bounded offset-keyed page requests", async () => {
    const many = Array.from({ length: 30 }, (_, index) =>
      runSummary({
        run_id: `run-page${String(index).padStart(4, "0")}`,
        prompt: `page run ${index}`,
      }),
    );
    mockListModelRuns(many);
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(runDetail({ run_id: runId })),
    );
    renderPage();

    await waitForForm();
    const results = screen.getByLabelText("Run results");
    await within(results).findByTitle("page run 0");
    expect(within(results).getAllByRole("listitem")).toHaveLength(8);
    const more = within(results).getByRole("button", { name: /load more/i });
    expect(more).toHaveTextContent("Load more (8 of 30)");

    // Every page request stays at the page size and pages by offset, so
    // no amount of Load more can exceed the API's limit ceiling.
    for (let click = 0; click < 2; click += 1) {
      const callsBefore = generationsApi.listRuns.mock.calls.length;
      fireEvent.click(
        within(results).getByRole("button", { name: /load more/i }),
      );
      await waitFor(() =>
        expect(within(results).getAllByRole("listitem")).toHaveLength(
          8 * (click + 2),
        ),
      );
      // Runtime polling may also refresh offset zero after this page arrives.
      expect(
        generationsApi.listRuns.mock.calls
          .slice(callsBefore)
          .map(([query]) => query),
      ).toContainEqual({
        model: "Tongyi-MAI/Z-Image",
        has_images: true,
        limit: 8,
        offset: 8 * (click + 1),
      });
    }
    expect(within(results).getAllByRole("listitem")).toHaveLength(24);
    expect(
      within(results).getByRole("button", { name: /load more/i }),
    ).toHaveTextContent("Load more (24 of 30)");
  });

  it("clearing the detail keeps the run list without reselecting", async () => {
    mockListModelRuns([
      runSummary({ run_id: "run-new1", prompt: "newest run" }),
      runSummary({
        run_id: "run-old1",
        prompt: "older run",
        created_at: new Date(Date.now() - 3_600_000).toISOString(),
      }),
    ]);
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(runDetail({ run_id: runId })),
    );
    renderPage();

    await waitForForm();
    const results = screen.getByLabelText("Run results");
    expect(
      await within(results).findByText(/run run-new1/i),
    ).toBeInTheDocument();
    // Following a terminal run reports once on first observation; let that
    // refresh land before taking the idle baseline.
    await waitFor(() =>
      expect(generationsApi.listRuns.mock.calls.length).toBe(2),
    );
    const listRunsCalls = generationsApi.listRuns.mock.calls.length;
    const getRunCalls = generationsApi.getRun.mock.calls.length;

    fireEvent.click(within(results).getByRole("button", { name: "Clear" }));

    expect(
      await within(results).findByText("No run selected"),
    ).toBeInTheDocument();
    expect(within(results).getByTitle("newest run")).toBeInTheDocument();
    expect(within(results).getByTitle("older run")).toBeInTheDocument();
    // No auto-reselect and no idle refetch loop follows the clear.
    await new Promise((resolve) => setTimeout(resolve, 300));
    expect(generationsApi.getRun.mock.calls.length).toBe(getRunCalls);
    expect(generationsApi.listRuns.mock.calls.length).toBe(listRunsCalls);
  });

  it("refreshes the run list when the first detail read is already terminal", async () => {
    // The run finishes before the first detail poll lands: the first read
    // reports a terminal status, and the runtime never observes a current
    // run — the list must still pick up the finished run.
    const history: RunSummary[] = [];
    mockListModelRuns(history);
    generationsApi.submitGeneration.mockResolvedValue(
      runDetail({ run_id: "run-new1", prompt: "quick run", status: "queued" }),
    );
    let releaseDetail: (runId: string) => void = () => undefined;
    generationsApi.getRun.mockImplementation(
      (_runId: string) =>
        new Promise<RunDetail>((resolve) => {
          releaseDetail = (id) =>
            resolve(
              runDetail({
                run_id: id,
                prompt: "quick run",
                status: "completed",
                finished_at: new Date().toISOString(),
              }),
            );
        }),
    );
    renderPage();

    await waitForForm();
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "quick run" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() =>
      expect(generationsApi.submitGeneration).toHaveBeenCalledOnce(),
    );

    // The backend persists the finished run; once the panel's first detail
    // poll is actually pending, resolve it with the terminal status.
    history.push(runSummary({ run_id: "run-new1", prompt: "quick run" }));
    await waitFor(() => expect(generationsApi.getRun).toHaveBeenCalled());
    releaseDetail("run-new1");

    const results = screen.getByLabelText("Run results");
    const chip = await within(results).findByTitle(
      "quick run",
      {},
      { timeout: 5_000 },
    );
    expect(within(chip).getByText("Completed")).toBeInTheDocument();
    expect(
      await within(results).findByText(/run run-new1/i),
    ).toBeInTheDocument();
  });

  it("keeps a newer model selection when a submission response arrives late", async () => {
    mockListModelRuns([
      runSummary({
        run_id: "run-tur1",
        prompt: "turbo newest",
        repo_id: "Tongyi-MAI/Z-Image-Turbo",
        profile: "z-image-turbo",
      }),
    ]);
    let resolveSubmit!: (detail: RunDetail) => void;
    generationsApi.submitGeneration.mockImplementation(
      () =>
        new Promise<RunDetail>((resolve) => {
          resolveSubmit = resolve;
        }),
    );
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(
        runDetail({
          run_id: runId,
          repo_id: "Tongyi-MAI/Z-Image-Turbo",
          profile: "z-image-turbo",
        }),
      ),
    );
    renderPage();

    await waitForForm();
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "delayed base submission" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    // While the POST is in flight, the user switches models and the panel
    // follows the new model's newest run.
    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "reg-turbo" },
    });
    const results = screen.getByLabelText("Run results");
    expect(
      await within(results).findByText(/run run-tur1/i),
    ).toBeInTheDocument();

    resolveSubmit(
      runDetail({
        run_id: "run-late1",
        prompt: "delayed base submission",
        status: "queued",
      }),
    );

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Generate" })).toBeEnabled(),
    );
    // The late response belongs to the previous model: it must neither
    // steal the selection nor leak its run into this model's list.
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(within(results).getByText(/run run-tur1/i)).toBeInTheDocument();
    expect(generationsApi.getRun).not.toHaveBeenCalledWith(
      "run-late1",
      expect.anything(),
    );
    expect(within(results).queryByTitle("delayed base submission")).toBeNull();
  });

  it("refreshes the run list when the worker replaces the current run directly", async () => {
    mockListModelRuns([]);
    generationsApi.getRun.mockImplementation((runId: string) =>
      Promise.resolve(runDetail({ run_id: runId })),
    );
    renderPage();
    // The runtime moves from run A straight to run B without a null
    // observation in between; the list must still refresh. The override
    // is installed after renderPage (which sets its own default) and the
    // current run is flipped only once run A has actually been observed.
    let workerRunId: string | null = "run-a";
    let overridePolls = 0;
    runtimeApi.getRuntime.mockImplementation(() => {
      overridePolls += 1;
      return Promise.resolve({
        ...residentRuntime("reg-base"),
        state: "generating",
        current_run_id: workerRunId,
      });
    });

    await waitForForm();
    await waitFor(() => expect(overridePolls).toBeGreaterThan(0), {
      timeout: 5_000,
    });
    workerRunId = "run-b";

    const listRunsCalls = generationsApi.listRuns.mock.calls.length;
    await waitFor(
      () =>
        expect(generationsApi.listRuns.mock.calls.length).toBeGreaterThan(
          listRunsCalls,
        ),
      { timeout: 5_000 },
    );
  });
});

describe("Anima capabilities", () => {
  it("switches between Turbo fixed CFG and 2.9B negative prompt controls", async () => {
    const turbo: ProfileSpec = {
      ...PROFILES[1],
      profile_id: "anima-turbo",
      label: "Anima-Turbo",
      default_steps: 10,
      guidance_default: 1,
      guidance_fixed: 1,
    };
    const expanded: ProfileSpec = {
      ...PROFILES[0],
      profile_id: "anima-2.9b",
      label: "Anima 2.9B",
      default_steps: 40,
    };
    const registrations: ModelRegistration[] = [
      {
        ...REGISTRATIONS[0],
        id: "reg-anima",
        repo_id: "circlestone-labs/Anima",
        profile: "anima-turbo",
      },
      {
        ...REGISTRATIONS[0],
        id: "reg-29",
        repo_id: "Gazingstars123/Anima-2.9B",
        profile: "anima-2.9b",
      },
    ];
    renderPage(undefined, [turbo, expanded], registrations);
    await waitFor(() =>
      expect(screen.getByLabelText("Model")).toHaveValue("reg-anima"),
    );
    expect(screen.getByLabelText("Steps")).toHaveValue(10);
    expect(screen.queryByLabelText("Guidance")).toBeNull();
    expect(screen.queryByLabelText("Negative prompt")).toBeNull();
    expect(
      screen.getByText(/Guidance is fixed at 1 for Anima-Turbo/),
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "reg-29" },
    });
    expect(screen.getByLabelText("Steps")).toHaveValue(40);
    expect(screen.getByLabelText("Guidance")).toHaveValue(4);
    expect(screen.getByLabelText("Negative prompt")).toBeInTheDocument();
  });
});
