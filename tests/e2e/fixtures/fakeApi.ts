import type { BrowserContext, Route } from "@playwright/test";

/**
 * Transport-level fake of the Image Studio HTTP API, shaped exactly like
 * `src/image_studio/schemas.py`. It exists only inside Playwright route
 * interception for browser tests; nothing here ships with the app. The
 * integration owner can instead point the same specs at a real backend
 * running `serve --fake-runtime --fake-hub` via E2E_BASE_URL.
 *
 * Simulation model: the fake advances the active run on every poll of
 * /api/runtime, /api/generations/queue, or the active run detail, so the
 * application's own 1 Hz polling drives progress deterministically.
 */

export interface FakeProfilesSpec {
  profile_id: "z-image" | "z-image-turbo";
  label: string;
  default_steps: number;
  min_steps: number;
  max_steps: number;
  guidance_default: number;
  guidance_fixed: number | null;
  negative_prompt_supported: boolean;
  default_width: number;
  default_height: number;
  dtype: string;
}

const PROFILES: FakeProfilesSpec[] = [
  {
    profile_id: "z-image",
    label: "Z-Image",
    default_steps: 50,
    min_steps: 1,
    max_steps: 100,
    guidance_default: 4.0,
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
    guidance_default: 0.0,
    guidance_fixed: 0.0,
    negative_prompt_supported: false,
    default_width: 1024,
    default_height: 1024,
    dtype: "bfloat16",
  },
];

const GPUS = [
  {
    uuid: "gpu-fake-0",
    name: "Fake GPU 0",
    index: 0,
    memory_total_bytes: 32_000_000_000,
  },
  {
    uuid: "gpu-fake-1",
    name: "Fake GPU 1",
    index: 1,
    memory_total_bytes: 32_000_000_000,
  },
];

// A valid 1x1 transparent PNG served for artifacts and thumbnails.
const PNG_BYTES = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

const NOW = () => new Date().toISOString();

interface FakeImage {
  artifact_id: string;
  index: number;
  seed: number;
  status: "pending" | "completed" | "failed" | "cancelled";
  width: number | null;
  height: number | null;
  size_bytes: number | null;
  error: string | null;
  url: string | null;
  thumbnail_url: string | null;
}

interface FakeRun {
  run_id: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  status:
    | "queued"
    | "paused"
    | "running"
    | "completed"
    | "partial"
    | "failed"
    | "cancelled"
    | "interrupted";
  favorite: boolean;
  trashed: boolean;
  registration_id: string;
  repo_id: string;
  commit_sha: string;
  profile: "z-image" | "z-image-turbo";
  dtype: string;
  gpu_uuid: string | null;
  gpu_name: string | null;
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  initial_seed: number;
  image_count: number;
  images: FakeImage[];
  error: { code: string; message: string; details: unknown } | null;
  cancel_requested: boolean;
  /** 1-based index of the image currently being generated. */
  current_image: number;
  current_step: number;
}

interface FakeRegistration {
  id: string;
  repo_id: string;
  commit_sha: string;
  profile: "z-image" | "z-image-turbo";
  display_name: string | null;
  status: "ready" | "missing_files";
  missing_files: string[];
  snapshot_path: string | null;
  created_at: string;
  last_used_at: string | null;
}

interface FakeDownload {
  id: string;
  repo_id: string;
  requested_revision: string | null;
  resolved_commit: string | null;
  profile: "z-image" | "z-image-turbo";
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  error: { code: string; message: string; details: unknown } | null;
  progress: {
    bytes_done: number;
    bytes_total: number | null;
    files_done: number;
    files_total: number | null;
  };
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

interface HubEntry {
  repo_id: string;
  author: string | null;
  gated: boolean;
  downloads: number | null;
  likes: number | null;
  license: string | null;
}

const HUB_CATALOG: HubEntry[] = [
  {
    repo_id: "Tongyi-MAI/Z-Image",
    author: "Tongyi-MAI",
    gated: false,
    downloads: 12_345,
    likes: 234,
    license: "apache-2.0",
  },
  {
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    author: "Tongyi-MAI",
    gated: false,
    downloads: 9_000,
    likes: 180,
    license: "apache-2.0",
  },
  {
    repo_id: "someone/z-image-derived",
    author: "someone",
    gated: false,
    downloads: 120,
    likes: 12,
    license: "openrail",
  },
  {
    repo_id: "someone/gated-model",
    author: "someone",
    gated: true,
    downloads: 5,
    likes: 1,
    license: null,
  },
  {
    repo_id: "someone/unrelated-diffusion",
    author: "someone",
    gated: false,
    downloads: 77,
    likes: 3,
    license: "mit",
  },
];

interface FakeState {
  registrations: Map<string, FakeRegistration>;
  runs: Map<string, FakeRun>;
  queue: string[];
  paused: boolean;
  downloads: Map<string, FakeDownload>;
  runtime: {
    state:
      "unloaded" | "loading" | "idle" | "generating" | "switching" | "ejecting";
    resident: {
      registration_id: string;
      repo_id: string;
      commit_sha: string;
      profile: "z-image" | "z-image-turbo";
      dtype: string;
      gpu: { uuid: string; name: string };
    } | null;
    current_run_id: string | null;
    last_error: { code: string; message: string; details: unknown } | null;
  };
  autoStart: boolean;
  stepsPerTick: number;
  hfLoggedIn: boolean;
  failNextRunWith: { code: string; message: string } | null;
  ids: number;
}

export interface FakeApi {
  /** Registers an existing snapshot as ready. */
  addRegistration(options: {
    repo_id?: string;
    profile?: "z-image" | "z-image-turbo";
    status?: "ready" | "missing_files";
  }): FakeRegistration;
  /** Places a finished or failed run into history. */
  seedRun(options: {
    prompt: string;
    status?: FakeRun["status"];
    profile?: "z-image" | "z-image-turbo";
    favorite?: boolean;
    trashed?: boolean;
    completed?: number;
    imageCount?: number;
    gpuUuid?: string;
    registrationId?: string;
  }): FakeRun;
  /** Loads a resident model in the idle state. */
  setResidentIdle(registrationId: string, gpuUuid?: string): void;
  /** Manually dispatch the next queued run (use with autoStart off). */
  startNextRun(): void;
  finishCurrentRun(): void;
  failCurrentRun(code?: string, message?: string): void;
  /** Server restart: unload, pause the queue, interrupt active work. */
  simulateRestart(): void;
  setAutoStart(enabled: boolean): void;
  /** Steps advanced per poll tick; 0 completes one image per tick. */
  setStepsPerTick(steps: number): void;
  /** Force a download job into a terminal state for retry tests. */
  failDownload(jobId: string, message?: string): void;
  state(): {
    runs: FakeRun[];
    registrations: FakeRegistration[];
    downloads: FakeDownload[];
    runtimeState: string;
    paused: boolean;
    currentRunId: string | null;
  };
}

function errorBody(code: string, message: string, details?: unknown) {
  return { error: { code, message, details: details ?? null } };
}

async function jsonResponse(route: Route, status: number, body: unknown) {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

export async function installFakeApi(
  context: BrowserContext,
): Promise<FakeApi> {
  const state: FakeState = {
    registrations: new Map(),
    runs: new Map(),
    queue: [],
    paused: false,
    downloads: new Map(),
    runtime: {
      state: "unloaded",
      resident: null,
      current_run_id: null,
      last_error: null,
    },
    autoStart: true,
    stepsPerTick: 0, // 0 = complete one image per tick
    hfLoggedIn: false,
    failNextRunWith: null,
    ids: 0,
  };

  const nextId = (prefix: string) =>
    `${prefix}${(++state.ids).toString().padStart(4, "0")}${Math.random()
      .toString(16)
      .slice(2, 10)}`;

  const gpuByUuid = (uuid: string) => GPUS.find((gpu) => gpu.uuid === uuid);

  function summary(run: FakeRun) {
    const preview = run.images.find((image) => image.status === "completed");
    return {
      run_id: run.run_id,
      created_at: run.created_at,
      status: run.status,
      favorite: run.favorite,
      trashed: run.trashed,
      prompt: run.prompt,
      negative_prompt: run.negative_prompt,
      repo_id: run.repo_id,
      profile: run.profile,
      image_count: run.image_count,
      completed_count: run.images.filter((i) => i.status === "completed")
        .length,
      preview_artifact_id: preview ? preview.artifact_id : null,
    };
  }

  function detail(run: FakeRun) {
    const queuePosition = state.queue.indexOf(run.run_id);
    return {
      run_id: run.run_id,
      created_at: run.created_at,
      started_at: run.started_at,
      finished_at: run.finished_at,
      status: run.status,
      favorite: run.favorite,
      trashed: run.trashed,
      registration_id: run.registration_id,
      repo_id: run.repo_id,
      commit_sha: run.commit_sha,
      profile: run.profile,
      dtype: run.dtype,
      gpu:
        run.gpu_uuid !== null
          ? {
              uuid: run.gpu_uuid,
              name: run.gpu_name ?? run.gpu_uuid,
            }
          : null,
      prompt: run.prompt,
      negative_prompt: run.negative_prompt,
      width: run.width,
      height: run.height,
      steps: run.steps,
      guidance: run.guidance,
      initial_seed: run.initial_seed,
      image_count: run.image_count,
      pipeline_class: "FakeZImagePipeline",
      dependency_versions: { fake: "1.0" },
      runtime_meta: {},
      queue_position: queuePosition >= 0 ? queuePosition + 1 : null,
      progress:
        run.status === "running"
          ? {
              image_index: run.current_image,
              step: run.current_step,
              total_steps: run.steps,
            }
          : null,
      error: run.error,
      images: run.images,
    };
  }

  function startNextRun() {
    if (state.runtime.current_run_id !== null || state.paused) return;
    while (state.queue.length > 0) {
      const runId = state.queue[0];
      const run = state.runs.get(runId);
      state.queue = state.queue.slice(1);
      if (run === undefined) continue;
      if (run.status !== "queued") continue;

      const registration = state.registrations.get(run.registration_id);
      const needsSwitch =
        state.runtime.resident === null ||
        state.runtime.resident.registration_id !== run.registration_id ||
        state.runtime.resident.gpu.uuid !== run.gpu_uuid;

      run.status = "running";
      run.started_at = NOW();
      run.current_image = 1;
      run.current_step = 0;
      state.runtime.current_run_id = run.run_id;
      state.runtime.state = "generating";
      if (needsSwitch && registration !== undefined) {
        const gpu = gpuByUuid(run.gpu_uuid ?? "");
        state.runtime.resident = {
          registration_id: registration.id,
          repo_id: registration.repo_id,
          commit_sha: registration.commit_sha,
          profile: registration.profile,
          dtype: "bfloat16",
          gpu: {
            uuid: run.gpu_uuid ?? "",
            name: gpu?.name ?? run.gpu_uuid ?? "",
          },
        };
      }
      return;
    }
  }

  function tick() {
    // Advance downloads by a chunk per tick.
    for (const job of state.downloads.values()) {
      if (job.status === "queued") {
        job.status = "running";
        job.started_at = NOW();
      }
      if (job.status !== "running") continue;
      const total = job.progress.bytes_total ?? 1_000_000;
      job.progress.bytes_done = Math.min(
        total,
        job.progress.bytes_done + Math.ceil(total / 4),
      );
      job.progress.files_done = Math.min(
        job.progress.files_total ?? 4,
        job.progress.files_done + 1,
      );
      if (job.progress.bytes_done >= total) {
        job.status = "completed";
        job.finished_at = NOW();
      }
    }

    if (
      state.autoStart &&
      state.runtime.current_run_id === null &&
      !state.paused
    ) {
      startNextRun();
    }

    const runId = state.runtime.current_run_id;
    if (runId === null) return;
    const run = state.runs.get(runId);
    if (run === undefined || run.status !== "running") {
      state.runtime.current_run_id = null;
      return;
    }

    if (state.failNextRunWith !== null) {
      const failure = state.failNextRunWith;
      state.failNextRunWith = null;
      failRun(run, failure.code, failure.message);
      return;
    }

    if (run.cancel_requested) {
      const completed = run.images.filter(
        (i) => i.status === "completed",
      ).length;
      for (const image of run.images) {
        if (image.status === "pending") image.status = "cancelled";
      }
      run.status = completed > 0 ? "partial" : "cancelled";
      run.finished_at = NOW();
      state.runtime.current_run_id = null;
      state.runtime.state = "idle";
      return;
    }

    const increment = state.stepsPerTick === 0 ? run.steps : state.stepsPerTick;
    run.current_step += increment;
    while (run.current_step >= run.steps) {
      const image = run.images[run.current_image - 1];
      if (image !== undefined && image.status === "pending") {
        image.status = "completed";
        image.width = run.width;
        image.height = run.height;
        image.size_bytes = PNG_BYTES.length;
        image.url = `/api/generations/${run.run_id}/artifacts/${image.artifact_id}`;
        image.thumbnail_url = `/api/generations/${run.run_id}/artifacts/${image.artifact_id}/thumbnail`;
      }
      if (run.current_image >= run.image_count) {
        run.status = "completed";
        run.finished_at = NOW();
        state.runtime.current_run_id = null;
        state.runtime.state = "idle";
        return;
      }
      run.current_image += 1;
      run.current_step = 0;
    }
  }

  function failRun(run: FakeRun, code: string, message: string) {
    for (const image of run.images) {
      if (image.status === "pending") {
        image.status = "failed";
        image.error = message;
      }
    }
    const completed = run.images.filter((i) => i.status === "completed").length;
    run.status = completed > 0 ? "partial" : "failed";
    run.error = { code, message, details: null };
    run.finished_at = NOW();
    state.runtime.current_run_id = null;
    state.runtime.state = "unloaded";
    state.runtime.resident = null;
    state.runtime.last_error = { code, message, details: null };
  }

  const api: FakeApi = {
    addRegistration({ repo_id, profile = "z-image", status = "ready" }) {
      const repo = repo_id ?? `test/model-${state.ids + 1}`;
      const registration: FakeRegistration = {
        id: nextId("reg"),
        repo_id: repo,
        commit_sha: `c${(state.ids + 1).toString(16)}deadbeef00`,
        profile,
        display_name: null,
        status,
        missing_files: status === "missing_files" ? ["model_index.json"] : [],
        snapshot_path: `/fake/cache/${repo}`,
        created_at: NOW(),
        last_used_at: null,
      };
      state.registrations.set(registration.id, registration);
      return registration;
    },
    seedRun({
      prompt,
      status = "completed",
      profile = "z-image",
      favorite = false,
      trashed = false,
      completed,
      imageCount = 2,
      gpuUuid = "gpu-fake-0",
      registrationId,
    }) {
      const registration =
        (registrationId !== undefined
          ? state.registrations.get(registrationId)
          : undefined) ??
        Array.from(state.registrations.values()).find(
          (entry) => entry.profile === profile,
        );
      const run: FakeRun = {
        run_id: nextId("run"),
        created_at: NOW(),
        started_at: status === "queued" ? null : NOW(),
        finished_at: [
          "completed",
          "failed",
          "cancelled",
          "partial",
          "interrupted",
        ].includes(status)
          ? NOW()
          : null,
        status,
        favorite,
        trashed,
        registration_id: registration?.id ?? "reg-unknown",
        repo_id: registration?.repo_id ?? "Tongyi-MAI/Z-Image",
        commit_sha: registration?.commit_sha ?? "abc123def456",
        profile,
        dtype: "bfloat16",
        gpu_uuid: gpuUuid,
        gpu_name: gpuByUuid(gpuUuid)?.name ?? gpuUuid,
        prompt,
        negative_prompt: null,
        width: 1024,
        height: 1024,
        steps: profile === "z-image-turbo" ? 9 : 50,
        guidance: profile === "z-image-turbo" ? 0 : 4,
        initial_seed: 1000 + state.ids,
        image_count: imageCount,
        images: [],
        error: null,
        cancel_requested: false,
        current_image: 1,
        current_step: 0,
      };
      const doneCount =
        completed !== undefined
          ? completed
          : ["completed"].includes(status)
            ? imageCount
            : status === "partial"
              ? Math.max(1, imageCount - 1)
              : 0;
      for (let index = 1; index <= imageCount; index += 1) {
        const done = index <= doneCount;
        run.images.push({
          artifact_id: `image-${String(index).padStart(3, "0")}`,
          index,
          seed: run.initial_seed + index - 1,
          status: done
            ? "completed"
            : status === "failed"
              ? "failed"
              : status === "partial"
                ? "failed"
                : status === "cancelled"
                  ? "cancelled"
                  : "pending",
          width: done ? run.width : null,
          height: done ? run.height : null,
          size_bytes: done ? PNG_BYTES.length : null,
          error: null,
          url: done
            ? `/api/generations/${run.run_id}/artifacts/image-${String(index).padStart(3, "0")}`
            : null,
          thumbnail_url: done
            ? `/api/generations/${run.run_id}/artifacts/image-${String(index).padStart(3, "0")}/thumbnail`
            : null,
        });
      }
      if (status === "failed") {
        run.error = {
          code: "worker_error",
          message: "Fake OOM failure.",
          details: null,
        };
      }
      state.runs.set(run.run_id, run);
      if (status === "queued" || status === "paused") {
        state.queue.push(run.run_id);
      }
      return run;
    },
    setResidentIdle(registrationId, gpuUuid = "gpu-fake-0") {
      const registration = state.registrations.get(registrationId);
      if (registration === undefined) return;
      const gpu = gpuByUuid(gpuUuid);
      state.runtime.resident = {
        registration_id: registration.id,
        repo_id: registration.repo_id,
        commit_sha: registration.commit_sha,
        profile: registration.profile,
        dtype: "bfloat16",
        gpu: { uuid: gpuUuid, name: gpu?.name ?? gpuUuid },
      };
      state.runtime.state = "idle";
    },
    startNextRun: () => {
      state.autoStart = false;
      startNextRun();
    },
    finishCurrentRun() {
      const runId = state.runtime.current_run_id;
      if (runId === null) return;
      const run = state.runs.get(runId);
      if (run === undefined) return;
      for (const image of run.images) {
        if (image.status !== "pending") continue;
        image.status = "completed";
        image.width = run.width;
        image.height = run.height;
        image.size_bytes = PNG_BYTES.length;
        image.url = `/api/generations/${run.run_id}/artifacts/${image.artifact_id}`;
        image.thumbnail_url = `/api/generations/${run.run_id}/artifacts/${image.artifact_id}/thumbnail`;
      }
      run.status = "completed";
      run.finished_at = NOW();
      state.runtime.current_run_id = null;
      state.runtime.state = "idle";
    },
    failCurrentRun(code = "worker_error", message = "Fake worker crash.") {
      state.failNextRunWith = { code, message };
    },
    simulateRestart() {
      for (const run of state.runs.values()) {
        if (run.status === "running") {
          run.status = "interrupted";
          for (const image of run.images) {
            if (image.status === "pending") image.status = "cancelled";
          }
          run.finished_at = NOW();
        } else if (run.status === "queued") {
          run.status = "paused";
        }
      }
      state.queue = [];
      state.paused = true;
      state.runtime.state = "unloaded";
      state.runtime.resident = null;
      state.runtime.current_run_id = null;
      state.runtime.last_error = null;
    },
    setAutoStart(enabled) {
      state.autoStart = enabled;
    },
    setStepsPerTick(steps) {
      state.stepsPerTick = Math.max(0, steps);
    },
    failDownload(jobId, message = "Hub unreachable.") {
      const job = state.downloads.get(jobId);
      if (job === undefined) return;
      job.status = "failed";
      job.error = { code: "hub_unreachable", message, details: null };
      job.finished_at = NOW();
    },
    state() {
      return {
        runs: Array.from(state.runs.values()),
        registrations: Array.from(state.registrations.values()),
        downloads: Array.from(state.downloads.values()),
        runtimeState: state.runtime.state,
        paused: state.paused,
        currentRunId: state.runtime.current_run_id,
      };
    },
  };

  // Match only real API paths; a glob like "**/api/**" would also swallow
  // Vite dev-server module URLs such as /src/api/profiles.ts.
  await context.route(
    (url) => url.pathname.startsWith("/api/"),
    async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const path = url.pathname.replace(/^\/api/, "");
      const method = request.method();
      const body =
        method === "GET" || method === "HEAD"
          ? undefined
          : (request.postDataJSON() as Record<string, unknown> | undefined);
      const segments = path.split("/").filter(Boolean).map(decodeURIComponent);
      const query = url.searchParams;

      // System, profiles
      if (method === "GET" && path === "/profiles") {
        return jsonResponse(route, 200, { profiles: PROFILES });
      }
      if (method === "GET" && path === "/system") {
        return jsonResponse(route, 200, {
          host: "127.0.0.1",
          port: 7860,
          paths: {
            config_file: "/fake/.config/image-studio/config.toml",
            data_dir: "/fake/.local/share/image-studio",
            database_file: "/fake/.local/share/image-studio/app.sqlite",
            outputs_dir: "/fake/.local/share/image-studio/outputs",
            trash_dir: "/fake/.local/share/image-studio/trash",
            thumbnails_dir: "/fake/.cache/image-studio/thumbnails",
            log_file: "/fake/.local/state/image-studio/logs/app.log",
            hub_cache_dir: "/fake/.cache/huggingface/hub",
          },
          hf_logged_in: state.hfLoggedIn,
          hf_username: state.hfLoggedIn ? "fake-user" : null,
          gpus: GPUS,
          development: { fake_runtime: true, fake_hub: true },
        });
      }

      // Runtime
      if (method === "GET" && path === "/runtime") {
        tick();
        return jsonResponse(route, 200, {
          implementation: "fake",
          state: state.runtime.state,
          resident: state.runtime.resident,
          current_run_id: state.runtime.current_run_id,
          queue_depth:
            state.queue.length + (state.runtime.current_run_id ? 1 : 0),
          last_error: state.runtime.last_error,
        });
      }
      if (method === "POST" && path === "/runtime/eject") {
        // Busy workers conflict; an idle or already-unloaded worker is an
        // idempotent success.
        const busy =
          state.runtime.state === "loading" ||
          state.runtime.state === "generating" ||
          state.runtime.state === "switching" ||
          state.runtime.state === "ejecting";
        if (busy) {
          return jsonResponse(
            route,
            409,
            errorBody(
              "conflict",
              "Eject is permitted only while the worker is idle.",
            ),
          );
        }
        state.runtime.state = "unloaded";
        state.runtime.resident = null;
        return route.fulfill({ status: 204 });
      }

      // Registrations
      if (method === "GET" && path === "/models") {
        return jsonResponse(route, 200, {
          registrations: Array.from(state.registrations.values()),
        });
      }
      if (method === "POST" && path === "/models") {
        const repoId = String(body?.repo_id ?? "");
        const profile = (body?.profile ??
          "z-image") as FakeRegistration["profile"];
        const cached = HUB_CATALOG.some((entry) => entry.repo_id === repoId);
        const registeredRepo = Array.from(state.registrations.values()).some(
          (entry) => entry.repo_id === repoId,
        );
        if (!cached && !registeredRepo) {
          return jsonResponse(
            route,
            422,
            errorBody(
              "cache_incomplete",
              "The repository is not present in the local cache.",
            ),
          );
        }
        const registration = api.addRegistration({ repo_id: repoId, profile });
        return jsonResponse(route, 201, registration);
      }
      if (method === "PATCH" && segments[0] === "models" && segments[1]) {
        const registration = state.registrations.get(segments[1]);
        if (registration === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown registration."),
          );
        }
        if (
          state.runtime.resident?.registration_id === registration.id ||
          Array.from(state.runs.values()).some(
            (run) =>
              run.registration_id === registration.id &&
              ["queued", "paused", "running"].includes(run.status),
          )
        ) {
          return jsonResponse(
            route,
            409,
            errorBody(
              "conflict",
              "The registration is resident or referenced by unfinished runs.",
            ),
          );
        }
        if (typeof body?.display_name === "string") {
          registration.display_name =
            body.display_name.trim() === "" ? null : body.display_name.trim();
        }
        if (typeof body?.profile === "string") {
          registration.profile = body.profile as FakeRegistration["profile"];
        }
        return jsonResponse(route, 200, registration);
      }
      if (method === "DELETE" && segments[0] === "models" && segments[1]) {
        const registration = state.registrations.get(segments[1]);
        if (registration === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown registration."),
          );
        }
        const referencedByUnfinished = Array.from(state.runs.values()).some(
          (run) =>
            run.registration_id === registration.id &&
            ["queued", "paused", "running"].includes(run.status),
        );
        if (
          state.runtime.resident?.registration_id === registration.id ||
          referencedByUnfinished
        ) {
          return jsonResponse(
            route,
            409,
            errorBody(
              "conflict",
              referencedByUnfinished
                ? "The registration is referenced by unfinished runs."
                : "The model is currently resident.",
            ),
          );
        }
        state.registrations.delete(registration.id);
        return route.fulfill({ status: 204 });
      }

      // Hub
      if (
        method === "GET" &&
        segments[0] === "hub" &&
        segments[1] === "models" &&
        segments.length === 2
      ) {
        const q = (query.get("q") ?? "").toLowerCase();
        const results = HUB_CATALOG.filter((entry) =>
          entry.repo_id.toLowerCase().includes(q),
        ).map((entry) => ({
          ...entry,
          private: false,
          last_modified: null,
          pipeline_tag: null,
        }));
        return jsonResponse(route, 200, { results });
      }
      if (
        method === "GET" &&
        segments[0] === "hub" &&
        segments[1] === "models" &&
        segments.length === 3
      ) {
        const entry = HUB_CATALOG.find(
          (model) => model.repo_id === segments[2],
        );
        if (entry === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown repository."),
          );
        }
        return jsonResponse(route, 200, {
          ...entry,
          private: false,
          last_modified: null,
          pipeline_tag: null,
          default_revision: "main",
          revisions: [
            {
              revision: "main",
              commit_sha: `rev${entry.repo_id.length}abcdef`,
            },
          ],
          files: [
            { path: "model_index.json", size: 512 },
            { path: "model-00001-of-00002.safetensors", size: 5_000_000_000 },
          ],
        });
      }
      if (
        method === "GET" &&
        segments[0] === "hub" &&
        segments[1] === "models" &&
        segments[3] === "compatibility"
      ) {
        const repoId = segments[2];
        const known = HUB_CATALOG.find((model) => model.repo_id === repoId);
        if (known === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown repository."),
          );
        }
        const compatible = !repoId.includes("unrelated");
        return jsonResponse(route, 200, {
          repo_id: repoId,
          revision: query.get("revision") ?? "main",
          commit_sha: `rev${repoId.length}abcdef`,
          structurally_compatible: compatible,
          selectable_profiles: compatible ? ["z-image", "z-image-turbo"] : [],
          findings: compatible
            ? ["Pipeline class declares ZImagePipeline components."]
            : ["Pipeline class is not ZImagePipeline."],
          notes: ["The distillation profile remains an explicit user choice."],
        });
      }

      // Cache
      if (method === "GET" && path === "/cache/models") {
        const repos = new Map<
          string,
          { repo_id: string; snapshots: Set<string> }
        >();
        for (const registration of state.registrations.values()) {
          repos.set(registration.repo_id, {
            repo_id: registration.repo_id,
            snapshots: new Set([registration.commit_sha]),
          });
        }
        for (const entry of HUB_CATALOG.slice(0, 2)) {
          if (!repos.has(entry.repo_id)) {
            repos.set(entry.repo_id, {
              repo_id: entry.repo_id,
              snapshots: new Set(["orphan0000111"]),
            });
          }
        }
        return jsonResponse(route, 200, {
          repos: Array.from(repos.values()).map((repo) => ({
            repo_id: repo.repo_id,
            size_on_disk_bytes: 5_000_000_000,
            refs: ["main"],
            snapshots: Array.from(repo.snapshots).map((sha) => ({
              commit_sha: sha,
              path: `/fake/cache/${repo.repo_id}/${sha}`,
              size_bytes: 5_000_000_000,
              incomplete: sha === "orphan0000111",
            })),
          })),
        });
      }

      // Downloads
      if (method === "GET" && path === "/downloads") {
        tick();
        return jsonResponse(route, 200, {
          jobs: Array.from(state.downloads.values()),
        });
      }
      if (method === "POST" && path === "/downloads") {
        const repoId = String(body?.repo_id ?? "");
        const entry = HUB_CATALOG.find((model) => model.repo_id === repoId);
        if (entry?.gated === true) {
          return jsonResponse(
            route,
            403,
            errorBody(
              "gated_model",
              "The model is gated; Hub access is required.",
            ),
          );
        }
        const job: FakeDownload = {
          id: nextId("dl"),
          repo_id: repoId,
          requested_revision: (body?.revision as string | undefined) ?? null,
          resolved_commit: `dlc${state.ids}`,
          profile: (body?.profile ?? "z-image") as FakeDownload["profile"],
          status: "queued",
          error: null,
          progress: {
            bytes_done: 0,
            bytes_total: 1_000_000,
            files_done: 0,
            files_total: 4,
          },
          created_at: NOW(),
          started_at: null,
          finished_at: null,
        };
        state.downloads.set(job.id, job);
        return jsonResponse(route, 202, job);
      }
      if (
        method === "POST" &&
        segments[0] === "downloads" &&
        segments[2] === "retry"
      ) {
        const job = state.downloads.get(segments[1]);
        if (job === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown job."),
          );
        }
        job.status = "queued";
        job.error = null;
        job.progress = {
          bytes_done: 0,
          bytes_total: 1_000_000,
          files_done: 0,
          files_total: 4,
        };
        job.finished_at = null;
        return jsonResponse(route, 202, job);
      }
      if (
        method === "POST" &&
        segments[0] === "downloads" &&
        segments[2] === "cancel"
      ) {
        const job = state.downloads.get(segments[1]);
        if (job === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown job."),
          );
        }
        if (job.status === "running") {
          return jsonResponse(
            route,
            409,
            errorBody("conflict", "A running download cannot be cancelled."),
          );
        }
        if (job.status === "queued") {
          job.status = "cancelled";
          job.finished_at = NOW();
        }
        return jsonResponse(route, 202, job);
      }

      // Generations
      if (method === "POST" && path === "/generations") {
        // A paused queue rejects new submissions without persisting a run.
        if (state.paused) {
          return jsonResponse(
            route,
            409,
            errorBody(
              "conflict",
              "The queue is paused after a restart. Resume it before submitting new runs.",
            ),
          );
        }
        const registration = state.registrations.get(
          String(body?.registration_id ?? ""),
        );
        if (registration === undefined || registration.status !== "ready") {
          return jsonResponse(
            route,
            422,
            errorBody("validation", "A ready registration is required."),
          );
        }
        if (gpuByUuid(String(body?.gpu_uuid ?? "")) === undefined) {
          return jsonResponse(
            route,
            422,
            errorBody("validation", "Unknown GPU."),
          );
        }
        const spec = PROFILES.find(
          (entry) => entry.profile_id === registration.profile,
        )!;
        const count = Number(body?.count ?? 1);
        const seedInput = body?.seed as number | null | undefined;
        const initialSeed =
          seedInput === null || seedInput === undefined
            ? Math.floor(Math.random() * 1_000_000)
            : seedInput;
        const run: FakeRun = {
          run_id: nextId("run"),
          created_at: NOW(),
          started_at: null,
          finished_at: null,
          status: "queued",
          favorite: false,
          trashed: false,
          registration_id: registration.id,
          repo_id: registration.repo_id,
          commit_sha: registration.commit_sha,
          profile: registration.profile,
          dtype: spec.dtype,
          gpu_uuid: String(body?.gpu_uuid),
          gpu_name: gpuByUuid(String(body?.gpu_uuid))?.name ?? null,
          prompt: String(body?.prompt ?? ""),
          negative_prompt:
            spec.negative_prompt_supported &&
            typeof body?.negative_prompt === "string" &&
            body.negative_prompt !== ""
              ? body.negative_prompt
              : null,
          width: Number(body?.width),
          height: Number(body?.height),
          steps: Number(body?.steps ?? spec.default_steps),
          guidance:
            spec.guidance_fixed !== null
              ? spec.guidance_fixed
              : Number(body?.guidance ?? spec.guidance_default),
          initial_seed: initialSeed,
          image_count: count,
          images: [],
          error: null,
          cancel_requested: false,
          current_image: 1,
          current_step: 0,
        };
        for (let index = 1; index <= count; index += 1) {
          run.images.push({
            artifact_id: `image-${String(index).padStart(3, "0")}`,
            index,
            seed: initialSeed + index - 1,
            status: "pending",
            width: null,
            height: null,
            size_bytes: null,
            error: null,
            url: null,
            thumbnail_url: null,
          });
        }
        state.runs.set(run.run_id, run);
        state.queue.push(run.run_id);
        return jsonResponse(route, 202, detail(run));
      }
      if (method === "GET" && path === "/generations/queue") {
        tick();
        const pending = state.queue
          .map((id) => state.runs.get(id))
          .filter((run): run is FakeRun => run !== undefined)
          .map(summary);
        return jsonResponse(route, 200, { paused: state.paused, pending });
      }
      if (method === "POST" && path === "/generations/queue/resume") {
        state.paused = false;
        for (const run of state.runs.values()) {
          if (run.status === "paused") {
            run.status = "queued";
            state.queue.push(run.run_id);
          }
        }
        const pending = state.queue
          .map((id) => state.runs.get(id))
          .filter((run): run is FakeRun => run !== undefined)
          .map(summary);
        return jsonResponse(route, 200, { paused: false, pending });
      }
      if (method === "GET" && path === "/generations") {
        // Contract validation: limit 1..200, offset >= 0, trashed only
        // "exclude" (default) or "only".
        const trashedParam = query.get("trashed");
        if (
          trashedParam !== null &&
          trashedParam !== "only" &&
          trashedParam !== "exclude"
        ) {
          return jsonResponse(
            route,
            422,
            errorBody("validation", "trashed must be 'exclude' or 'only'."),
          );
        }
        const limitRaw = query.get("limit") ?? "24";
        const offsetRaw = query.get("offset") ?? "0";
        const limit = Number(limitRaw);
        const offset = Number(offsetRaw);
        if (
          !Number.isInteger(limit) ||
          limit < 1 ||
          limit > 200 ||
          !Number.isInteger(offset) ||
          offset < 0
        ) {
          return jsonResponse(
            route,
            422,
            errorBody(
              "validation",
              "limit must be 1-200 and offset must be 0 or greater.",
            ),
          );
        }
        const runs = Array.from(state.runs.values());
        const q = (query.get("q") ?? "").toLowerCase();
        const status = query.get("status");
        const model = query.get("model");
        const favorite = query.get("favorite") === "true";
        const trashedOnly = trashedParam === "only";
        const filtered = runs
          .filter((run) => (trashedOnly ? run.trashed : !run.trashed))
          .filter((run) =>
            q === "" ? true : run.prompt.toLowerCase().includes(q),
          )
          .filter((run) =>
            status === null || status === "" ? true : run.status === status,
          )
          .filter((run) =>
            model === null || model === "" ? true : run.repo_id === model,
          )
          .filter((run) => (favorite ? run.favorite : true))
          // Newest first; the run id (which embeds the insertion counter)
          // breaks created_at ties deterministically for seeded runs.
          .sort(
            (a, b) =>
              b.created_at.localeCompare(a.created_at) ||
              b.run_id.localeCompare(a.run_id),
          );
        return jsonResponse(route, 200, {
          runs: filtered.slice(offset, offset + limit).map(summary),
          total: filtered.length,
        });
      }
      if (
        method === "GET" &&
        segments[0] === "generations" &&
        segments.length === 2
      ) {
        const run = state.runs.get(segments[1]);
        if (run === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown run."),
          );
        }
        if (run.status === "running") tick();
        return jsonResponse(route, 200, detail(run));
      }
      if (
        method === "PATCH" &&
        segments[0] === "generations" &&
        segments.length === 2
      ) {
        const run = state.runs.get(segments[1]);
        if (run === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown run."),
          );
        }
        run.favorite = Boolean(body?.favorite);
        return jsonResponse(route, 200, summary(run));
      }
      if (
        method === "POST" &&
        segments[0] === "generations" &&
        segments[2] === "cancel"
      ) {
        const run = state.runs.get(segments[1]);
        if (run === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown run."),
          );
        }
        if (run.status === "queued") {
          run.status = "cancelled";
          run.finished_at = NOW();
          for (const image of run.images) image.status = "cancelled";
          state.queue = state.queue.filter((id) => id !== run.run_id);
        } else if (run.status === "running") {
          run.cancel_requested = true;
        }
        return jsonResponse(route, 202, detail(run));
      }
      if (
        method === "GET" &&
        segments[0] === "generations" &&
        segments[2] === "artifacts" &&
        segments.length === 4
      ) {
        const run = state.runs.get(segments[1]);
        const image = run?.images.find(
          (entry) => entry.artifact_id === segments[3],
        );
        if (
          run === undefined ||
          image === undefined ||
          image.status !== "completed"
        ) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown artifact."),
          );
        }
        const download = query.get("download") === "1";
        return route.fulfill({
          status: 200,
          contentType: "image/png",
          headers: download
            ? {
                "content-disposition": `attachment; filename="${image.artifact_id}.png"`,
              }
            : {},
          body: PNG_BYTES,
        });
      }
      if (
        method === "GET" &&
        segments[0] === "generations" &&
        segments[2] === "artifacts" &&
        segments[3] !== undefined &&
        segments[4] === "thumbnail"
      ) {
        const run = state.runs.get(segments[1]);
        const image = run?.images.find(
          (entry) => entry.artifact_id === segments[3],
        );
        if (
          run === undefined ||
          image === undefined ||
          image.status !== "completed"
        ) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown thumbnail."),
          );
        }
        return route.fulfill({
          status: 200,
          // Truthful content type: the fake serves PNG bytes, not WebP.
          contentType: "image/png",
          body: PNG_BYTES,
        });
      }
      if (
        method === "GET" &&
        segments[0] === "generations" &&
        segments[2] === "metadata"
      ) {
        const run = state.runs.get(segments[1]);
        if (run === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown run."),
          );
        }
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          headers: {
            "content-disposition": `attachment; filename="metadata.json"`,
          },
          body: JSON.stringify(detail(run), null, 2),
        });
      }
      if (
        method === "POST" &&
        segments[0] === "generations" &&
        (segments[2] === "trash" || segments[2] === "restore")
      ) {
        const run = state.runs.get(segments[1]);
        if (run === undefined) {
          return jsonResponse(
            route,
            404,
            errorBody("not_found", "Unknown run."),
          );
        }
        run.trashed = segments[2] === "trash";
        return jsonResponse(route, 200, summary(run));
      }

      return jsonResponse(
        route,
        404,
        errorBody("not_found", `No such endpoint: ${method} ${path}`),
      );
    },
  );

  return api;
}
