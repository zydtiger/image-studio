import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RunDetail, RunSummary } from "../api/types";

const modelsApi = vi.hoisted(() => ({ listRegistrations: vi.fn() }));
const runtimeApi = vi.hoisted(() => ({
  getRuntime: vi.fn(),
  ejectRuntime: vi.fn(),
}));
const generationsApi = vi.hoisted(() => ({
  listRuns: vi.fn(),
  getRun: vi.fn(),
  setFavorite: vi.fn(),
  trashRun: vi.fn(),
  restoreRun: vi.fn(),
  getQueue: vi.fn(),
  resumeQueue: vi.fn(),
  cancelRun: vi.fn(),
  submitGeneration: vi.fn(),
  artifactUrl: (runId: string, artifactId: string, download = false) =>
    `/api/generations/${runId}/artifacts/${artifactId}${download ? "?download=1" : ""}`,
  thumbnailUrl: (runId: string, artifactId: string) =>
    `/api/generations/${runId}/artifacts/${artifactId}/thumbnail`,
  metadataUrl: (runId: string) => `/api/generations/${runId}/metadata`,
}));

vi.mock("../api/models", () => modelsApi);
vi.mock("../api/runtime", () => runtimeApi);
vi.mock("../api/generations", () => generationsApi);

import HistoryPage from "./HistoryPage";
import { ToastProvider } from "../state/toast/ToastProvider";
import { RuntimeProvider } from "../state/runtime/RuntimeProvider";

function summary(overrides: Partial<RunSummary>): RunSummary {
  return {
    run_id: "run-0001",
    created_at: "2026-09-16T00:00:00Z",
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

function detail(overrides: Partial<RunDetail>): RunDetail {
  return {
    run_id: "run-0001",
    created_at: "2026-09-16T00:00:00Z",
    started_at: "2026-09-16T00:00:05Z",
    finished_at: "2026-09-16T00:00:30Z",
    status: "completed",
    favorite: false,
    trashed: false,
    registration_id: "reg-1",
    repo_id: "Tongyi-MAI/Z-Image",
    commit_sha: "abcdef123",
    profile: "z-image",
    dtype: "bfloat16",
    gpu: { uuid: "gpu-old", name: "Old GPU" },
    prompt: "a quiet harbor",
    negative_prompt: null,
    width: 1024,
    height: 1024,
    steps: 50,
    guidance: 4,
    initial_seed: 11,
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

const PROBE_ATTR = "data-reuse-probe";
function GenerateProbe() {
  const location = useLocation();
  // Synchronize router state to the DOM so tests can assert on it.
  useEffect(() => {
    document.body.setAttribute(
      PROBE_ATTR,
      JSON.stringify(location.state ?? null),
    );
  });
  return <p>generate-probe</p>;
}

function renderHistory(runs: RunSummary[], total = runs.length) {
  modelsApi.listRegistrations.mockResolvedValue([
    {
      id: "reg-1",
      repo_id: "Tongyi-MAI/Z-Image",
      commit_sha: "aaaa",
      profile: "z-image",
      display_name: null,
      status: "ready",
      missing_files: [],
      snapshot_path: "/x",
      created_at: "2026-09-16T00:00:00Z",
      last_used_at: null,
    },
  ]);
  runtimeApi.getRuntime.mockResolvedValue({
    implementation: "real",
    state: "unloaded",
    resident: null,
    current_run_id: null,
    queue_depth: 0,
    last_error: null,
  });
  generationsApi.listRuns.mockResolvedValue({ runs, total });
  return render(
    <ToastProvider>
      <RuntimeProvider>
        <MemoryRouter initialEntries={["/history"]}>
          <Routes>
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/generate" element={<GenerateProbe />} />
          </Routes>
        </MemoryRouter>
      </RuntimeProvider>
    </ToastProvider>,
  );
}

beforeEach(() => {
  modelsApi.listRegistrations.mockReset();
  runtimeApi.getRuntime.mockReset();
  generationsApi.listRuns.mockReset();
  generationsApi.getRun.mockReset();
  generationsApi.setFavorite.mockReset();
  generationsApi.trashRun.mockReset();
  generationsApi.restoreRun.mockReset();
  document.body.removeAttribute(PROBE_ATTR);
});

describe("HistoryPage", () => {
  it("renders run cards with statuses and thumbnails", async () => {
    renderHistory([
      summary({ run_id: "r1", prompt: "first" }),
      summary({
        run_id: "r2",
        status: "failed",
        completed_count: 0,
        preview_artifact_id: null,
        prompt: "second",
      }),
    ]);

    expect(await screen.findByText("first")).toBeInTheDocument();
    expect(screen.getByText("second")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(
      screen.getByAltText(""), // preview thumbnail
    ).toHaveAttribute(
      "src",
      "/api/generations/r1/artifacts/image-001/thumbnail",
    );
  });

  it("toggles favorites through the API", async () => {
    generationsApi.setFavorite.mockResolvedValue(summary({ favorite: true }));
    renderHistory([summary({ run_id: "r1" })]);

    const star = await screen.findByRole("button", {
      name: "Mark favorite",
    });
    fireEvent.click(star);

    await waitFor(() =>
      expect(generationsApi.setFavorite).toHaveBeenCalledWith("r1", true),
    );
  });

  it("requests the trash view explicitly", async () => {
    generationsApi.listRuns.mockResolvedValue({ runs: [], total: 0 });
    renderHistory([]);

    await screen.findByText("No runs");
    fireEvent.click(screen.getByRole("button", { name: "Trash" }));

    await waitFor(() => {
      const params = generationsApi.listRuns.mock.calls.at(-1)?.[0];
      expect(params?.trashed).toBe("only");
    });
  });

  it("passes filters into the query", async () => {
    generationsApi.listRuns.mockResolvedValue({ runs: [], total: 0 });
    renderHistory([]);

    await screen.findByText("No runs");
    fireEvent.change(screen.getByLabelText("Status"), {
      target: { value: "failed" },
    });
    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "Tongyi-MAI/Z-Image" },
    });
    fireEvent.click(screen.getByLabelText("Favorites only"));

    await waitFor(() => {
      const params = generationsApi.listRuns.mock.calls.at(-1)?.[0];
      expect(params).toMatchObject({
        status: "failed",
        model: "Tongyi-MAI/Z-Image",
        favorite: true,
      });
    });
  });

  it("scopes the empty-cancellation filter to the default library view", async () => {
    generationsApi.listRuns.mockResolvedValue({ runs: [], total: 0 });
    renderHistory([]);

    await screen.findByText("No runs");
    // Default library view: cancelled runs without images are hidden.
    expect(generationsApi.listRuns.mock.calls[0]?.[0]).toMatchObject({
      exclude_empty_cancelled: true,
    });
    expect(screen.getByLabelText("Status")).toHaveValue("");
    expect(screen.getByText("Default view")).toBeInTheDocument();

    // Any explicit status (including Cancelled) exposes every run.
    fireEvent.change(screen.getByLabelText("Status"), {
      target: { value: "cancelled" },
    });
    await waitFor(() => {
      const params = generationsApi.listRuns.mock.calls.at(-1)?.[0];
      expect(params?.status).toBe("cancelled");
      expect(params?.exclude_empty_cancelled).toBeUndefined();
    });

    // The Trash view is always fully inspectable.
    fireEvent.change(screen.getByLabelText("Status"), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Trash" }));
    await waitFor(() => {
      const params = generationsApi.listRuns.mock.calls.at(-1)?.[0];
      expect(params?.trashed).toBe("only");
      expect(params?.exclude_empty_cancelled).toBeUndefined();
    });
  });

  it("reuses parameters by navigating to Generate with a payload", async () => {
    generationsApi.getRun.mockResolvedValue(detail({}));
    renderHistory([summary({ run_id: "run-0001" })]);

    fireEvent.click(
      await screen.findByRole("button", { name: /Open run from/i }),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Reuse parameters" }),
    );

    await waitFor(() =>
      expect(document.body.getAttribute(PROBE_ATTR)).not.toBeNull(),
    );
    const reuse = (
      JSON.parse(document.body.getAttribute(PROBE_ATTR) ?? "null") as {
        reuse: Record<string, unknown>;
      }
    ).reuse;
    expect(reuse).toMatchObject({
      prompt: "a quiet harbor",
      seed: 11,
      gpuUuid: "gpu-old",
      gpuName: "Old GPU",
    });
  });

  it("trashes a run from the detail drawer", async () => {
    generationsApi.getRun.mockResolvedValue(detail({}));
    generationsApi.trashRun.mockResolvedValue(summary({ trashed: true }));
    renderHistory([summary({ run_id: "run-0001" })]);

    fireEvent.click(
      await screen.findByRole("button", { name: /Open run from/i }),
    );
    // The drawer footer and the confirmation dialog both carry the label.
    fireEvent.click(
      (await screen.findAllByRole("button", { name: "Move to Trash" }))[0],
    );
    fireEvent.click(
      (await screen.findAllByRole("button", { name: "Move to Trash" }))[1],
    );

    await waitFor(() =>
      expect(generationsApi.trashRun).toHaveBeenCalledWith("run-0001"),
    );
  });
});
