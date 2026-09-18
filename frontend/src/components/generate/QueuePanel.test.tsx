import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { QueueState, RunDetail, RunSummary } from "../../api/types";

const runtimeApi = vi.hoisted(() => ({
  getRuntime: vi.fn(),
  ejectRuntime: vi.fn(),
}));

const generationsApi = vi.hoisted(() => ({
  getQueue: vi.fn(),
  getRun: vi.fn(),
  cancelRun: vi.fn(),
  resumeQueue: vi.fn(),
}));

vi.mock("../../api/runtime", () => runtimeApi);
vi.mock("../../api/generations", () => ({
  ...generationsApi,
  listRuns: vi.fn(),
  submitGeneration: vi.fn(),
  setFavorite: vi.fn(),
  trashRun: vi.fn(),
  restoreRun: vi.fn(),
  artifactUrl: (runId: string, artifactId: string) =>
    `/api/generations/${runId}/artifacts/${artifactId}`,
  thumbnailUrl: (runId: string, artifactId: string) =>
    `/api/generations/${runId}/artifacts/${artifactId}/thumbnail`,
  metadataUrl: (runId: string) => `/api/generations/${runId}/metadata`,
}));

import { QueuePanel } from "./QueuePanel";
import { RuntimeProvider } from "../../state/runtime/RuntimeProvider";
import { ToastProvider } from "../../state/toast/ToastProvider";

function pendingRun(overrides: Partial<RunSummary>): RunSummary {
  return {
    run_id: "run-0001",
    created_at: "2026-09-16T00:00:00Z",
    status: "queued",
    favorite: false,
    trashed: false,
    prompt: "a quiet harbor",
    negative_prompt: null,
    repo_id: "Tongyi-MAI/Z-Image",
    profile: "z-image",
    image_count: 2,
    completed_count: 0,
    preview_artifact_id: null,
    ...overrides,
  };
}

function renderPanel(queue: QueueState, currentRunId: string | null = null) {
  generationsApi.getQueue.mockResolvedValue(queue);
  return renderPanelRaw(currentRunId);
}

/** Same providers without touching the getQueue mock (tests set their own). */
function renderPanelRaw(currentRunId: string | null = null) {
  runtimeApi.getRuntime.mockResolvedValue({
    implementation: "real",
    state: currentRunId ? "generating" : "idle",
    resident: null,
    current_run_id: currentRunId,
    queue_depth: 0,
    last_error: null,
  });
  return render(
    <ToastProvider>
      <RuntimeProvider>
        <QueuePanel onFocusRun={vi.fn()} />
      </RuntimeProvider>
    </ToastProvider>,
  );
}

beforeEach(() => {
  generationsApi.getQueue.mockReset();
  generationsApi.getRun.mockReset();
  generationsApi.cancelRun.mockReset();
  generationsApi.resumeQueue.mockReset();
  runtimeApi.getRuntime.mockReset();
});

describe("QueuePanel", () => {
  it("renders an empty queue state", async () => {
    renderPanel({ paused: false, pending: [] });

    expect(await screen.findByText("Queue is empty")).toBeInTheDocument();
  });

  it("lists pending runs and cancels them individually", async () => {
    generationsApi.cancelRun.mockResolvedValue({});
    renderPanel({
      paused: false,
      pending: [
        pendingRun({ run_id: "run-a", prompt: "first prompt" }),
        pendingRun({ run_id: "run-b", prompt: "second prompt" }),
      ],
    });

    expect(await screen.findByText("first prompt")).toBeInTheDocument();
    expect(screen.getByText("second prompt")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Cancel" })[0]);
    await waitFor(() =>
      expect(generationsApi.cancelRun).toHaveBeenCalledWith("run-a"),
    );
  });

  it("offers resume for the paused queue after a restart", async () => {
    generationsApi.resumeQueue.mockResolvedValue({
      paused: false,
      pending: [],
    });
    renderPanel({
      paused: true,
      pending: [pendingRun({ status: "paused" })],
    });

    expect(
      await screen.findByText(/queue is paused after a server restart/i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Resume queue" }));
    await waitFor(() =>
      expect(generationsApi.resumeQueue).toHaveBeenCalledOnce(),
    );
  });

  it("reports an unavailable queue on fetch errors and recovers", async () => {
    generationsApi.getQueue
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValue({
        paused: false,
        pending: [pendingRun({ run_id: "run-x", prompt: "recovered run" })],
      });
    renderPanelRaw();

    // No infinite spinner: the failure is stated truthfully.
    expect(
      await screen.findByText(/Queue status is unavailable/i),
    ).toBeInTheDocument();
    expect(screen.queryByText("Loading queue")).toBeNull();

    // The next poll (1 s cadence) succeeds: the notice clears and the
    // queue renders.
    expect(
      await screen.findByText("recovered run", {}, { timeout: 4_000 }),
    ).toBeInTheDocument();
    await waitFor(
      () =>
        expect(screen.queryByText(/Queue status is unavailable/i)).toBeNull(),
      { timeout: 4_000 },
    );
  });

  it("keeps the last known queue with an error indication", async () => {
    generationsApi.getQueue
      .mockResolvedValueOnce({
        paused: false,
        pending: [pendingRun({ run_id: "run-y", prompt: "stale run" })],
      })
      .mockRejectedValue(new Error("connection lost"));
    renderPanelRaw();

    expect(await screen.findByText("stale run")).toBeInTheDocument();
    expect(
      await screen.findByText(
        /Showing the last known queue/i,
        {},
        {
          timeout: 4_000,
        },
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("stale run")).toBeInTheDocument();
  });

  it("shows the active run with a cancel action", async () => {
    const active: RunDetail = {
      run_id: "run-live",
      created_at: "2026-09-16T00:00:00Z",
      started_at: "2026-09-16T00:00:05Z",
      finished_at: null,
      status: "running",
      favorite: false,
      trashed: false,
      registration_id: "reg-1",
      repo_id: "Tongyi-MAI/Z-Image",
      commit_sha: "abcdef123",
      profile: "z-image",
      dtype: "bfloat16",
      gpu: { uuid: "gpu-0", name: "RTX A" },
      prompt: "active prompt",
      negative_prompt: null,
      width: 1024,
      height: 1024,
      steps: 9,
      guidance: 0,
      initial_seed: 5,
      image_count: 1,
      pipeline_class: null,
      dependency_versions: {},
      runtime_meta: {},
      queue_position: null,
      progress: null,
      error: null,
      images: [],
    };
    generationsApi.getRun.mockResolvedValue(active);
    generationsApi.cancelRun.mockResolvedValue(active);
    renderPanel({ paused: false, pending: [] }, "run-live");

    expect(await screen.findByText("active prompt")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(generationsApi.cancelRun).toHaveBeenCalledWith("run-live"),
    );
  });
});
