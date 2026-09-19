import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  QueueState,
  RunDetail,
  RunSummary,
  RuntimeStatus,
} from "../../api/types";

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

function runDetail(overrides: Partial<RunDetail>): RunDetail {
  return {
    run_id: "run-0001",
    created_at: "2026-09-16T00:00:00Z",
    started_at: "2026-09-16T00:00:05Z",
    finished_at: "2026-09-16T00:00:09Z",
    status: "cancelled",
    favorite: false,
    trashed: false,
    registration_id: "reg-1",
    repo_id: "Tongyi-MAI/Z-Image",
    commit_sha: "abcdef123",
    profile: "z-image",
    dtype: "bfloat16",
    gpu: { uuid: "gpu-0", name: "RTX A" },
    prompt: "a quiet harbor",
    negative_prompt: null,
    width: 1024,
    height: 1024,
    steps: 9,
    guidance: 0,
    initial_seed: 5,
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
  it.each([
    ["queue-first", "loading", "Loading model"],
    ["runtime-first", "loading", "Loading model"],
    ["queue-first", "switching", "Switching model"],
    ["runtime-first", "switching", "Switching model"],
    ["queue-first", "generating", "Running"],
    ["runtime-first", "generating", "Running"],
  ] as const)(
    "shows each run once with %s responses while %s",
    async (order, state, label) => {
      let resolveQueue!: (queue: QueueState) => void;
      let resolveRuntime!: (runtime: RuntimeStatus) => void;
      generationsApi.getQueue.mockReturnValue(
        new Promise<QueueState>((resolve) => {
          resolveQueue = resolve;
        }),
      );
      runtimeApi.getRuntime.mockReturnValue(
        new Promise<RuntimeStatus>((resolve) => {
          resolveRuntime = resolve;
        }),
      );
      // A claimed run can remain queued while its model loads, or in an
      // older queue response. A second submission of the same prompt is
      // a distinct task and must still be shown.
      const queue: QueueState = {
        paused: false,
        pending: [
          pendingRun({ run_id: "run-a", prompt: "same prompt" }),
          pendingRun({ run_id: "run-b", prompt: "same prompt" }),
        ],
      };
      const runtime: RuntimeStatus = {
        implementation: "real",
        state,
        resident: null,
        current_run_id: "run-a",
        queue_depth: 1,
        last_error: null,
      };
      generationsApi.getRun.mockResolvedValue(
        runDetail({
          run_id: "run-a",
          prompt: "same prompt",
          status: state === "generating" ? "running" : "queued",
        }),
      );
      const onFocusRun = vi.fn();
      render(
        <ToastProvider>
          <RuntimeProvider>
            <QueuePanel onFocusRun={onFocusRun} />
          </RuntimeProvider>
        </ToastProvider>,
      );
      const panel = within(screen.getByLabelText("Generation queue"));
      if (order === "queue-first") {
        await act(async () => resolveQueue(queue));
        expect(panel.getAllByRole("listitem")).toHaveLength(2);
        await act(async () => resolveRuntime(runtime));
      } else {
        await act(async () => resolveRuntime(runtime));
        expect(panel.getAllByRole("listitem")).toHaveLength(1);
        await act(async () => resolveQueue(queue));
      }
      await waitFor(() =>
        expect(panel.getAllByRole("listitem")).toHaveLength(2),
      );
      const rows = panel.getAllByRole("listitem");
      expect(within(rows[0]).getByText(label)).toBeInTheDocument();
      expect(within(rows[1]).getByText("Queued")).toBeInTheDocument();
      expect(within(rows[1]).getByText(/#1 ·/)).toBeInTheDocument();
      for (const row of rows)
        fireEvent.click(within(row).getByRole("button", { name: "View" }));
      expect(onFocusRun.mock.calls).toEqual([["run-a"], ["run-b"]]);
    },
  );

  it("renders an empty queue state", async () => {
    renderPanel({ paused: false, pending: [] });

    expect(await screen.findByText("Queue is empty")).toBeInTheDocument();
  });

  it("lists pending runs and cancels them individually", async () => {
    generationsApi.cancelRun.mockResolvedValue(
      runDetail({ run_id: "run-a", status: "cancelled" }),
    );
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

  it.each(["queued", "paused"] as const)(
    "removes a cancelled %s row from the response and filters stale queue reads",
    async (status) => {
      generationsApi.getQueue.mockResolvedValueOnce({
        paused: status === "paused",
        pending: [
          pendingRun({ run_id: "run-a", prompt: "first prompt", status }),
          pendingRun({ run_id: "run-b", prompt: "second prompt", status }),
        ],
      });
      // Every later queue read hangs until released with a payload that left
      // the server before the cancellation and still lists run-a.
      let releaseStale: ((state: QueueState) => void) | undefined;
      const staleRead = new Promise<QueueState>((resolve) => {
        releaseStale = resolve;
      });
      generationsApi.getQueue.mockReturnValue(staleRead);
      generationsApi.cancelRun.mockResolvedValue(
        runDetail({ run_id: "run-a", status: "cancelled" }),
      );
      renderPanelRaw();

      expect(await screen.findByText("first prompt")).toBeInTheDocument();
      fireEvent.click(screen.getAllByRole("button", { name: "Cancel" })[0]);
      await waitFor(() =>
        expect(generationsApi.cancelRun).toHaveBeenCalledWith("run-a"),
      );

      // The terminal response itself removes the row; no queue read has
      // completed yet (the next one is still in flight).
      await waitFor(() =>
        expect(screen.queryByText("first prompt")).toBeNull(),
      );
      expect(screen.getByText("second prompt")).toBeInTheDocument();
      expect(
        screen.getByText(
          status === "paused"
            ? "Paused run cancelled."
            : "Run removed from the queue.",
        ),
      ).toBeInTheDocument();

      // The stale read resolves after the cancellation and must not
      // resurrect the removed row.
      releaseStale!({
        paused: status === "paused",
        pending: [
          pendingRun({ run_id: "run-a", prompt: "first prompt", status }),
          pendingRun({ run_id: "run-b", prompt: "second prompt", status }),
        ],
      });
      await waitFor(() =>
        expect(
          generationsApi.getQueue.mock.calls.length,
        ).toBeGreaterThanOrEqual(2),
      );
      await waitFor(() =>
        expect(screen.queryByText("first prompt")).toBeNull(),
      );
      expect(screen.getByText("second prompt")).toBeInTheDocument();
    },
  );

  it("keeps the row and never claims removal while the cancel response is still active", async () => {
    generationsApi.getQueue.mockResolvedValue({
      paused: false,
      pending: [pendingRun({ run_id: "run-a", prompt: "still queued" })],
    });
    // Dispatch-race shape: the server answered with the run still queued,
    // so nothing has been removed yet.
    generationsApi.cancelRun.mockResolvedValue(
      runDetail({ run_id: "run-a", status: "queued", error: null }),
    );
    renderPanelRaw();

    expect(await screen.findByText("still queued")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "Cancel" })[0]);
    await waitFor(() =>
      expect(generationsApi.cancelRun).toHaveBeenCalledWith("run-a"),
    );

    expect(
      await screen.findByText(
        /Cancellation requested at the next safe boundary/,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("Run removed from the queue.")).toBeNull();
    expect(screen.getByText("still queued")).toBeInTheDocument();
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

  it("reports an already finished run truthfully on the active row", async () => {
    // The cooperative cancellation raced with natural completion: the
    // server answered with the unchanged completed detail.
    generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });
    generationsApi.getRun.mockResolvedValue(
      runDetail({
        run_id: "run-live",
        prompt: "active prompt",
        status: "running",
      }),
    );
    generationsApi.cancelRun.mockResolvedValue(
      runDetail({
        run_id: "run-live",
        prompt: "active prompt",
        status: "completed",
        error: null,
      }),
    );
    renderPanelRaw("run-live");

    expect(await screen.findByText("active prompt")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(generationsApi.cancelRun).toHaveBeenCalledWith("run-live"),
    );

    expect(
      await screen.findByText(/Run already finished \(completed\)/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Run cancelled.")).toBeNull();
    expect(screen.queryByText(/Cancellation requested/i)).toBeNull();
  });

  it("reports an already finished run truthfully on a pending row", async () => {
    generationsApi.getQueue.mockResolvedValueOnce({
      paused: false,
      pending: [pendingRun({ run_id: "run-a", prompt: "late finish" })],
    });
    generationsApi.cancelRun.mockResolvedValue(
      runDetail({
        run_id: "run-a",
        prompt: "late finish",
        status: "completed",
      }),
    );
    renderPanelRaw();

    expect(await screen.findByText("late finish")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "Cancel" })[0]);
    await waitFor(() =>
      expect(generationsApi.cancelRun).toHaveBeenCalledWith("run-a"),
    );

    expect(
      await screen.findByText(/Run already finished \(completed\)/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Run removed from the queue.")).toBeNull();
  });

  it("hides the active row on a terminal cancel response while the runtime is stale", async () => {
    // The runtime poll stays frozen on the old current_run_id; the
    // authoritative terminal response must remove the row anyway and the
    // stale runtime must not resurrect it.
    generationsApi.getQueue.mockResolvedValue({ paused: false, pending: [] });
    generationsApi.getRun.mockResolvedValue(
      runDetail({
        run_id: "run-live",
        prompt: "active prompt",
        status: "running",
      }),
    );
    generationsApi.cancelRun.mockResolvedValue(
      runDetail({ run_id: "run-live", prompt: "active prompt" }),
    );
    renderPanelRaw("run-live");

    expect(await screen.findByText("active prompt")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(generationsApi.cancelRun).toHaveBeenCalledWith("run-live"),
    );

    await waitFor(() => expect(screen.queryByText("active prompt")).toBeNull());
    expect(screen.queryByText("Running")).toBeNull();
    expect(screen.getByText("Run cancelled.")).toBeInTheDocument();
  });
});
