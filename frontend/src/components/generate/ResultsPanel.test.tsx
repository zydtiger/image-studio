import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RunDetail, RunSummary } from "../../api/types";

const generationsApi = vi.hoisted(() => ({
  getRun: vi.fn(),
  setFavorite: vi.fn(),
  trashRun: vi.fn(),
}));

vi.mock("../../api/generations", () => ({
  ...generationsApi,
  listRuns: vi.fn(),
  submitGeneration: vi.fn(),
  cancelRun: vi.fn(),
  restoreRun: vi.fn(),
  getQueue: vi.fn(),
  resumeQueue: vi.fn(),
  artifactUrl: (runId: string, artifactId: string, download = false) =>
    `/api/generations/${runId}/artifacts/${artifactId}${download ? "?download=1" : ""}`,
  thumbnailUrl: (runId: string, artifactId: string) =>
    `/api/generations/${runId}/artifacts/${artifactId}/thumbnail`,
  metadataUrl: (runId: string) => `/api/generations/${runId}/metadata`,
}));

vi.mock("../../state/toast/context", () => ({
  useToast: () => ({ pushToast: vi.fn() }),
}));

import { ResultsPanel, type ModelRunsView } from "./ResultsPanel";

function run(overrides: Partial<RunDetail>): RunDetail {
  return {
    run_id: "run-1234abcdef",
    created_at: "2026-09-16T00:00:00Z",
    started_at: "2026-09-16T00:00:05Z",
    finished_at: null,
    status: "running",
    favorite: false,
    trashed: false,
    registration_id: "reg-1",
    repo_id: "Tongyi-MAI/Z-Image",
    commit_sha: "abcdef123456",
    profile: "z-image",
    dtype: "bfloat16",
    gpu: { uuid: "gpu-0", name: "RTX A" },
    prompt: "a quiet harbor",
    negative_prompt: null,
    width: 1024,
    height: 1024,
    steps: 9,
    guidance: 0,
    initial_seed: 100,
    image_count: 2,
    pipeline_class: "ZImagePipeline",
    dependency_versions: { torch: "2.9.0" },
    runtime_meta: {},
    queue_position: null,
    progress: null,
    error: null,
    images: [],
    ...overrides,
  };
}

beforeEach(() => {
  generationsApi.getRun.mockReset();
  generationsApi.setFavorite.mockReset();
  generationsApi.trashRun.mockReset();
});

describe("ResultsPanel", () => {
  it("keeps a saved favorite when an earlier polling read arrives late", async () => {
    let resolveRead!: (detail: RunDetail) => void;
    generationsApi.getRun
      .mockResolvedValueOnce(run({}))
      .mockImplementationOnce(
        () =>
          new Promise<RunDetail>((resolve) => {
            resolveRead = resolve;
          }),
      )
      .mockResolvedValue(run({ favorite: true }));
    generationsApi.setFavorite.mockResolvedValue({
      ...run({}),
      favorite: true,
    });
    render(
      <MemoryRouter>
        <ResultsPanel runId="run-1234abcdef" onClear={vi.fn()} />
      </MemoryRouter>,
    );
    await waitFor(
      () => expect(generationsApi.getRun).toHaveBeenCalledTimes(2),
      { timeout: 2000 },
    );
    fireEvent.click(screen.getByRole("button", { name: "Mark favorite" }));
    await screen.findByRole("button", { name: "Remove favorite" });
    await act(async () => resolveRead(run({ favorite: false })));
    expect(
      screen.getByRole("button", { name: "Remove favorite" }),
    ).toHaveAttribute("aria-pressed", "true");
  });
  it("renders the empty state before any run is followed", () => {
    render(<ResultsPanel runId={null} onClear={vi.fn()} />);

    expect(screen.getByText("No run selected")).toBeInTheDocument();
    expect(generationsApi.getRun).not.toHaveBeenCalled();
  });

  it("shows running progress with accessible values", async () => {
    generationsApi.getRun.mockResolvedValue(
      run({
        status: "running",
        progress: { image_index: 1, step: 3, total_steps: 9 },
      }),
    );

    render(
      <MemoryRouter>
        <ResultsPanel runId="run-1234abcdef" onClear={vi.fn()} />
      </MemoryRouter>,
    );

    expect(
      await screen.findByRole("progressbar", { name: /image 1 of 2/i }),
    ).toHaveAttribute("aria-valuenow", "17");
    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("renders completed images with their seeds and a download link", async () => {
    generationsApi.getRun.mockResolvedValue(
      run({
        status: "completed",
        finished_at: "2026-09-16T00:00:20Z",
        images: [
          {
            artifact_id: "image-001",
            index: 1,
            seed: 100,
            status: "completed",
            width: 1024,
            height: 1024,
            size_bytes: 1024,
            error: null,
            url: "/api/generations/run-1234abcdef/artifacts/image-001",
            thumbnail_url:
              "/api/generations/run-1234abcdef/artifacts/image-001/thumbnail",
          },
          {
            artifact_id: "image-002",
            index: 2,
            seed: 101,
            status: "completed",
            width: 1024,
            height: 1024,
            size_bytes: 1024,
            error: null,
            url: "/api/generations/run-1234abcdef/artifacts/image-002",
            thumbnail_url:
              "/api/generations/run-1234abcdef/artifacts/image-002/thumbnail",
          },
        ],
      }),
    );

    render(
      <MemoryRouter>
        <ResultsPanel runId="run-1234abcdef" onClear={vi.fn()} />
      </MemoryRouter>,
    );

    expect((await screen.findAllByText("seed 100")).length).toBeGreaterThan(0);
    expect(screen.getByText("seed 101")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /Open image 1, seed 100/i }),
    );
    const download = await screen.findByRole("link", {
      name: "Download PNG",
    });
    expect(download).toHaveAttribute(
      "href",
      "/api/generations/run-1234abcdef/artifacts/image-001?download=1",
    );
  });

  it("keeps partial and failed images visible with clear states", async () => {
    generationsApi.getRun.mockResolvedValue(
      run({
        status: "partial",
        error: {
          code: "worker_error",
          message: "CUDA out of memory on image 2.",
          details: null,
        },
        images: [
          {
            artifact_id: "image-001",
            index: 1,
            seed: 100,
            status: "completed",
            width: 1024,
            height: 1024,
            size_bytes: 1024,
            error: null,
            url: "/api/generations/r/artifacts/image-001",
            thumbnail_url: "/api/generations/r/artifacts/image-001/thumbnail",
          },
          {
            artifact_id: "image-002",
            index: 2,
            seed: 101,
            status: "failed",
            width: null,
            height: null,
            size_bytes: null,
            error: "OOM",
            url: null,
            thumbnail_url: null,
          },
        ],
      }),
    );

    render(
      <MemoryRouter>
        <ResultsPanel runId="run-1234abcdef" onClear={vi.fn()} />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Partial")).toBeInTheDocument();
    expect(screen.getByText(/CUDA out of memory/i)).toBeInTheDocument();
    expect(screen.getByText("OOM")).toBeInTheDocument();
    expect(screen.getByText(/1\/2 images/)).toBeInTheDocument();
  });

  it("replaces a terminal run when the followed run id changes", async () => {
    generationsApi.getRun.mockResolvedValueOnce(
      run({
        run_id: "run-aaaa0000",
        status: "completed",
        finished_at: "2026-09-16T00:00:20Z",
        prompt: "first prompt",
      }),
    );
    const { rerender } = render(
      <MemoryRouter>
        <ResultsPanel runId="run-aaaa" onClear={vi.fn()} />
      </MemoryRouter>,
    );
    expect(await screen.findByText(/run run-aaaa/i)).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();

    generationsApi.getRun.mockResolvedValueOnce(
      run({
        run_id: "run-bbbb0000",
        status: "running",
        prompt: "second prompt",
        progress: { image_index: 1, step: 3, total_steps: 9 },
      }),
    );
    rerender(
      <MemoryRouter>
        <ResultsPanel runId="run-bbbb" onClear={vi.fn()} />
      </MemoryRouter>,
    );

    await waitFor(
      () => expect(screen.getByText(/run run-bbbb/i)).toBeInTheDocument(),
      { timeout: 4_000 },
    );
    expect(screen.getByText("Running")).toBeInTheDocument();
    expect(screen.queryByText(/run run-aaaa/i)).toBeNull();
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-bbbb",
        expect.anything(),
      ),
    );
  });

  it("applies a pushed cancel record and ignores a stale in-flight read", async () => {
    // The first read hangs until released: it left the server before the
    // cancellation and still reports the run as queued.
    let releaseStale: ((detail: RunDetail) => void) | undefined;
    generationsApi.getRun.mockImplementationOnce(
      () =>
        new Promise<RunDetail>((resolve) => {
          releaseStale = resolve;
        }),
    );
    const view = (update?: { seq: number; run: RunDetail }) => (
      <MemoryRouter>
        <ResultsPanel
          runId="run-1234abcdef"
          onClear={vi.fn()}
          runUpdate={update ?? null}
        />
      </MemoryRouter>
    );
    const { rerender } = render(view());

    // The queue's authoritative terminal response arrives while the read
    // is still in flight and applies at once.
    rerender(
      view({
        seq: 1,
        run: run({
          status: "cancelled",
          finished_at: "2026-09-16T00:00:20Z",
          error: null,
        }),
      }),
    );
    expect(await screen.findByText("Cancelled")).toBeInTheDocument();

    // The stale read resolves afterwards; it must not resurrect the run.
    releaseStale?.(run({ status: "queued", started_at: null }));
    await waitFor(() => expect(generationsApi.getRun).toHaveBeenCalledOnce());
    expect(screen.getByText("Cancelled")).toBeInTheDocument();
    expect(screen.queryByText("Queued")).toBeNull();
  });

  it("ignores a pushed record for a run other than the followed one", async () => {
    generationsApi.getRun.mockResolvedValue(
      run({ status: "queued", started_at: null }),
    );
    const view = (update?: { seq: number; run: RunDetail }) => (
      <MemoryRouter>
        <ResultsPanel
          runId="run-1234abcdef"
          onClear={vi.fn()}
          runUpdate={update ?? null}
        />
      </MemoryRouter>
    );
    const { rerender } = render(view());
    expect(await screen.findByText("Queued")).toBeInTheDocument();

    rerender(
      view({
        seq: 1,
        run: run({
          run_id: "run-other0000",
          status: "cancelled",
          finished_at: "2026-09-16T00:00:20Z",
        }),
      }),
    );

    expect(screen.getByText("Queued")).toBeInTheDocument();
    expect(screen.queryByText("Cancelled")).toBeNull();
  });

  it("does not revert a terminal observation for a delayed still-active cancel response", async () => {
    // The GET poll already observed the terminal cancellation; a slow
    // cooperative cancel response captured earlier still says "running"
    // and must not revert the followed run.
    generationsApi.getRun.mockResolvedValueOnce(
      run({
        status: "cancelled",
        finished_at: "2026-09-16T00:00:20Z",
        error: null,
      }),
    );
    generationsApi.getRun.mockReturnValue(
      new Promise<RunDetail>(() => undefined),
    );
    const view = (update?: { seq: number; run: RunDetail }) => (
      <MemoryRouter>
        <ResultsPanel
          runId="run-1234abcdef"
          onClear={vi.fn()}
          runUpdate={update ?? null}
        />
      </MemoryRouter>
    );
    const { rerender } = render(view());
    expect(await screen.findByText("Cancelled")).toBeInTheDocument();

    rerender(
      view({ seq: 1, run: run({ status: "running", finished_at: null }) }),
    );

    expect(screen.getByText("Cancelled")).toBeInTheDocument();
    expect(screen.queryByText("Running")).toBeNull();
  });

  it("never lets a straggler read for a previous run clobber the current record", async () => {
    // Run A's read resolves just after its abort raced with completion,
    // while run B is already followed and loaded.
    let resolveA: ((detail: RunDetail) => void) | undefined;
    generationsApi.getRun.mockImplementationOnce(
      (runId: string) =>
        new Promise<RunDetail>((resolve) => {
          resolveA = resolve;
          void runId;
        }),
    );
    generationsApi.getRun.mockResolvedValueOnce(
      run({ run_id: "run-bbbb0000", status: "completed" }),
    );
    const view = (id: string | null) => (
      <MemoryRouter>
        <ResultsPanel runId={id} onClear={vi.fn()} runUpdate={null} />
      </MemoryRouter>
    );
    const { rerender } = render(view("run-aaaa"));
    rerender(view("run-bbbb"));
    await waitFor(
      () => expect(screen.getByText(/run run-bbbb/i)).toBeInTheDocument(),
      { timeout: 4_000 },
    );

    resolveA?.(run({ run_id: "run-aaaa0000", status: "running" }));

    await waitFor(() => expect(generationsApi.getRun).toHaveBeenCalledTimes(2));
    expect(screen.getByText(/run run-bbbb/i)).toBeInTheDocument();
    expect(screen.queryByText("Loading run")).toBeNull();
  });

  it("ignores a late response for a previous run after switching", async () => {
    let resolveFirst: (detail: RunDetail) => void = () => undefined;
    generationsApi.getRun.mockImplementationOnce(
      (runId: string) =>
        new Promise<RunDetail>((resolve) => {
          resolveFirst = (detail) => resolve(detail);
          void runId;
        }),
    );
    generationsApi.getRun.mockResolvedValue(
      run({ run_id: "run-bbbb0000", status: "running" }),
    );

    const { rerender } = render(
      <MemoryRouter>
        <ResultsPanel runId="run-aaaa" onClear={vi.fn()} />
      </MemoryRouter>,
    );
    rerender(
      <MemoryRouter>
        <ResultsPanel runId="run-bbbb" onClear={vi.fn()} />
      </MemoryRouter>,
    );

    // The switch re-keys the schedule: run B is fetched at once, while
    // run A's request is still pending.
    await waitFor(
      () =>
        expect(generationsApi.getRun).toHaveBeenCalledWith(
          "run-bbbb",
          expect.anything(),
        ),
      { timeout: 4_000 },
    );

    // The stale response for run A lands after the switch to run B.
    resolveFirst(run({ status: "completed", prompt: "stale first prompt" }));

    await waitFor(
      () => expect(screen.getByText(/run run-bbbb/i)).toBeInTheDocument(),
      { timeout: 4_000 },
    );
    expect(screen.queryByText(/run run-aaaa/i)).toBeNull();
  });

  it("clears back to the empty state and follows a fresh run afterwards", async () => {
    generationsApi.getRun.mockResolvedValue(
      run({ run_id: "run-aaaa0000", status: "completed" }),
    );
    const view = (id: string | null) => (
      <MemoryRouter>
        <ResultsPanel runId={id} onClear={vi.fn()} />
      </MemoryRouter>
    );
    const { rerender } = render(view("run-aaaa"));
    expect(await screen.findByText(/run run-aaaa/i)).toBeInTheDocument();

    rerender(view(null));
    expect(screen.getByText("No run selected")).toBeInTheDocument();

    generationsApi.getRun.mockImplementationOnce(() =>
      Promise.resolve(run({ run_id: "run-cccc0000" })),
    );
    rerender(view("run-cccc"));
    await waitFor(() =>
      expect(generationsApi.getRun).toHaveBeenCalledWith(
        "run-cccc",
        expect.anything(),
      ),
    );
    expect(await screen.findByText(/run run-cccc/i)).toBeInTheDocument();
  });
});

function listSummary(overrides: Partial<RunSummary>): RunSummary {
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

describe("ResultsPanel model run list", () => {
  const runs: RunSummary[] = [
    listSummary({ run_id: "run-new1", prompt: "newest run" }),
    listSummary({
      run_id: "run-old1",
      prompt: "older run",
      status: "partial",
      completed_count: 1,
      created_at: new Date(Date.now() - 3_600_000).toISOString(),
    }),
  ];
  const modelRuns: ModelRunsView = {
    repoId: "Tongyi-MAI/Z-Image",
    runs,
    total: 5,
  };

  it("renders the model's runs with selection state and load more", () => {
    const onSelectRun = vi.fn();
    const onLoadMoreModelRuns = vi.fn();
    const { container } = render(
      <MemoryRouter>
        <ResultsPanel
          runId={null}
          onClear={vi.fn()}
          modelRuns={modelRuns}
          onSelectRun={onSelectRun}
          onLoadMoreModelRuns={onLoadMoreModelRuns}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText("Tongyi-MAI/Z-Image")).toBeInTheDocument();
    expect(
      screen.getByRole("list", { name: "Recent runs of this model" }),
    ).toBeInTheDocument();
    // Chips stay real buttons inside list items: button semantics and
    // aria-pressed remain valid for assistive tech and keyboard use.
    const chips = screen.getAllByRole("button", { name: /View run from/ });
    expect(chips).toHaveLength(2);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    const newest = screen.getByTitle("newest run");
    expect(newest).toHaveAttribute("aria-pressed", "false");
    // Partial runs with completed images stay listed with their preview.
    expect(screen.getByTitle("older run")).toBeInTheDocument();
    expect(container.querySelector("img")).toHaveAttribute(
      "src",
      "/api/generations/run-new1/artifacts/image-001/thumbnail",
    );

    fireEvent.click(newest);
    expect(onSelectRun).toHaveBeenCalledWith("run-new1");

    const more = screen.getByRole("button", { name: /load more/i });
    expect(more).toHaveTextContent("Load more (2 of 5)");
    fireEvent.click(more);
    expect(onLoadMoreModelRuns).toHaveBeenCalledOnce();
  });

  it("marks the followed run in the list", () => {
    render(
      <MemoryRouter>
        <ResultsPanel
          runId="run-old1"
          onClear={vi.fn()}
          modelRuns={modelRuns}
        />
      </MemoryRouter>,
    );

    expect(screen.getByTitle("older run")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTitle("newest run")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("shows a spinner while the model's runs load and no empty-state stack", () => {
    render(
      <MemoryRouter>
        <ResultsPanel runId={null} onClear={vi.fn()} modelRunsLoading={true} />
      </MemoryRouter>,
    );

    expect(screen.getByText("Loading runs")).toBeInTheDocument();
    expect(screen.queryByText("No run selected")).toBeNull();
  });

  it("shows an error with retry when the run list fails", () => {
    const onRetryModelRuns = vi.fn();
    render(
      <MemoryRouter>
        <ResultsPanel
          runId={null}
          onClear={vi.fn()}
          modelRunsError={new Error("network down")}
          onRetryModelRuns={onRetryModelRuns}
        />
      </MemoryRouter>,
    );

    expect(
      screen.getByText("Runs of this model could not be loaded."),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetryModelRuns).toHaveBeenCalledOnce();
  });

  it("shows a single empty state when the model has no runs", () => {
    render(
      <MemoryRouter>
        <ResultsPanel
          runId={null}
          onClear={vi.fn()}
          modelRuns={{ repoId: "Tongyi-MAI/Z-Image", runs: [], total: 0 }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText("No results yet")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Runs with completed images from Tongyi-MAI/Z-Image appear here.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("No run selected")).toBeNull();
  });

  it("hints at picking a listed run when none is selected", () => {
    render(
      <MemoryRouter>
        <ResultsPanel runId={null} onClear={vi.fn()} modelRuns={modelRuns} />
      </MemoryRouter>,
    );

    expect(screen.getByText("No run selected")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Pick a run from the list above, or submit a generation to follow it here.",
      ),
    ).toBeInTheDocument();
  });
});
