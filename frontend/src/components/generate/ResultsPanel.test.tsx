import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RunDetail } from "../../api/types";

const generationsApi = vi.hoisted(() => ({
  getRun: vi.fn(),
}));

vi.mock("../../api/generations", () => ({
  ...generationsApi,
  listRuns: vi.fn(),
  submitGeneration: vi.fn(),
  cancelRun: vi.fn(),
  setFavorite: vi.fn(),
  trashRun: vi.fn(),
  restoreRun: vi.fn(),
  getQueue: vi.fn(),
  resumeQueue: vi.fn(),
  artifactUrl: (runId: string, artifactId: string, download = false) =>
    `/api/generations/${runId}/artifacts/${artifactId}${download ? "?download=1" : ""}`,
  thumbnailUrl: (runId: string, artifactId: string) =>
    `/api/generations/${runId}/artifacts/${artifactId}/thumbnail`,
  metadataUrl: (runId: string) => `/api/generations/${runId}/metadata`,
}));

import { ResultsPanel } from "./ResultsPanel";

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
});

describe("ResultsPanel", () => {
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
