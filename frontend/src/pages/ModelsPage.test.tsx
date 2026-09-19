import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  CachedRepo,
  DownloadJob,
  HubModelDetail,
  ModelRegistration,
} from "../api/types";

const modelsApi = vi.hoisted(() => ({
  listRegistrations: vi.fn(),
  createRegistration: vi.fn(),
  updateRegistration: vi.fn(),
  deleteRegistration: vi.fn(),
}));
const hubApi = vi.hoisted(() => ({
  searchHubModels: vi.fn(),
  getHubModelDetail: vi.fn(),
  getCompatibility: vi.fn(),
}));
const cacheApi = vi.hoisted(() => ({ listCachedRepos: vi.fn() }));
const downloadsApi = vi.hoisted(() => ({
  listDownloads: vi.fn(),
  createDownload: vi.fn(),
  retryDownload: vi.fn(),
  cancelDownload: vi.fn(),
}));
const runtimeApi = vi.hoisted(() => ({
  getRuntime: vi.fn(),
  ejectRuntime: vi.fn(),
}));

vi.mock("../api/models", () => modelsApi);
vi.mock("../api/hub", () => hubApi);
vi.mock("../api/cache", () => cacheApi);
vi.mock("../api/downloads", () => downloadsApi);
vi.mock("../api/runtime", () => runtimeApi);

import ModelsPage from "./ModelsPage";
import { ApiError } from "../api/client";
import { ToastProvider } from "../state/toast/ToastProvider";
import { RuntimeProvider } from "../state/runtime/RuntimeProvider";

const REGISTRATIONS: ModelRegistration[] = [
  {
    id: "reg-1",
    repo_id: "Tongyi-MAI/Z-Image",
    commit_sha: "aaaa1111",
    profile: "z-image",
    display_name: null,
    status: "ready",
    missing_files: [],
    snapshot_path: "/a",
    created_at: "2026-09-16T00:00:00Z",
    last_used_at: null,
  },
  {
    id: "reg-2",
    repo_id: "someone/broken",
    commit_sha: "bbbb2222",
    profile: "z-image-turbo",
    display_name: null,
    status: "missing_files",
    missing_files: ["model_index.json"],
    snapshot_path: "/b",
    created_at: "2026-09-16T00:00:00Z",
    last_used_at: null,
  },
];

function renderModels() {
  modelsApi.listRegistrations.mockResolvedValue(REGISTRATIONS);
  runtimeApi.getRuntime.mockResolvedValue({
    implementation: "real",
    state: "unloaded",
    resident: null,
    current_run_id: null,
    queue_depth: 0,
    last_error: null,
  });
  // Tests set listDownloads / listCachedRepos data themselves.
  downloadsApi.listDownloads.mockResolvedValue([]);
  cacheApi.listCachedRepos.mockResolvedValue([]);
  return render(
    <ToastProvider>
      <RuntimeProvider>
        <MemoryRouter>
          <ModelsPage />
        </MemoryRouter>
      </RuntimeProvider>
    </ToastProvider>,
  );
}

beforeEach(() => {
  modelsApi.listRegistrations.mockReset();
  modelsApi.createRegistration.mockReset();
  modelsApi.deleteRegistration.mockReset();
  hubApi.searchHubModels.mockReset();
  hubApi.getHubModelDetail.mockReset();
  hubApi.getCompatibility.mockReset();
  cacheApi.listCachedRepos.mockReset();
  downloadsApi.listDownloads.mockReset();
  downloadsApi.retryDownload.mockReset();
  downloadsApi.cancelDownload.mockReset();
  runtimeApi.getRuntime.mockReset();
});

describe("ModelsPage", () => {
  it("switches between the four model views", async () => {
    renderModels();

    for (const tab of ["My Models", "Local Cache", "Downloads", "Discover"]) {
      fireEvent.click(screen.getByRole("tab", { name: tab }));
      expect(
        await screen.findByRole("tab", { name: tab, selected: true }),
      ).toBeInTheDocument();
    }
  });

  it("shows a completed download's registration while My Models stays open", async () => {
    renderModels();
    modelsApi.listRegistrations.mockResolvedValue([]);
    fireEvent.click(screen.getByRole("tab", { name: "My Models" }));
    await waitFor(() => expect(modelsApi.listRegistrations).toHaveBeenCalled());
    modelsApi.listRegistrations.mockResolvedValue([REGISTRATIONS[0]]);
    expect(
      await screen.findByText("Tongyi-MAI/Z-Image", {}, { timeout: 3_000 }),
    ).toBeInTheDocument();
    expect(modelsApi.createRegistration).not.toHaveBeenCalled();
  });

  it("marks registrations with missing files", async () => {
    renderModels();

    fireEvent.click(screen.getByRole("tab", { name: "My Models" }));
    expect(await screen.findByText("someone/broken")).toBeInTheDocument();
    expect(screen.getByText("Missing files")).toBeInTheDocument();
    expect(
      screen.getByText(/Missing from the cache: model_index\.json/i),
    ).toBeInTheDocument();
  });

  it("removes a registration only after confirmation", async () => {
    modelsApi.deleteRegistration.mockResolvedValue(undefined);
    renderModels();

    fireEvent.click(screen.getByRole("tab", { name: "My Models" }));
    // Both registrations carry a Remove action; take the first.
    fireEvent.click(
      (await screen.findAllByRole("button", { name: "Remove" }))[0],
    );
    expect(
      screen.getByText(
        /Files in the shared Hugging Face cache are never deleted/i,
      ),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Remove registration" }),
    );
    await waitFor(() =>
      expect(modelsApi.deleteRegistration).toHaveBeenCalledWith("reg-1"),
    );
  });

  it("searches the Hub and opens the compatibility detail", async () => {
    hubApi.searchHubModels.mockResolvedValue([
      {
        repo_id: "Tongyi-MAI/Z-Image",
        author: "Tongyi-MAI",
        gated: false,
        private: false,
        downloads: 1000,
        likes: 50,
        last_modified: null,
        pipeline_tag: null,
        license: "apache-2.0",
      },
    ]);
    hubApi.getHubModelDetail.mockResolvedValue({
      repo_id: "Tongyi-MAI/Z-Image",
      author: "Tongyi-MAI",
      gated: false,
      private: false,
      downloads: 1000,
      likes: 50,
      last_modified: null,
      pipeline_tag: null,
      license: "apache-2.0",
      default_revision: "main",
      revisions: [{ revision: "main", commit_sha: "abcdef123456" }],
      files: [{ path: "model_index.json", size: 512 }],
    } satisfies HubModelDetail);
    hubApi.getCompatibility.mockResolvedValue({
      repo_id: "Tongyi-MAI/Z-Image",
      revision: "main",
      commit_sha: "abcdef123456",
      structurally_compatible: true,
      selectable_profiles: ["z-image", "z-image-turbo"],
      findings: ["Uses the ZImagePipeline class."],
      notes: [],
    });
    renderModels();

    fireEvent.change(await screen.findByLabelText("Search Hugging Face"), {
      target: { value: "z-image" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    fireEvent.click(await screen.findByText("Tongyi-MAI/Z-Image"));
    expect(
      await screen.findByText("Structurally compatible"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Uses the ZImagePipeline class\./i),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Profile")).toHaveValue("z-image");
    expect(screen.getByRole("button", { name: "Download…" })).toBeEnabled();
  });

  function jobsFixture(): DownloadJob[] {
    return [
      {
        id: "dl-1",
        repo_id: "Tongyi-MAI/Z-Image",
        requested_revision: "main",
        resolved_commit: "abcdef",
        profile: "z-image",
        status: "failed",
        error: {
          code: "hub_unreachable",
          message: "The Hub could not be reached.",
          details: null,
        },
        progress: {
          bytes_done: 10,
          bytes_total: 100,
          files_done: 1,
          files_total: 5,
        },
        created_at: "2026-09-16T00:00:00Z",
        started_at: null,
        finished_at: null,
      },
      {
        id: "dl-2",
        repo_id: "someone/other",
        requested_revision: null,
        resolved_commit: null,
        profile: "z-image",
        status: "queued",
        error: null,
        progress: {
          bytes_done: 0,
          bytes_total: null,
          files_done: 0,
          files_total: null,
        },
        created_at: "2026-09-16T00:00:00Z",
        started_at: null,
        finished_at: null,
      },
    ];
  }

  it("offers retry and cancel actions on downloads", async () => {
    downloadsApi.retryDownload.mockResolvedValue({});
    downloadsApi.cancelDownload.mockResolvedValue({});
    renderModels();
    // The panel polls; set job data after renderModels' empty default.
    downloadsApi.listDownloads.mockResolvedValue(jobsFixture());

    fireEvent.click(screen.getByRole("tab", { name: "Downloads" }));
    expect(
      await screen.findByText(/The Hub could not be reached\./i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() =>
      expect(downloadsApi.retryDownload).toHaveBeenCalledWith("dl-1"),
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(downloadsApi.cancelDownload).toHaveBeenCalledWith("dl-2"),
    );
  });

  function cachedFixture(): CachedRepo[] {
    return [
      {
        repo_id: "Tongyi-MAI/Z-Image",
        size_on_disk_bytes: 12345,
        refs: ["main"],
        snapshots: [
          {
            commit_sha: "abcdef123456",
            path: "/cache/snap",
            size_bytes: 12345,
            incomplete: false,
          },
        ],
      },
    ];
  }

  it("shows the component validation reason when cache registration fails", async () => {
    renderModels();
    cacheApi.listCachedRepos.mockResolvedValue(cachedFixture());
    modelsApi.createRegistration.mockRejectedValue(
      new ApiError("cached model does not satisfy its component manifest", {
        kind: "http",
        status: 422,
        details: { problems: ["Anima component recipe mismatch"] },
      }),
    );
    fireEvent.click(screen.getByRole("tab", { name: "Local Cache" }));
    fireEvent.click(
      (await screen.findAllByRole("button", { name: "Register" }))[0],
    );
    fireEvent.click(
      (await screen.findAllByRole("button", { name: "Register" }))[1],
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Anima component recipe mismatch",
    );
  });

  it("lists cached snapshots and registers them", async () => {
    modelsApi.createRegistration.mockResolvedValue(REGISTRATIONS[0]);
    renderModels();
    // The panel queries once mounted; set cache data after the default.
    cacheApi.listCachedRepos.mockResolvedValue(cachedFixture());

    fireEvent.click(screen.getByRole("tab", { name: "Local Cache" }));
    // The snapshot row and the dialog footer both carry the label.
    fireEvent.click(
      (await screen.findAllByRole("button", { name: "Register" }))[0],
    );
    fireEvent.click(
      (await screen.findAllByRole("button", { name: "Register" }))[1],
    );

    await waitFor(() =>
      expect(modelsApi.createRegistration).toHaveBeenCalledWith(
        expect.objectContaining({
          repo_id: "Tongyi-MAI/Z-Image",
          revision: "abcdef123456",
          profile: "z-image",
        }),
      ),
    );
  });
});
