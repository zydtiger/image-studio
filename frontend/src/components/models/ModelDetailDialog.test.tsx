import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CompatibilityReport, HubModelDetail } from "../../api/types";

const hubApi = vi.hoisted(() => ({
  getHubModelDetail: vi.fn(),
  getCompatibility: vi.fn(),
}));
const modelsApi = vi.hoisted(() => ({ createRegistration: vi.fn() }));
const downloadsApi = vi.hoisted(() => ({ createDownload: vi.fn() }));

vi.mock("../../api/hub", () => hubApi);
vi.mock("../../api/models", () => modelsApi);
vi.mock("../../api/downloads", () => downloadsApi);

import { ModelDetailDialog } from "./ModelDetailDialog";
import { ToastProvider } from "../../state/toast/ToastProvider";

const COMMIT = "a".repeat(40);
const OTHER_COMMIT = "b".repeat(40);

function detail(): HubModelDetail {
  return {
    repo_id: "Tongyi-MAI/Z-Image",
    author: "Tongyi-MAI",
    private: false,
    gated: false,
    downloads: 1,
    likes: 2,
    last_modified: null,
    pipeline_tag: null,
    license: null,
    default_revision: "main",
    revisions: [
      { revision: "main", commit_sha: COMMIT },
      { revision: "dev", commit_sha: OTHER_COMMIT },
    ],
    files: [],
  };
}

function report(revision: string, commit: string): CompatibilityReport {
  return {
    repo_id: "Tongyi-MAI/Z-Image",
    revision,
    commit_sha: commit,
    structurally_compatible: true,
    selectable_profiles: ["z-image", "z-image-turbo"],
    findings: [],
    notes: [],
  };
}

function renderDialog() {
  return render(
    <ToastProvider>
      <ModelDetailDialog
        repoId="Tongyi-MAI/Z-Image"
        onClose={() => {}}
        onDownloaded={() => {}}
        onRegistered={() => {}}
      />
    </ToastProvider>,
  );
}

async function settledDetail() {
  await waitFor(() => expect(hubApi.getCompatibility).toHaveBeenCalled());
}

beforeEach(() => {
  hubApi.getHubModelDetail.mockReset();
  hubApi.getCompatibility.mockReset();
  modelsApi.createRegistration.mockReset();
  downloadsApi.createDownload.mockReset();
  modelsApi.createRegistration.mockResolvedValue({});
  hubApi.getHubModelDetail.mockResolvedValue(detail());
});

describe("ModelDetailDialog register-by-commit", () => {
  it("registers using the compatibility report's resolved commit, not the branch name", async () => {
    hubApi.getCompatibility.mockResolvedValue(report("main", COMMIT));
    renderDialog();
    await settledDetail();

    fireEvent.click(screen.getByRole("button", { name: /Register/ }));
    await waitFor(() =>
      expect(modelsApi.createRegistration).toHaveBeenCalled(),
    );
    expect(modelsApi.createRegistration).toHaveBeenCalledWith({
      repo_id: "Tongyi-MAI/Z-Image",
      revision: COMMIT,
      profile: "z-image",
    });
    expect(modelsApi.createRegistration.mock.calls[0][0].revision).not.toBe(
      "main",
    );
  });

  it("keeps register disabled while the compatibility report is loading", async () => {
    let release!: (value: CompatibilityReport) => void;
    hubApi.getCompatibility.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      }),
    );
    renderDialog();
    await waitFor(() =>
      expect(screen.getByText("Tongyi-MAI/Z-Image")).toBeInTheDocument(),
    );

    const register = screen.getByRole("button", { name: /Register/ });
    expect(register).toBeDisabled();
    fireEvent.click(register);
    expect(modelsApi.createRegistration).not.toHaveBeenCalled();

    release(report("main", COMMIT));
    await waitFor(() => expect(register).toBeEnabled());
  });

  it("keeps register disabled while a stale report belongs to a different revision", async () => {
    // The report for "main" resolves instantly; after switching to "dev"
    // the new report stays pending, so the stale main report must not
    // authorize registering dev's snapshot.
    hubApi.getCompatibility.mockImplementation((_repo: string, rev?: string) =>
      rev === "main"
        ? Promise.resolve(report("main", COMMIT))
        : new Promise(() => {}),
    );
    renderDialog();
    await settledDetail();
    const register = screen.getByRole("button", { name: /Register/ });
    await waitFor(() => expect(register).toBeEnabled());

    fireEvent.change(screen.getByLabelText("Revision"), {
      target: { value: "dev" },
    });

    expect(register).toBeDisabled();
    fireEvent.click(register);
    expect(modelsApi.createRegistration).not.toHaveBeenCalled();
  });

  it("registers the switched revision's commit once its report arrives", async () => {
    hubApi.getCompatibility.mockImplementation((_repo: string, rev?: string) =>
      Promise.resolve(
        rev === "dev" ? report("dev", OTHER_COMMIT) : report("main", COMMIT),
      ),
    );
    renderDialog();
    await settledDetail();

    fireEvent.change(screen.getByLabelText("Revision"), {
      target: { value: "dev" },
    });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Register/ })).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /Register/ }));
    await waitFor(() =>
      expect(modelsApi.createRegistration).toHaveBeenCalled(),
    );
    expect(modelsApi.createRegistration).toHaveBeenCalledWith(
      expect.objectContaining({ revision: OTHER_COMMIT }),
    );
  });
});
