import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { RunSummary } from "../../api/types";
import { RunCard } from "./RunCard";

function summary(overrides: Partial<RunSummary>): RunSummary {
  return {
    run_id: "run-1",
    created_at: "2026-09-16T00:00:00Z",
    status: "completed",
    favorite: false,
    trashed: false,
    prompt: "a quiet harbor at dawn",
    negative_prompt: null,
    repo_id: "Tongyi-MAI/Z-Image",
    profile: "z-image",
    image_count: 2,
    completed_count: 2,
    preview_artifact_id: "image-001",
    ...overrides,
  };
}

describe("RunCard", () => {
  it("renders the thumbnail, status, and prompt", () => {
    render(
      <RunCard run={summary({})} onOpen={vi.fn()} onToggleFavorite={vi.fn()} />,
    );

    const image = screen.getByAltText("") as HTMLImageElement;
    expect(image).toHaveAttribute(
      "src",
      "/api/generations/run-1/artifacts/image-001/thumbnail",
    );
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.getByText("a quiet harbor at dawn")).toBeInTheDocument();
  });

  it("falls back to a count placeholder without a preview", () => {
    render(
      <RunCard
        run={summary({
          preview_artifact_id: null,
          completed_count: 1,
          status: "partial",
        })}
        onOpen={vi.fn()}
        onToggleFavorite={vi.fn()}
      />,
    );

    expect(screen.getByText("1/2 images")).toBeInTheDocument();
    expect(screen.getByText("Partial")).toBeInTheDocument();
  });

  it("toggles favorites", () => {
    const onToggleFavorite = vi.fn();
    render(
      <RunCard
        run={summary({ favorite: true })}
        onOpen={vi.fn()}
        onToggleFavorite={onToggleFavorite}
      />,
    );

    const star = screen.getByRole("button", { name: "Remove favorite" });
    expect(star).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(star);
    expect(onToggleFavorite).toHaveBeenCalledWith(false);
  });
});
