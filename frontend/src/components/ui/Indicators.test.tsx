import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Badge } from "./Badge";
import { EmptyState } from "./EmptyState";
import { ErrorState } from "./ErrorState";
import { ProgressBar } from "./ProgressBar";
import { Spinner } from "./Spinner";

describe("Badge", () => {
  it("renders its tone class and content", () => {
    render(<Badge tone="warning">Partial</Badge>);

    expect(screen.getByText("Partial")).toHaveClass("badge--warning");
  });
});

describe("ProgressBar", () => {
  it("exposes determinate values to assistive technology", () => {
    render(<ProgressBar value={42} label="Downloading model" />);

    expect(
      screen.getByRole("progressbar", { name: "Downloading model" }),
    ).toHaveAttribute("aria-valuenow", "42");
  });

  it("clamps values into the 0-100 range", () => {
    render(<ProgressBar value={150} label="Downloading model" />);

    expect(
      screen.getByRole("progressbar", { name: "Downloading model" }),
    ).toHaveAttribute("aria-valuenow", "100");
  });

  it("renders an indeterminate track without a value", () => {
    render(<ProgressBar label="Downloading model" />);

    const bar = screen.getByRole("progressbar", {
      name: "Downloading model",
    });
    expect(bar).toHaveClass("progress--indeterminate");
    expect(bar).not.toHaveAttribute("aria-valuenow");
  });
});

describe("Spinner", () => {
  it("announces its loading state", () => {
    render(<Spinner label="Loading settings" />);

    expect(screen.getByRole("status")).toHaveTextContent("Loading settings");
  });
});

describe("EmptyState", () => {
  it("renders title, description, and action", () => {
    const onOpen = vi.fn();
    render(
      <EmptyState
        title="No generations yet"
        description="Submit a prompt on the Generate page."
        action={
          <button type="button" onClick={onOpen}>
            Open Generate
          </button>
        }
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Open Generate" }));
    expect(onOpen).toHaveBeenCalledOnce();
    expect(screen.getByText("No generations yet")).toBeInTheDocument();
  });
});

describe("ErrorState", () => {
  it("announces errors and retries", () => {
    const onRetry = vi.fn();
    render(
      <ErrorState
        message="The model list could not be loaded."
        detail="HTTP 500"
        onRetry={onRetry}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent(
      "The model list could not be loaded.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
