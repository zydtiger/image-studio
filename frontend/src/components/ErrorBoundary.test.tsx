import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ErrorBoundary } from "./ErrorBoundary";

let shouldThrow = false;

function Volatile() {
  if (shouldThrow) {
    throw new Error("kaboom");
  }
  return <p>Fine</p>;
}

function Harness() {
  const [generation, setGeneration] = useState(0);
  return (
    <>
      <ErrorBoundary resetKey={`gen-${generation}`}>
        <Volatile />
      </ErrorBoundary>
      <button type="button" onClick={() => setGeneration(1)}>
        Change key
      </button>
    </>
  );
}

afterEach(() => {
  shouldThrow = false;
  vi.restoreAllMocks();
});

describe("ErrorBoundary", () => {
  it("renders children when nothing throws", () => {
    render(
      <ErrorBoundary>
        <p>Fine</p>
      </ErrorBoundary>,
    );

    expect(screen.getByText("Fine")).toBeInTheDocument();
  });

  it("shows the fallback and recovers when the reset key changes", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    shouldThrow = true;

    render(<Harness />);

    expect(screen.getByText("Unexpected interface error")).toBeInTheDocument();
    expect(screen.getByText("kaboom")).toBeInTheDocument();

    shouldThrow = false;
    fireEvent.click(screen.getByRole("button", { name: "Change key" }));

    expect(screen.getByText("Fine")).toBeInTheDocument();
    expect(screen.queryByText("Unexpected interface error")).toBeNull();
  });

  it("supports a custom fallback", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    shouldThrow = true;

    render(
      <ErrorBoundary
        fallback={(error, reset) => (
          <button type="button" onClick={reset}>
            Recover: {error.message}
          </button>
        )}
      >
        <Volatile />
      </ErrorBoundary>,
    );

    expect(
      screen.getByRole("button", { name: "Recover: kaboom" }),
    ).toBeInTheDocument();
  });
});
