import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import App from "./App";

beforeEach(() => {
  window.location.hash = "";
});

describe("App", () => {
  it("redirects to Generate by default", async () => {
    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "Generate" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Generate" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("navigates between pages through the nav links", async () => {
    render(<App />);

    await screen.findByRole("heading", { name: "Generate" });
    fireEvent.click(screen.getByRole("link", { name: "History" }));

    expect(
      await screen.findByRole("heading", { name: "History" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "History" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("shows a not-found page for unknown routes", async () => {
    window.location.hash = "#/unknown";
    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "Page not found" }),
    ).toBeInTheDocument();
  });

  it("exposes a skip link as the first focusable element", async () => {
    render(<App />);

    expect(
      screen.getByRole("link", { name: "Skip to main content" }),
    ).toHaveAttribute("href", "#main-content");
  });

  it("shows the runtime banner on every page", async () => {
    render(<App />);

    await screen.findByRole("heading", { name: "Generate" });
    expect(
      screen.getByRole("region", { name: "Runtime status" }),
    ).toBeInTheDocument();
  });
});
