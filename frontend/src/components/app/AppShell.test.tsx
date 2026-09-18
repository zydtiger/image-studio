import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { AppShell } from "./AppShell";
import { ToastProvider } from "../../state/toast/ToastProvider";
import { RuntimeProvider } from "../../state/runtime/RuntimeProvider";

function renderShell() {
  return render(
    <ToastProvider>
      <RuntimeProvider>
        <MemoryRouter initialEntries={["/generate"]}>
          <Routes>
            <Route element={<AppShell />}>
              <Route path="/generate" element={<p>Generate content</p>} />
              <Route path="/history" element={<p>History content</p>} />
            </Route>
          </Routes>
        </MemoryRouter>
      </RuntimeProvider>
    </ToastProvider>,
  );
}

describe("AppShell", () => {
  it("renders the four primary navigation links", () => {
    renderShell();

    for (const label of ["Generate", "Models", "History", "Settings"]) {
      expect(screen.getByRole("link", { name: label })).toBeInTheDocument();
    }
    expect(
      screen.getByRole("navigation", { name: "Primary" }),
    ).toBeInTheDocument();
  });

  it("toggles the mobile menu and closes it with Escape", () => {
    renderShell();

    const open = screen.getByRole("button", { name: "Open navigation menu" });
    expect(open).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(open);
    const close = screen.getByRole("button", {
      name: "Close navigation menu",
    });
    expect(close).toHaveAttribute("aria-expanded", "true");

    fireEvent.keyDown(document, { key: "Escape" });
    expect(
      screen.getByRole("button", { name: "Open navigation menu" }),
    ).toHaveAttribute("aria-expanded", "false");
  });

  it("closes the menu after following a navigation link", () => {
    renderShell();

    fireEvent.click(
      screen.getByRole("button", { name: "Open navigation menu" }),
    );
    fireEvent.click(screen.getByRole("link", { name: "History" }));

    expect(
      screen.getByRole("button", { name: "Open navigation menu" }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("History content")).toBeInTheDocument();
  });

  it("activates the skip link by focusing main without navigating", () => {
    renderShell();

    const skip = screen.getByRole("link", { name: "Skip to main content" });
    expect(skip).toHaveAttribute("href", "#main-content");
    fireEvent.click(skip);

    // The routed content survives and main holds focus; no navigation
    // occurred (the hash is untouched in this MemoryRouter render).
    expect(screen.getByText("Generate content")).toBeInTheDocument();
    expect(document.activeElement).toBe(screen.getByRole("main"));
  });

  it("moves focus to main content on in-app navigation", () => {
    renderShell();

    // Real browsers focus a link on activation; jsdom does not, so focus
    // it explicitly before clicking.
    const link = screen.getByRole("link", { name: "History" });
    link.focus();
    fireEvent.click(link);

    const main = screen.getByRole("main");
    expect(main).toHaveAttribute("tabindex", "-1");
    expect(document.activeElement).toBe(main);
  });

  it("keeps the document focus on initial load", () => {
    renderShell();

    expect(document.activeElement).toBe(document.body);
  });
});
