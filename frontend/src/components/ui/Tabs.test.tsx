import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Tabs, type TabItem } from "./Tabs";

const tabs: TabItem[] = [
  { id: "discover", label: "Discover", content: "Discover panel" },
  { id: "library", label: "My Models", content: "Library panel" },
  { id: "cache", label: "Local Cache", content: "Cache panel" },
];

function renderTabs(activeId: string, onChange: (id: string) => void) {
  render(
    <Tabs
      aria-label="Model views"
      tabs={tabs}
      activeId={activeId}
      onChange={onChange}
    />,
  );
}

describe("Tabs", () => {
  it("renders the active tab and its panel", () => {
    renderTabs("library", vi.fn());

    expect(screen.getByRole("tab", { name: "My Models" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "Discover" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(screen.getByRole("tabpanel")).toHaveTextContent("Library panel");
  });

  it("keeps inactive tabs out of the tab order", () => {
    renderTabs("library", vi.fn());

    expect(screen.getByRole("tab", { name: "My Models" })).toHaveAttribute(
      "tabindex",
      "0",
    );
    expect(screen.getByRole("tab", { name: "Discover" })).toHaveAttribute(
      "tabindex",
      "-1",
    );
  });

  it("switches tabs on click", () => {
    const onChange = vi.fn();
    renderTabs("discover", onChange);

    fireEvent.click(screen.getByRole("tab", { name: "Local Cache" }));

    expect(onChange).toHaveBeenCalledWith("cache");
  });

  it("activates and focuses the next tab on ArrowRight", () => {
    const onChange = vi.fn();
    renderTabs("discover", onChange);

    const discover = screen.getByRole("tab", { name: "Discover" });
    discover.focus();
    fireEvent.keyDown(discover, { key: "ArrowRight" });

    expect(onChange).toHaveBeenCalledWith("library");
    expect(document.activeElement).toBe(
      screen.getByRole("tab", { name: "My Models" }),
    );
  });

  it("wraps ArrowLeft from the first tab to the last", () => {
    const onChange = vi.fn();
    renderTabs("discover", onChange);

    fireEvent.keyDown(screen.getByRole("tab", { name: "Discover" }), {
      key: "ArrowLeft",
    });

    expect(onChange).toHaveBeenCalledWith("cache");
  });

  it("supports Home and End keys", () => {
    const onChange = vi.fn();
    renderTabs("library", onChange);

    fireEvent.keyDown(screen.getByRole("tab", { name: "My Models" }), {
      key: "End",
    });
    expect(onChange).toHaveBeenCalledWith("cache");

    fireEvent.keyDown(screen.getByRole("tab", { name: "My Models" }), {
      key: "Home",
    });
    expect(onChange).toHaveBeenCalledWith("discover");
  });
});
