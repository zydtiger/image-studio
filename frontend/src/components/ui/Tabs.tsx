import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";

import { cx } from "../../lib/cx";

export interface TabItem {
  id: string;
  label: ReactNode;
  /** Panel content rendered while this tab is active. */
  content?: ReactNode;
}

export interface TabsProps {
  /** Accessible name describing the tab set's purpose. */
  "aria-label": string;
  tabs: TabItem[];
  activeId: string;
  onChange: (id: string) => void;
  className?: string;
}

/**
 * Keyboard-navigable tabs. Arrow, Home, and End keys both move focus and
 * activate the target tab (automatic activation), matching the WAI-ARIA
 * Tabs pattern.
 */
export function Tabs({
  tabs,
  activeId,
  onChange,
  className,
  "aria-label": ariaLabel,
}: TabsProps) {
  const uid = useId();
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  const handleKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    let next: number;
    switch (event.key) {
      case "ArrowRight":
        next = (index + 1) % tabs.length;
        break;
      case "ArrowLeft":
        next = (index - 1 + tabs.length) % tabs.length;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = tabs.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    const target = tabs[next];
    onChange(target.id);
    tabRefs.current[target.id]?.focus();
  };

  const active = tabs.find((tab) => tab.id === activeId);

  return (
    <div className={cx("tabs", className)}>
      <div role="tablist" aria-label={ariaLabel} className="tab-list">
        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            ref={(element) => {
              tabRefs.current[tab.id] = element;
            }}
            type="button"
            role="tab"
            id={`${uid}-tab-${tab.id}`}
            className="tab"
            aria-selected={tab.id === activeId}
            aria-controls={`${uid}-tabpanel-${tab.id}`}
            tabIndex={tab.id === activeId ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => handleKeyDown(event, index)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {active !== undefined ? (
        <div
          role="tabpanel"
          id={`${uid}-tabpanel-${active.id}`}
          aria-labelledby={`${uid}-tab-${active.id}`}
          className="tab-panel"
          tabIndex={0}
        >
          {active.content}
        </div>
      ) : null}
    </div>
  );
}
