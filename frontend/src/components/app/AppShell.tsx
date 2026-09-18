import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router";

import { useRouteFocus } from "../../hooks/useRouteFocus";
import { RuntimeBanner } from "./RuntimeBanner";
import { CloseIcon, MenuIcon } from "../ui/icons";
import {
  GenerateIcon,
  HistoryIcon,
  ModelsIcon,
  SettingsIcon,
} from "./nav-icons";

const NAV_ITEMS = [
  { to: "/generate", label: "Generate", Icon: GenerateIcon },
  { to: "/models", label: "Models", Icon: ModelsIcon },
  { to: "/history", label: "History", Icon: HistoryIcon },
  { to: "/settings", label: "Settings", Icon: SettingsIcon },
] as const;

/**
 * Application frame: skip link, primary navigation, and routed main
 * content. On narrow viewports the sidebar becomes an off-canvas menu
 * toggled from the top bar and closable with Escape, the scrim, or a
 * navigation event.
 */
export function AppShell() {
  const [navOpen, setNavOpen] = useState(false);
  const mainRef = useRouteFocus<HTMLElement>();

  useEffect(() => {
    if (!navOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNavOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [navOpen]);

  const onSkip = (event: React.MouseEvent<HTMLAnchorElement>) => {
    // HashRouter would parse "#main-content" as a route and land on the
    // not-found page; activate the skip target without touching the URL.
    event.preventDefault();
    mainRef.current?.focus();
    // Optional call: jsdom does not implement scrollIntoView.
    mainRef.current?.scrollIntoView?.({ block: "start" });
  };

  return (
    <div className="app-shell">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(event) => onSkip(event)}
      >
        Skip to main content
      </a>
      <header className="app-topbar">
        <button
          type="button"
          className="icon-button"
          aria-expanded={navOpen}
          aria-controls="app-sidebar"
          aria-label={
            navOpen ? "Close navigation menu" : "Open navigation menu"
          }
          onClick={() => setNavOpen((open) => !open)}
        >
          {navOpen ? <CloseIcon /> : <MenuIcon />}
        </button>
        <span className="app-brand app-brand--compact">Image Studio</span>
      </header>
      <nav
        id="app-sidebar"
        className="app-sidebar"
        aria-label="Primary"
        data-open={navOpen}
      >
        <span className="app-brand">Image Studio</span>
        <ul className="app-nav">
          {NAV_ITEMS.map(({ to, label, Icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                className="nav-item"
                onClick={() => setNavOpen(false)}
              >
                <Icon className="nav-icon" />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
        <p className="app-sidebar-footer">Local image generation</p>
      </nav>
      <div
        className="app-nav-scrim"
        data-open={navOpen}
        aria-hidden="true"
        onClick={() => setNavOpen(false)}
      />
      <main ref={mainRef} id="main-content" className="app-main" tabIndex={-1}>
        <RuntimeBanner />
        <Outlet />
      </main>
    </div>
  );
}
