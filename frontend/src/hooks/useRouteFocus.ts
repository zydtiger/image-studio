import { useEffect, useRef } from "react";
import { useLocation } from "react-router";

/**
 * Returns a ref for a main content element and focuses it whenever the
 * route changes, so keyboard and screen-reader users land on the new
 * content instead of staying on the navigation. Focus is only moved when
 * it already lives inside the application (a real in-app navigation); a
 * fresh load or a background tab keeps the browser's default focus so
 * Tab starts at the top, including the skip link.
 */
export function useRouteFocus<T extends HTMLElement>() {
  const { pathname } = useLocation();
  const ref = useRef<T>(null);

  useEffect(() => {
    const main = ref.current;
    if (main === null) return;
    const shell = main.parentElement;
    const active = document.activeElement;
    if (
      shell !== null &&
      active instanceof Node &&
      shell.contains(active) &&
      active !== main
    ) {
      main.focus({ preventScroll: true });
    }
  }, [pathname]);

  return ref;
}
