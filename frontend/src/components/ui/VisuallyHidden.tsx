import type { ReactNode } from "react";

/** Hides content visually while keeping it available to assistive tech. */
export function VisuallyHidden({ children }: { children: ReactNode }) {
  return <span className="visually-hidden">{children}</span>;
}
