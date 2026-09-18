import type { ReactNode } from "react";

import { cx } from "../../lib/cx";

export type BadgeTone = "neutral" | "info" | "success" | "warning" | "danger";

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: BadgeTone;
  children: ReactNode;
}) {
  return <span className={cx("badge", `badge--${tone}`)}>{children}</span>;
}
