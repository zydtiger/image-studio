import type { ReactNode } from "react";

import { cx } from "../../lib/cx";
import { Button } from "./Button";

export function ErrorState({
  title = "Something went wrong",
  message,
  /** Technical detail, e.g. an error message; rendered in monospace. */
  detail,
  onRetry,
  retryLabel = "Retry",
  className,
}: {
  title?: ReactNode;
  message?: ReactNode;
  detail?: ReactNode;
  onRetry?: () => void;
  retryLabel?: string;
  className?: string;
}) {
  return (
    <div role="alert" className={cx("state", "state--error", className)}>
      <p className="state__title">{title}</p>
      {message !== undefined ? (
        <p className="state__description">{message}</p>
      ) : null}
      {detail !== undefined ? <p className="state__detail">{detail}</p> : null}
      {onRetry !== undefined ? (
        <div className="state__action">
          <Button variant="secondary" onClick={onRetry}>
            {retryLabel}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
