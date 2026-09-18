import type { ReactNode } from "react";

export function EmptyState({
  title,
  description,
  action,
}: {
  title: ReactNode;
  description?: ReactNode;
  /** Optional call-to-action, usually a Button. */
  action?: ReactNode;
}) {
  return (
    <div className="state">
      <p className="state__title">{title}</p>
      {description !== undefined ? (
        <p className="state__description">{description}</p>
      ) : null}
      {action !== undefined ? (
        <div className="state__action">{action}</div>
      ) : null}
    </div>
  );
}
