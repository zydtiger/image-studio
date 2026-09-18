import { cx } from "../../lib/cx";

export interface ProgressBarProps {
  /** Percentage from 0 to 100; omit or pass null for indeterminate. */
  value?: number | null;
  /** Accessible name describing what is progressing. */
  label: string;
  className?: string;
}

export function ProgressBar({ value, label, className }: ProgressBarProps) {
  const determinate = typeof value === "number" && Number.isFinite(value);
  const clamped = determinate
    ? Math.min(100, Math.max(0, value as number))
    : undefined;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={determinate ? Math.round(clamped as number) : undefined}
      className={cx(
        "progress",
        !determinate && "progress--indeterminate",
        className,
      )}
    >
      <div
        className="progress__fill"
        style={determinate ? { width: `${clamped}%` } : undefined}
      />
    </div>
  );
}
