import { cx } from "../../lib/cx";
import { VisuallyHidden } from "./VisuallyHidden";

export function Spinner({
  label = "Loading",
  size = "md",
}: {
  label?: string;
  size?: "sm" | "md" | "lg";
}) {
  return (
    <span role="status">
      <svg
        viewBox="0 0 20 20"
        className={cx("spinner", `spinner--${size}`)}
        aria-hidden="true"
        focusable="false"
      >
        <circle
          cx="10"
          cy="10"
          r="8"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeDasharray="38 12"
          strokeLinecap="round"
        />
      </svg>
      <VisuallyHidden>{label}</VisuallyHidden>
    </span>
  );
}
