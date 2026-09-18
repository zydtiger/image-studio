import { createContext, useContext } from "react";

export type ToastKind = "info" | "success" | "warning" | "error";

export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface ToastInput {
  kind: ToastKind;
  /** Message text. Unicode content is rendered as-is. */
  message: string;
  action?: ToastAction;
  /**
   * Milliseconds before automatic dismissal. Defaults depend on the kind;
   * pass 0 to keep the toast until dismissed manually.
   */
  durationMs?: number;
}

export interface ToastApi {
  pushToast: (toast: ToastInput) => void;
}

export const ToastContext = createContext<ToastApi | null>(null);

export const DEFAULT_TOAST_DURATION_MS = 6000;
export const ERROR_TOAST_DURATION_MS = 9000;

export function useToast(): ToastApi {
  const context = useContext(ToastContext);
  if (context === null) {
    throw new Error("useToast must be used within a ToastProvider.");
  }
  return context;
}
