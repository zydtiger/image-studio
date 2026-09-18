import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { cx } from "../../lib/cx";
import { CloseIcon } from "../../components/ui/icons";
import {
  DEFAULT_TOAST_DURATION_MS,
  ERROR_TOAST_DURATION_MS,
  ToastContext,
  type ToastInput,
  type ToastKind,
} from "./context";

interface ActiveToast extends ToastInput {
  id: number;
}

interface ToastItemProps {
  toast: ActiveToast;
  onDismiss: (id: number) => void;
}

function defaultDuration(kind: ToastKind): number {
  return kind === "error" || kind === "warning"
    ? ERROR_TOAST_DURATION_MS
    : DEFAULT_TOAST_DURATION_MS;
}

function ToastItem({ toast, onDismiss }: ToastItemProps) {
  const { kind, message, action, durationMs } = toast;

  useEffect(() => {
    const duration = durationMs ?? defaultDuration(kind);
    if (duration <= 0) return;
    const timer = setTimeout(() => onDismiss(toast.id), duration);
    return () => clearTimeout(timer);
  }, [kind, durationMs, toast.id, onDismiss]);

  return (
    <div
      className={cx("toast", `toast--${kind}`)}
      role={kind === "error" || kind === "warning" ? "alert" : "status"}
    >
      <p className="toast__content">{message}</p>
      {action ? (
        <button
          type="button"
          className="toast__action"
          onClick={() => {
            action.onClick();
            onDismiss(toast.id);
          }}
        >
          {action.label}
        </button>
      ) : null}
      <button
        type="button"
        className="icon-button icon-button--sm"
        aria-label="Dismiss notification"
        onClick={() => onDismiss(toast.id)}
      >
        <CloseIcon />
      </button>
    </div>
  );
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ActiveToast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const pushToast = useCallback((toast: ToastInput) => {
    const id = nextId.current;
    nextId.current += 1;
    setToasts((current) => [...current, { ...toast, id }]);
  }, []);

  return (
    <ToastContext.Provider value={{ pushToast }}>
      {children}
      <div className="toast-region" aria-label="Notifications">
        {toasts.map((toast) => (
          <ToastItem key={toast.id} toast={toast} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
