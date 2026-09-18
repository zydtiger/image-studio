import type {
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";

import { cx } from "../../lib/cx";

/**
 * Labeled form controls. Callers own the `id` so labels stay associated
 * across renders and tests; hint and error text are wired through
 * aria-describedby and aria-invalid.
 */

interface FieldExtras {
  id: string;
  label: ReactNode;
  /** Helper text shown below the control. */
  hint?: ReactNode;
  /** Error text; marks the control invalid and announces it. */
  error?: ReactNode;
  wrapperClassName?: string;
}

function describedBy(id: string, hint?: ReactNode, error?: ReactNode) {
  const ids = [
    hint !== undefined ? `${id}-hint` : null,
    error !== undefined ? `${id}-error` : null,
  ];
  const joined = ids.filter(Boolean).join(" ");
  return joined === "" ? undefined : joined;
}

export interface TextFieldProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "id">, FieldExtras {
  /** Renders a textarea instead of a single-line input. */
  multiline?: boolean;
  rows?: number;
}

export function TextField({
  id,
  label,
  hint,
  error,
  wrapperClassName,
  className,
  multiline = false,
  rows,
  ...input
}: TextFieldProps) {
  const controlProps = {
    id,
    className: cx("field-control", className),
    "aria-invalid": error !== undefined ? true : undefined,
    "aria-describedby": describedBy(id, hint, error),
  };
  return (
    <div className={cx("field", wrapperClassName)}>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      {multiline ? (
        <textarea
          {...controlProps}
          rows={rows ?? 3}
          {...(input as TextareaHTMLAttributes<HTMLTextAreaElement>)}
        />
      ) : (
        <input {...controlProps} {...input} />
      )}
      {hint !== undefined ? (
        <p className="field-hint" id={`${id}-hint`}>
          {hint}
        </p>
      ) : null}
      {error !== undefined ? (
        <p className="field-error" id={`${id}-error`} role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export interface NumberFieldProps
  extends
    Omit<InputHTMLAttributes<HTMLInputElement>, "id" | "type">,
    FieldExtras {}

export function NumberField({
  id,
  label,
  hint,
  error,
  wrapperClassName,
  className,
  ...input
}: NumberFieldProps) {
  return (
    <div className={cx("field", wrapperClassName)}>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <input
        type="number"
        id={id}
        className={cx("field-control", className)}
        aria-invalid={error !== undefined ? true : undefined}
        aria-describedby={describedBy(id, hint, error)}
        {...input}
      />
      {hint !== undefined ? (
        <p className="field-hint" id={`${id}-hint`}>
          {hint}
        </p>
      ) : null}
      {error !== undefined ? (
        <p className="field-error" id={`${id}-error`} role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export interface SelectFieldProps
  extends Omit<SelectHTMLAttributes<HTMLSelectElement>, "id">, FieldExtras {}

export function SelectField({
  id,
  label,
  hint,
  error,
  wrapperClassName,
  className,
  children,
  ...select
}: SelectFieldProps) {
  return (
    <div className={cx("field", wrapperClassName)}>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <select
        id={id}
        className={cx("field-control", className)}
        aria-invalid={error !== undefined ? true : undefined}
        aria-describedby={describedBy(id, hint, error)}
        {...select}
      >
        {children}
      </select>
      {hint !== undefined ? (
        <p className="field-hint" id={`${id}-hint`}>
          {hint}
        </p>
      ) : null}
      {error !== undefined ? (
        <p className="field-error" id={`${id}-error`} role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
