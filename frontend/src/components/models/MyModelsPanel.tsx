import { useState } from "react";

import { deleteRegistration, updateRegistration } from "../../api/models";
import type { ModelRegistration, ProfileId } from "../../api/types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { ConfirmDialog } from "../ui/ConfirmDialog";
import { EmptyState } from "../ui/EmptyState";
import { ErrorState } from "../ui/ErrorState";
import { Modal } from "../ui/Modal";
import { SelectField, TextField } from "../ui/fields";
import { useToast } from "../../state/toast/context";
import { errorMessage } from "../../api/client";
import { formatRelativeTime } from "../../lib/format";
import { shortCommit } from "../../lib/statusTone";

function EditDialog({
  registration,
  profiles,
  onClose,
  onSaved,
}: {
  registration: ModelRegistration;
  profiles: ProfileId[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const toast = useToast();
  const [displayName, setDisplayName] = useState(
    registration.display_name ?? "",
  );
  const [profile, setProfile] = useState<ProfileId>(registration.profile);
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    setError(undefined);
    try {
      await updateRegistration(registration.id, {
        display_name: displayName.trim() === "" ? null : displayName.trim(),
        profile,
      });
      toast.pushToast({
        kind: "success",
        message: "Registration updated.",
      });
      onSaved();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open
      onClose={onClose}
      title={`Edit ${registration.repo_id}`}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={busy} onClick={() => void save()}>
            Save
          </Button>
        </>
      }
    >
      <TextField
        id="edit-display-name"
        label="Display name"
        value={displayName}
        onChange={(event) => setDisplayName(event.target.value)}
      />
      <SelectField
        id="edit-profile"
        label="Profile"
        value={profile}
        onChange={(event) => setProfile(event.target.value as ProfileId)}
        hint="Changing the profile is refused while the model is resident or referenced by unfinished runs."
      >
        {profiles.map((entry) => (
          <option key={entry} value={entry}>
            {entry}
          </option>
        ))}
      </SelectField>
      {error ? (
        <div className="notice notice--error" role="alert">
          {error}
        </div>
      ) : null}
    </Modal>
  );
}

/** The application's model registrations. */
export function MyModelsPanel({
  registrations,
  loading,
  error,
  onRetry,
  onChanged,
  onGoDiscover,
}: {
  registrations: ModelRegistration[];
  loading: boolean;
  error: unknown;
  onRetry: () => void;
  onChanged: () => void;
  onGoDiscover: () => void;
}) {
  const toast = useToast();
  const [editing, setEditing] = useState<ModelRegistration | null>(null);
  const [removing, setRemoving] = useState<ModelRegistration | null>(null);

  const remove = async () => {
    if (removing === null) return;
    try {
      await deleteRegistration(removing.id);
      toast.pushToast({
        kind: "success",
        message: `${removing.repo_id} removed from the library. Cache files were not deleted.`,
      });
      onChanged();
    } catch (cause) {
      toast.pushToast({ kind: "error", message: errorMessage(cause) });
    } finally {
      setRemoving(null);
    }
  };

  if (error !== undefined) {
    return (
      <ErrorState
        title="Registrations unavailable"
        message="The registration list could not be loaded."
        onRetry={onRetry}
      />
    );
  }

  if (loading && registrations.length === 0) {
    return <p className="field-hint">Loading registrations…</p>;
  }

  if (registrations.length === 0) {
    return (
      <EmptyState
        title="No models registered"
        description="Register a cached snapshot from Discover or Local Cache, or download one from the Hub."
        action={
          <Button variant="primary" onClick={onGoDiscover}>
            Open Discover
          </Button>
        }
      />
    );
  }

  return (
    <>
      <ul className="model-list">
        {registrations.map((registration) => (
          <li key={registration.id} className="model-card">
            <div className="model-card__main">
              <span className="model-card__repo" title={registration.repo_id}>
                {registration.display_name ?? registration.repo_id}
              </span>
              <div className="model-card__meta">
                <Badge tone="info">{registration.profile}</Badge>
                {registration.status === "ready" ? (
                  <Badge tone="success">Ready</Badge>
                ) : (
                  <Badge tone="danger">Missing files</Badge>
                )}
                <span className="mono" title={registration.commit_sha}>
                  {shortCommit(registration.commit_sha)}
                </span>
                <span>
                  used{" "}
                  {registration.last_used_at
                    ? formatRelativeTime(registration.last_used_at)
                    : "never"}
                </span>
              </div>
              {registration.missing_files.length > 0 ? (
                <p className="model-card__missing">
                  Missing from the cache:{" "}
                  {registration.missing_files.join(", ")}
                </p>
              ) : null}
            </div>
            <div className="model-card__actions">
              <Button size="sm" onClick={() => setEditing(registration)}>
                Edit
              </Button>
              <Button
                size="sm"
                variant="danger"
                onClick={() => setRemoving(registration)}
              >
                Remove
              </Button>
            </div>
          </li>
        ))}
      </ul>
      {editing !== null ? (
        <EditDialog
          registration={editing}
          profiles={
            editing.profile === "z-image" || editing.profile === "z-image-turbo"
              ? ["z-image", "z-image-turbo"]
              : [editing.profile]
          }
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null);
            onChanged();
          }}
        />
      ) : null}
      <ConfirmDialog
        open={removing !== null}
        onClose={() => setRemoving(null)}
        onConfirm={() => void remove()}
        title={`Remove ${removing?.repo_id ?? ""}?`}
        confirmLabel="Remove registration"
        tone="danger"
      >
        The registration is removed from this application only. Files in the
        shared Hugging Face cache are never deleted. Refused while the model is
        resident or referenced by unfinished runs.
      </ConfirmDialog>
    </>
  );
}
