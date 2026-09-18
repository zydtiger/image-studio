import { useState } from "react";

import { createDownload } from "../../api/downloads";
import { createRegistration } from "../../api/models";
import { getCompatibility, getHubModelDetail } from "../../api/hub";
import type { ProfileId } from "../../api/types";
import { isApiError } from "../../api/client";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";
import { SelectField } from "../ui/fields";
import { Spinner } from "../ui/Spinner";
import { useApiQuery } from "../../hooks/useApiQuery";
import { useToast } from "../../state/toast/context";
import { errorMessage } from "../../api/client";
import { shortCommit } from "../../lib/statusTone";
import { formatBytes, formatRelativeTime } from "../../lib/format";

/**
 * Hub model detail: card metadata, revisions, and the structural
 * compatibility report. The distillation profile is an explicit user
 * choice, never inferred from compatibility.
 */
export function ModelDetailDialog({
  repoId,
  onClose,
  onDownloaded,
  onRegistered,
}: {
  repoId: string;
  onClose: () => void;
  onDownloaded: () => void;
  onRegistered: () => void;
}) {
  const toast = useToast();
  const [revision, setRevision] = useState<string | null>(null);
  const [profile, setProfile] = useState<ProfileId | "">("");
  const [actionError, setActionError] = useState<string>();
  const [busy, setBusy] = useState(false);

  const detailQuery = useApiQuery(
    (signal) => getHubModelDetail(repoId, signal),
    [repoId],
  );
  const detail = detailQuery.data;
  const effectiveRevision =
    revision ??
    detail?.default_revision ??
    detail?.revisions[0]?.revision ??
    null;

  const compatQuery = useApiQuery(
    (signal) => getCompatibility(repoId, effectiveRevision, signal),
    [repoId, effectiveRevision],
    { enabled: detailQuery.data !== undefined },
  );
  const compat = compatQuery.data;

  const selectable = compat?.selectable_profiles ?? [];
  const effectiveProfile: ProfileId | "" =
    profile === "" && selectable.length > 0 ? selectable[0] : profile;
  // The compatibility report must belong to the currently selected
  // revision: a report that is still loading, failed, or was fetched for a
  // previous selection must not authorize registering a fixed snapshot.
  const compatCurrent =
    compat !== undefined &&
    !compatQuery.loading &&
    compatQuery.error === undefined &&
    compat.revision === effectiveRevision;
  const canAct =
    compatCurrent &&
    compat.structurally_compatible === true &&
    effectiveProfile !== "";

  const act = async (action: "download" | "register") => {
    if (effectiveProfile === "" || effectiveRevision === null) return;
    if (action === "register" && !compatCurrent) return;
    setBusy(true);
    setActionError(undefined);
    try {
      if (action === "download") {
        await createDownload({
          repo_id: repoId,
          revision: effectiveRevision,
          profile: effectiveProfile,
        });
        toast.pushToast({
          kind: "success",
          message: `Download queued for ${repoId}.`,
        });
        onDownloaded();
      } else {
        // Re-check the report at the action site: register only from the
        // compatibility report that belongs to the currently selected
        // revision. Register by its resolved fixed commit — SDK downloads
        // by commit never create refs/<branch>, so a complete commit-only
        // cache has no branch ref to resolve and a branch revision 422s.
        const report = compat;
        if (
          report === undefined ||
          report.revision !== effectiveRevision ||
          compatQuery.error !== undefined
        ) {
          return;
        }
        await createRegistration({
          repo_id: repoId,
          revision: report.commit_sha ?? effectiveRevision,
          profile: effectiveProfile,
        });
        toast.pushToast({
          kind: "success",
          message: `${repoId} registered.`,
        });
        onRegistered();
      }
    } catch (error) {
      setActionError(isApiError(error) ? error.message : errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open
      onClose={onClose}
      title={repoId}
      footer={
        <>
          <Button onClick={onClose}>Close</Button>
          <Button
            variant="secondary"
            disabled={!canAct || busy}
            onClick={() => void act("register")}
            title={
              compat && !compat.structurally_compatible
                ? "The repository is not structurally compatible."
                : undefined
            }
          >
            Register cached snapshot
          </Button>
          <Button
            variant="primary"
            disabled={!canAct || busy}
            onClick={() => void act("download")}
            title={
              compat && !compat.structurally_compatible
                ? "The repository is not structurally compatible."
                : undefined
            }
          >
            Download…
          </Button>
        </>
      }
    >
      {detailQuery.loading && detail === undefined ? (
        <p>
          <Spinner label="Loading model details" />
        </p>
      ) : detail ? (
        <div className="model-detail">
          <div className="model-detail__meta">
            {detail.author ? <span>By {detail.author}</span> : null}
            {detail.gated ? <Badge tone="warning">Gated</Badge> : null}
            {detail.private ? <Badge tone="neutral">Private</Badge> : null}
            {detail.license ? <span>{detail.license}</span> : null}
            {detail.downloads !== null && detail.downloads !== undefined ? (
              <span>{detail.downloads.toLocaleString("en")} downloads</span>
            ) : null}
            {detail.likes !== null && detail.likes !== undefined ? (
              <span>{detail.likes.toLocaleString("en")} likes</span>
            ) : null}
            {detail.last_modified ? (
              <span>Updated {formatRelativeTime(detail.last_modified)}</span>
            ) : null}
          </div>

          {detail.revisions.length > 0 ? (
            <SelectField
              id="hub-revision"
              label="Revision"
              value={effectiveRevision ?? ""}
              onChange={(event) => setRevision(event.target.value)}
              hint={
                detail.revisions.find(
                  (entry) => entry.revision === effectiveRevision,
                )
                  ? `Commit ${shortCommit(
                      detail.revisions.find(
                        (entry) => entry.revision === effectiveRevision,
                      )!.commit_sha,
                    )}`
                  : undefined
              }
            >
              {detail.revisions.map((entry) => (
                <option key={entry.revision} value={entry.revision}>
                  {entry.revision}
                </option>
              ))}
            </SelectField>
          ) : null}

          <div className="model-detail__compat">
            <h3>Compatibility</h3>
            {compatQuery.loading && compat === undefined ? (
              <p>
                <Spinner label="Checking compatibility" />
              </p>
            ) : compat ? (
              <>
                <p>
                  {compat.structurally_compatible ? (
                    <Badge tone="success">Structurally compatible</Badge>
                  ) : (
                    <Badge tone="danger">Not compatible</Badge>
                  )}
                </p>
                {selectable.length > 0 ? (
                  <SelectField
                    id="hub-profile"
                    label="Profile"
                    value={effectiveProfile}
                    onChange={(event) =>
                      setProfile(event.target.value as ProfileId)
                    }
                    hint="Base and Turbo are your explicit choice; compatibility does not guarantee output quality."
                  >
                    {selectable.map((entry) => (
                      <option key={entry} value={entry}>
                        {entry}
                      </option>
                    ))}
                  </SelectField>
                ) : null}
                {compat.findings.length > 0 ? (
                  <ul className="model-detail__findings">
                    {compat.findings.map((finding) => (
                      <li key={finding}>{finding}</li>
                    ))}
                  </ul>
                ) : null}
                {compat.notes.length > 0 ? (
                  <ul className="model-detail__notes">
                    {compat.notes.map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                ) : null}
              </>
            ) : (
              <p className="field-hint">
                Compatibility information is unavailable.
              </p>
            )}
          </div>

          {detail.files.length > 0 ? (
            <details className="model-detail__files">
              <summary>Files ({detail.files.length})</summary>
              <ul>
                {detail.files.map((file) => (
                  <li key={file.path}>
                    <span className="mono">{file.path}</span>
                    {file.size !== null && file.size !== undefined ? (
                      <span className="field-hint">
                        {" "}
                        · {formatBytes(file.size)}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            </details>
          ) : null}

          {actionError ? (
            <div className="notice notice--error" role="alert">
              {actionError}
            </div>
          ) : null}
        </div>
      ) : (
        <p className="field-hint">
          Model details are unavailable. {errorMessage(detailQuery.error)}
        </p>
      )}
    </Modal>
  );
}
