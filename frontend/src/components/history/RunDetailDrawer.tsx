import { useState } from "react";

import {
  getRun,
  metadataUrl,
  restoreRun,
  setFavorite,
  trashRun,
} from "../../api/generations";
import type { ArtifactView } from "../../api/types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { ConfirmDialog } from "../ui/ConfirmDialog";
import { EmptyState } from "../ui/EmptyState";
import { Modal } from "../ui/Modal";
import { Spinner } from "../ui/Spinner";
import { StarIcon } from "../ui/icons";
import { ImageTile } from "../generate/ImageTile";
import { Lightbox } from "../media/Lightbox";
import { useApiQuery } from "../../hooks/useApiQuery";
import { useToast } from "../../state/toast/context";
import { errorMessage } from "../../api/client";
import { buildReusePayload } from "../../lib/reuse";
import { formatRelativeTime, formatTimestamp } from "../../lib/format";
import {
  runStatusLabel,
  runStatusTone,
  shortCommit,
} from "../../lib/statusTone";

function Row({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="detail-row">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

/**
 * Full run record: frozen parameters, environment, per-image results,
 * favorite, trash/restore, downloads, and parameter reuse.
 */
export function RunDetailDrawer({
  runId,
  onClose,
  onChanged,
  onReuse,
}: {
  runId: string;
  onClose: () => void;
  onChanged: () => void;
  onReuse: (payload: ReturnType<typeof buildReusePayload>) => void;
}) {
  const toast = useToast();
  const [version, setVersion] = useState(0);
  const [confirmTrash, setConfirmTrash] = useState(false);
  const [openArtifact, setOpenArtifact] = useState<ArtifactView>();

  const runQuery = useApiQuery(
    (signal) => getRun(runId, signal),
    [runId, version],
  );
  const run = runQuery.data;

  const act = async (action: () => Promise<unknown>, message: string) => {
    try {
      await action();
      setVersion((current) => current + 1);
      onChanged();
      toast.pushToast({ kind: "success", message });
    } catch (error) {
      toast.pushToast({ kind: "error", message: errorMessage(error) });
    }
  };

  return (
    <Modal
      open
      onClose={onClose}
      title={`Run ${runId.slice(0, 8)}`}
      variant="drawer"
      footer={
        run !== undefined ? (
          <>
            <a
              className="button button--secondary button--md"
              href={metadataUrl(run.run_id)}
              download
            >
              Metadata
            </a>
            {run.trashed ? (
              <Button
                variant="secondary"
                onClick={() =>
                  void act(
                    () => restoreRun(run.run_id),
                    "Run restored from Trash.",
                  )
                }
              >
                Restore
              </Button>
            ) : (
              <Button variant="danger" onClick={() => setConfirmTrash(true)}>
                Move to Trash
              </Button>
            )}
            <Button
              variant="primary"
              onClick={() => onReuse(buildReusePayload(run))}
            >
              Reuse parameters
            </Button>
          </>
        ) : null
      }
    >
      {run === undefined ? (
        runQuery.error !== undefined ? (
          <EmptyState
            title="Run unavailable"
            description="The run could not be loaded."
          />
        ) : (
          <p>
            <Spinner label="Loading run" />
          </p>
        )
      ) : (
        <div className="run-detail">
          <div className="run-detail__head">
            <Badge tone={runStatusTone(run.status)}>
              {runStatusLabel(run.status)}
            </Badge>
            <button
              type="button"
              className={
                run.favorite
                  ? "icon-button run-detail__star run-detail__star--on"
                  : "icon-button run-detail__star"
              }
              aria-pressed={run.favorite}
              aria-label={run.favorite ? "Remove favorite" : "Mark favorite"}
              onClick={() =>
                void act(
                  () => setFavorite(run.run_id, !run.favorite),
                  run.favorite ? "Removed favorite." : "Marked favorite.",
                )
              }
            >
              <StarIcon filled={run.favorite} />
            </button>
          </div>

          {run.error ? (
            <div className="notice notice--error" role="alert">
              {run.error.message}
            </div>
          ) : null}

          <dl className="detail-list">
            <Row label="Prompt">
              <p className="detail-prompt">{run.prompt}</p>
            </Row>
            {run.negative_prompt !== null &&
            run.negative_prompt !== undefined ? (
              <Row label="Negative prompt">
                <p className="detail-prompt">{run.negative_prompt}</p>
              </Row>
            ) : null}
            <Row label="Model">
              <span title={run.repo_id}>
                {run.repo_id} · {run.profile} ·{" "}
                <span className="mono">{shortCommit(run.commit_sha)}</span>
              </span>
            </Row>
            <Row label="GPU">{run.gpu?.name ?? "Unknown"}</Row>
            <Row label="Size">
              {run.width} × {run.height}
            </Row>
            <Row label="Steps / Guidance">
              {run.steps} / {run.guidance}
            </Row>
            <Row label="Initial seed">
              {run.initial_seed} (+1 per image, never wrapping)
            </Row>
            <Row label="Images">
              {run.images.filter((i) => i.status === "completed").length} of{" "}
              {run.image_count} completed
            </Row>
            <Row label="Precision">{run.dtype}</Row>
            {run.pipeline_class ? (
              <Row label="Pipeline">{run.pipeline_class}</Row>
            ) : null}
            <Row label="Created">
              {formatTimestamp(run.created_at)} (
              {formatRelativeTime(run.created_at)})
            </Row>
            {run.started_at ? (
              <Row label="Started">{formatTimestamp(run.started_at)}</Row>
            ) : null}
            {run.finished_at ? (
              <Row label="Finished">{formatTimestamp(run.finished_at)}</Row>
            ) : null}
            {Object.keys(run.dependency_versions).length > 0 ? (
              <Row label="Dependencies">
                {Object.entries(run.dependency_versions)
                  .map(([name, version]) => `${name} ${version}`)
                  .join(" · ")}
              </Row>
            ) : null}
          </dl>

          {run.images.length > 0 ? (
            <div className="results-grid results-grid--drawer">
              {run.images.map((artifact) => (
                <ImageTile
                  key={artifact.artifact_id}
                  runId={run.run_id}
                  artifact={artifact}
                  onOpen={setOpenArtifact}
                />
              ))}
            </div>
          ) : (
            <EmptyState
              title="No image records"
              description="This run has no per-image records."
            />
          )}
        </div>
      )}

      <ConfirmDialog
        open={confirmTrash}
        onClose={() => setConfirmTrash(false)}
        onConfirm={() =>
          void act(
            () => trashRun(runId),
            "Run moved to Trash. Restore anytime.",
          )
        }
        title="Move run to Trash?"
        confirmLabel="Move to Trash"
        tone="danger"
      >
        Images move to the recoverable Trash. Thumbnails are dropped; no file is
        permanently deleted.
      </ConfirmDialog>

      {openArtifact !== undefined && run !== undefined ? (
        <Lightbox
          runId={run.run_id}
          artifact={openArtifact}
          imageCount={run.image_count}
          onClose={() => setOpenArtifact(undefined)}
        />
      ) : null}
    </Modal>
  );
}
