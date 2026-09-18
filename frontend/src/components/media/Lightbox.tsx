import type { ArtifactView } from "../../api/types";
import { artifactUrl } from "../../api/generations";
import { EmptyState } from "../ui/EmptyState";
import { Modal } from "../ui/Modal";

/** Full-size view of one artifact with its seed and a PNG download. */
export function Lightbox({
  runId,
  artifact,
  imageCount,
  onClose,
}: {
  runId: string;
  artifact: ArtifactView;
  imageCount: number;
  onClose: () => void;
}) {
  return (
    <Modal
      open
      onClose={onClose}
      title={`Image ${artifact.index} of ${imageCount}`}
      footer={
        <>
          <span className="lightbox__seed">Seed {artifact.seed}</span>
          <a
            className="button button--secondary button--md"
            href={artifactUrl(runId, artifact.artifact_id, true)}
            download
          >
            Download PNG
          </a>
        </>
      }
    >
      {artifact.url ? (
        <img
          className="lightbox__image"
          src={artifact.url}
          alt={`Generated image ${artifact.index} (seed ${artifact.seed})`}
        />
      ) : (
        <EmptyState
          title="Image unavailable"
          description="The artifact file is missing from storage."
        />
      )}
    </Modal>
  );
}
