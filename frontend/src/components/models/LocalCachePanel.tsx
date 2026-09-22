import { useState } from "react";

import { listCachedRepos } from "../../api/cache";
import { createRegistration } from "../../api/models";
import type { CachedRepo, ProfileId } from "../../api/types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { EmptyState } from "../ui/EmptyState";
import { ErrorState } from "../ui/ErrorState";
import { Modal } from "../ui/Modal";
import { SelectField, TextField } from "../ui/fields";
import { useApiQuery } from "../../hooks/useApiQuery";
import { useToast } from "../../state/toast/context";
import { errorMessage } from "../../api/client";
import { formatBytes } from "../../lib/format";
import { shortCommit } from "../../lib/statusTone";

function RegisterSnapshotDialog({
  repo,
  commitSha,
  onClose,
  onRegistered,
}: {
  repo: CachedRepo;
  commitSha: string | null;
  onClose: () => void;
  onRegistered: () => void;
}) {
  const toast = useToast();
  const [profile, setProfile] = useState<ProfileId>(
    repo.repo_id === "circlestone-labs/Anima"
      ? "anima-turbo"
      : repo.repo_id === "Gazingstars123/Anima-2.9B"
        ? "anima-2.9b"
        : repo.repo_id === "Qwen/Qwen-Image-2.1"
          ? "qwen-image-2.1"
          : "z-image",
  );
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);

  const register = async () => {
    setBusy(true);
    setError(undefined);
    try {
      await createRegistration({
        repo_id: repo.repo_id,
        revision: commitSha,
        profile,
        display_name: displayName.trim() === "" ? null : displayName.trim(),
      });
      toast.pushToast({
        kind: "success",
        message: `${repo.repo_id} registered.`,
      });
      onRegistered();
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
      title={`Register ${repo.repo_id}`}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            variant="primary"
            disabled={busy}
            onClick={() => void register()}
          >
            Register
          </Button>
        </>
      }
    >
      <p className="field-hint">
        Registering references the existing snapshot; nothing is downloaded.
      </p>
      <SelectField
        id="register-profile"
        label="Profile"
        value={profile}
        onChange={(event) => setProfile(event.target.value as ProfileId)}
      >
        <option value="z-image">z-image</option>
        <option value="z-image-turbo">z-image-turbo</option>
        <option value="anima-turbo">anima-turbo</option>
        <option value="anima-2.9b">anima-2.9b</option>
        <option value="qwen-image-2.1">qwen-image-2.1</option>
      </SelectField>
      <TextField
        id="register-display-name"
        label="Display name"
        value={displayName}
        onChange={(event) => setDisplayName(event.target.value)}
      />
      {error ? (
        <div className="notice notice--error" role="alert">
          {error}
        </div>
      ) : null}
    </Modal>
  );
}

/**
 * Every cached repository, including unsupported and incomplete
 * snapshots. Registering a snapshot never triggers a download.
 */
export function LocalCachePanel({
  onRegistered,
}: {
  onRegistered: () => void;
}) {
  const cacheQuery = useApiQuery(listCachedRepos, []);
  const [registering, setRegistering] = useState<{
    repo: CachedRepo;
    commitSha: string | null;
  } | null>(null);

  const repos = cacheQuery.data ?? [];

  if (cacheQuery.error !== undefined) {
    return (
      <ErrorState
        title="Cache scan failed"
        message="The local Hugging Face cache could not be read."
        onRetry={() => cacheQuery.refetch()}
      />
    );
  }

  if (cacheQuery.loading && repos.length === 0) {
    return <p className="field-hint">Scanning the local cache…</p>;
  }

  if (repos.length === 0) {
    return (
      <EmptyState
        title="Cache is empty"
        description="Downloaded model repositories appear here, including partial and unsupported ones."
      />
    );
  }

  return (
    <>
      <ul className="model-list">
        {repos.map((repo) => (
          <li key={repo.repo_id} className="model-card model-card--cache">
            <div className="model-card__main">
              <span className="model-card__repo" title={repo.repo_id}>
                {repo.repo_id}
              </span>
              <div className="model-card__meta">
                {repo.size_on_disk_bytes !== null &&
                repo.size_on_disk_bytes !== undefined ? (
                  <span>{formatBytes(repo.size_on_disk_bytes)}</span>
                ) : null}
                {repo.refs.map((ref) => (
                  <Badge key={ref} tone="neutral">
                    {ref}
                  </Badge>
                ))}
              </div>
              <ul className="model-card__snapshots">
                {repo.snapshots.map((snapshot) => (
                  <li key={snapshot.path}>
                    <span className="mono">
                      {snapshot.commit_sha
                        ? shortCommit(snapshot.commit_sha)
                        : "unknown commit"}
                    </span>
                    {snapshot.size_bytes !== null &&
                    snapshot.size_bytes !== undefined ? (
                      <span className="field-hint">
                        {" "}
                        · {formatBytes(snapshot.size_bytes)}
                      </span>
                    ) : null}
                    {snapshot.incomplete ? (
                      <Badge tone="warning">Incomplete</Badge>
                    ) : null}
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() =>
                        setRegistering({
                          repo,
                          commitSha: snapshot.commit_sha ?? null,
                        })
                      }
                    >
                      Register
                    </Button>
                  </li>
                ))}
              </ul>
            </div>
          </li>
        ))}
      </ul>
      {registering !== null ? (
        <RegisterSnapshotDialog
          repo={registering.repo}
          commitSha={registering.commitSha}
          onClose={() => setRegistering(null)}
          onRegistered={() => {
            setRegistering(null);
            onRegistered();
          }}
        />
      ) : null}
    </>
  );
}
