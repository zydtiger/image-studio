import { useState } from "react";

import { listRegistrations } from "../api/models";
import type { ModelRegistration } from "../api/types";
import { Tabs } from "../components/ui/Tabs";
import { useApiQuery } from "../hooks/useApiQuery";
import { DiscoverPanel } from "../components/models/DiscoverPanel";
import { DownloadsPanel } from "../components/models/DownloadsPanel";
import { LocalCachePanel } from "../components/models/LocalCachePanel";
import { MyModelsPanel } from "../components/models/MyModelsPanel";

type ModelsTab = "discover" | "library" | "cache" | "downloads";

/**
 * Model management: Hub discovery, application registrations, the local
 * cache, and the separate download queue.
 */
export default function ModelsPage() {
  const [tab, setTab] = useState<ModelsTab>("discover");
  const modelsQuery = useApiQuery(listRegistrations, []);
  const registrations: ModelRegistration[] = modelsQuery.data ?? [];

  return (
    <section className="page page--models">
      <header className="page-header">
        <h1>Models</h1>
        <p>
          Search Hugging Face, manage registrations, inspect the local cache,
          and follow downloads.
        </p>
      </header>
      <Tabs
        aria-label="Model views"
        activeId={tab}
        onChange={(id) => setTab(id as ModelsTab)}
        tabs={[
          {
            id: "library",
            label: "My Models",
            content: (
              <MyModelsPanel
                registrations={registrations}
                loading={modelsQuery.loading}
                error={modelsQuery.error}
                onRetry={() => modelsQuery.refetch()}
                onChanged={() => modelsQuery.refetch()}
                onGoDiscover={() => setTab("discover")}
              />
            ),
          },
          {
            id: "cache",
            label: "Local Cache",
            content: (
              <LocalCachePanel onRegistered={() => modelsQuery.refetch()} />
            ),
          },
          {
            id: "discover",
            label: "Discover",
            content: (
              <DiscoverPanel
                onDownloaded={() => setTab("downloads")}
                onRegistered={() => modelsQuery.refetch()}
              />
            ),
          },
          {
            id: "downloads",
            label: "Downloads",
            content: <DownloadsPanel />,
          },
        ]}
      />
    </section>
  );
}
