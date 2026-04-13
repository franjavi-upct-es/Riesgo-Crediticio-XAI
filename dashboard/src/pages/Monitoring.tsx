// dashboard/src/pages/Monitoring.tsx

import {
  Activity,
  Database,
  ExternalLink,
  HeartPulse,
  Server,
} from "lucide-react";
import { useDatasets, useHealth } from "@/hooks/useApi";
import Card from "@/components/ui/Card";
import MetricCard from "@/components/ui/MetricCard";
import { LoadingState } from "@/components/ui/StatusStates";

export default function Monitoring() {
  const health = useHealth();
  const { data: datasetsData } = useDatasets();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold tracking-tight text-foreground">
          Monitoring
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          API health, loaded models, and observability endpoints
        </p>
      </div>

      {health.isLoading ? (
        <LoadingState message="Checking API health…" />
      ) : (
        <div className="grid gap-4 sm:grid-cols-3">
          <MetricCard
            label="API status"
            value={health.data?.status ?? "unknown"}
            icon={<HeartPulse size={16} />}
            variant={health.data?.status === "healthy" ? "success" : "danger"}
          />
          <MetricCard
            label="Models loaded"
            value={health.data?.loaded_datasets?.length ?? 0}
            subtitle={health.data?.loaded_datasets?.join(", ") || "None"}
            icon={<Server size={16} />}
            variant={health.data?.model_loaded ? "success" : "danger"}
          />
          <MetricCard
            label="API version"
            value={health.data?.version ?? "—"}
            icon={<Activity size={16} />}
            variant="info"
          />
        </div>
      )}

      {datasetsData && datasetsData.count > 0 && (
        <Card
          title="Available datasets"
          subtitle={`${datasetsData.count} dataset configurations found`}
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {datasetsData.datasets.map((ds) => (
              <div
                key={ds.id}
                className="rounded-lg border border-border p-4 transition-colors hover:border-primary/40"
              >
                <div className="flex items-center gap-2">
                  <Database size={14} className="text-primary" />
                  <span className="text-sm font-semibold text-foreground">
                    {ds.name}
                  </span>
                </div>
                <p className="mt-1.5 line-clamp-2 text-xs text-muted-foreground">
                  {ds.description}
                </p>
                <div className="mt-2 flex flex-wrap gap-2 text-[10px] text-muted-foreground">
                  <span className="rounded bg-muted px-1.5 py-0.5">
                    {ds.n_features} features
                  </span>
                  <span className="rounded bg-muted px-1.5 py-0.5">
                    {ds.n_categorical} cat
                  </span>
                  <span className="rounded bg-muted px-1.5 py-0.5">
                    {ds.n_numerical} num
                  </span>
                  {ds.n_protected > 0 && (
                    <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                      {ds.n_protected} protected
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Observability endpoints">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {[
            {
              label: "API docs (Swagger)",
              url: "http://localhost:8000/docs",
              desc: "Interactive API documentation",
            },
            {
              label: "Prometheus metrics",
              url: "http://localhost:8000/metrics",
              desc: "Raw Prometheus text format",
            },
            {
              label: "Prometheus UI",
              url: "http://localhost:9090",
              desc: "Query and graph metrics",
            },
            {
              label: "Grafana dashboards",
              url: "http://localhost:3000",
              desc: "Visualization (admin / admin)",
            },
            {
              label: "Jaeger traces",
              url: "http://localhost:16686",
              desc: "Distributed tracing UI",
            },
            {
              label: "Dataset schemas",
              url: "http://localhost:8000/datasets/",
              desc: "All registered datasets",
            },
          ].map((link) => (
            <a
              key={link.url}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-3 rounded-lg border border-border p-4 transition-colors hover:border-primary/40 hover:bg-primary/5"
            >
              <ExternalLink
                size={16}
                className="mt-0.5 shrink-0 text-muted-foreground group-hover:text-primary"
              />
              <div>
                <p className="text-sm font-medium text-foreground group-hover:text-primary">
                  {link.label}
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {link.desc}
                </p>
              </div>
            </a>
          ))}
        </div>
      </Card>
    </div>
  );
}
