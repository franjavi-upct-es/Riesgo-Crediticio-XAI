// dashboard/src/pages/Monitoring.tsx

import { useDatasets, useHealth } from "@/hooks/useApi";
import Card from "@/components/ui/Card";
import MetricCard from "@/components/ui/MetricCard";
import { LoadingState } from "@/components/ui/StatusStates";
import { Activity, Database, ExternalLink, HeartPulse, Server } from "lucide-react";

export default function Monitoring() {
  const health = useHealth();
  const { data: datasetsData } = useDatasets();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold tracking-tight text-slate-900">
          Monitoring
        </h2>
        <p className="mt-1 text-sm text-accent-muted">
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

      {/* Available datasets */}
      {datasetsData && datasetsData.count > 0 && (
        <Card title="Available datasets" subtitle={`${datasetsData.count} dataset configurations found`}>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {datasetsData.datasets.map((ds) => (
              <div
                key={ds.id}
                className="rounded-lg border border-slate-200 p-4 transition-colors hover:border-violet-200"
              >
                <div className="flex items-center gap-2">
                  <Database size={14} className="text-violet-500" />
                  <span className="text-sm font-semibold text-slate-800">{ds.name}</span>
                </div>
                <p className="mt-1.5 line-clamp-2 text-xs text-accent-muted">
                  {ds.description}
                </p>
                <div className="mt-2 flex gap-2 text-[10px] text-accent-muted">
                  <span className="rounded bg-slate-100 px-1.5 py-0.5">{ds.n_features} features</span>
                  <span className="rounded bg-slate-100 px-1.5 py-0.5">{ds.n_categorical} cat</span>
                  <span className="rounded bg-slate-100 px-1.5 py-0.5">{ds.n_numerical} num</span>
                  {ds.n_protected > 0 && (
                    <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-700">
                      {ds.n_protected} protected
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Links */}
      <Card title="Observability endpoints">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {[
            { label: "API docs (Swagger)", url: "http://localhost:8000/docs", desc: "Interactive API documentation" },
            { label: "Prometheus metrics", url: "http://localhost:8000/metrics", desc: "Raw Prometheus text format" },
            { label: "Prometheus UI", url: "http://localhost:9090", desc: "Query and graph metrics" },
            { label: "Grafana dashboards", url: "http://localhost:3000", desc: "Visualization (admin / admin)" },
            { label: "Jaeger traces", url: "http://localhost:16686", desc: "Distributed tracing UI" },
            { label: "Dataset schemas", url: "http://localhost:8000/datasets/", desc: "All registered datasets" },
          ].map((link) => (
            <a
              key={link.url}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-3 rounded-lg border border-slate-200 p-4 transition-colors hover:border-violet-300 hover:bg-violet-50/30"
            >
              <ExternalLink size={16} className="mt-0.5 shrink-0 text-accent-muted group-hover:text-violet-600" />
              <div>
                <p className="text-sm font-medium text-slate-800 group-hover:text-violet-700">{link.label}</p>
                <p className="mt-0.5 text-xs text-accent-muted">{link.desc}</p>
              </div>
            </a>
          ))}
        </div>
      </Card>
    </div>
  );
}
