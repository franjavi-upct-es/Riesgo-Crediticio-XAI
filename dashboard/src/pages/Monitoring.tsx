// dashboard/src/pages/Monitoring.tsx

import { useHealth } from "@/hooks/useApi";
import Card from "@/components/ui/Card";
import MetricCard from "@/components/ui/MetricCard";
import { LoadingState } from "@/components/ui/StatusStates";
import { Activity, ExternalLink, HeartPulse, Server } from "lucide-react";

export default function Monitoring() {
  const health = useHealth();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold tracking-tight text-slate-900">
          Monitoring
        </h2>
        <p className="mt-1 text-sm text-accent-muted">
          API health, Prometheus metrics, and observability links
        </p>
      </div>

      {/* Health cards */}
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
            label="Model loaded"
            value={health.data?.model_loaded ? "Yes" : "No"}
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

      {/* Quick links */}
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
              label: "Health probe",
              url: "http://localhost:8000/health",
              desc: "Readiness check (model loaded)",
            },
            {
              label: "Liveness probe",
              url: "http://localhost:8000/alive",
              desc: "Process heartbeat",
            },
          ].map((link) => (
            <a
              key={link.url}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-3 rounded-lg border border-slate-200 p-4 transition-colors hover:border-violet-300 hover:bg-violet-50/30"
            >
              <ExternalLink
                size={16}
                className="mt-0.5 shrink-0 text-accent-muted group-hover:text-violet-600"
              />
              <div>
                <p className="text-sm font-medium text-slate-800 group-hover:text-violet-700">
                  {link.label}
                </p>
                <p className="mt-0.5 text-xs text-accent-muted">{link.desc}</p>
              </div>
            </a>
          ))}
        </div>
      </Card>

      {/* Prometheus metrics info */}
      <Card
        title="Available Prometheus metrics"
        subtitle="Scraped by Prometheus at /metrics every 15s"
      >
        <div className="overflow-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-medium uppercase tracking-wider text-accent-muted">
                <th className="pb-2 pr-4">Metric</th>
                <th className="pb-2 pr-4">Type</th>
                <th className="pb-2">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {[
                [
                  "api_requests_total",
                  "counter",
                  "Total API requests (method, endpoint, status)",
                ],
                [
                  "api_request_duration_seconds",
                  "histogram",
                  "Request latency distribution",
                ],
                [
                  "api_requests_in_progress",
                  "gauge",
                  "Currently processing requests",
                ],
                ["predictions_total", "counter", "Predictions by risk label"],
                [
                  "prediction_probability",
                  "histogram",
                  "Predicted risk probability distribution",
                ],
                [
                  "shap_computation_duration_seconds",
                  "histogram",
                  "SHAP explanation computation time",
                ],
                [
                  "prediction_errors_total",
                  "counter",
                  "Prediction failures by error type",
                ],
                ["model_info", "info", "Loaded model metadata"],
              ].map(([name, type, desc]) => (
                <tr key={name}>
                  <td className="py-2 pr-4 font-mono text-xs text-slate-800">
                    {name}
                  </td>
                  <td className="py-2 pr-4">
                    <span className="rounded-md bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-600">
                      {type}
                    </span>
                  </td>
                  <td className="py-2 text-xs text-accent-subtle">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
