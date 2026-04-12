// dashboard/src/layouts/DashboardLayout.tsx

import { useEffect } from "react";
import { clsx } from "clsx";
import {
  Activity,
  BarChart3,
  Database,
  Shield,
  UserSearch,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useDatasets, useHealth } from "@/hooks/useApi";
import { useActiveDataset } from "@/hooks/useDatasetContext";

const NAV_ITEMS = [
  { to: "/", icon: BarChart3, label: "Global evaluation" },
  { to: "/predict", icon: UserSearch, label: "Local prediction" },
  { to: "/monitoring", icon: Activity, label: "Monitoring" },
] as const;

export default function DashboardLayout() {
  const health = useHealth();
  const { data: datasetsData } = useDatasets();
  const { activeDatasetId, setActiveDatasetId } = useActiveDataset();
  const isHealthy = health.data?.status === "healthy";

  // Auto-select first loaded dataset
  useEffect(() => {
    if (!activeDatasetId && health.data?.loaded_datasets?.length) {
      setActiveDatasetId(health.data.loaded_datasets[0]);
    }
  }, [activeDatasetId, health.data, setActiveDatasetId]);

  const datasets = datasetsData?.datasets ?? [];

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 z-30 flex h-screen w-60 flex-col border-r border-slate-200 bg-white">
        {/* Brand */}
        <div className="flex items-center gap-2.5 border-b border-slate-200 px-5 py-4">
          <Shield size={22} className="text-violet-600" />
          <div>
            <h1 className="text-sm font-bold tracking-tight text-slate-900">
              Credit Risk XAI
            </h1>
            <span className="text-[10px] font-medium uppercase tracking-wider text-accent-muted">
              Multi-dataset
            </span>
          </div>
        </div>

        {/* Dataset selector */}
        {datasets.length > 0 && (
          <div className="border-b border-slate-200 px-3 py-3">
            <label className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-accent-muted">
              <Database size={12} />
              Active dataset
            </label>
            <select
              value={activeDatasetId ?? ""}
              onChange={(e) => setActiveDatasetId(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-800 outline-none transition-colors focus:border-violet-400 focus:ring-2 focus:ring-violet-100"
            >
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name}
                </option>
              ))}
            </select>
            {activeDatasetId && (
              <p className="mt-1 text-[10px] text-accent-muted">
                {datasets.find((d) => d.id === activeDatasetId)?.n_features ?? 0} features
              </p>
            )}
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 space-y-0.5 px-3 pt-4">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-violet-50 text-violet-700"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900",
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Health status footer */}
        <div className="border-t border-slate-200 px-5 py-3">
          <div className="flex items-center gap-2 text-xs text-accent-muted">
            <span
              className={clsx(
                "h-2 w-2 rounded-full",
                health.isLoading
                  ? "bg-amber-400 animate-pulse"
                  : isHealthy
                    ? "bg-emerald-500"
                    : "bg-red-500",
              )}
            />
            {health.isLoading
              ? "Connecting…"
              : isHealthy
                ? `v${health.data?.version} · ${health.data?.loaded_datasets?.length ?? 0} models`
                : "API unreachable"}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-60 flex-1 p-6 lg:p-8">
        <Outlet />
      </main>
    </div>
  );
}
