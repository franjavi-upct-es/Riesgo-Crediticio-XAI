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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import ThemeToggle from "@/components/ui/ThemeToggle";
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

  useEffect(() => {
    if (!activeDatasetId && health.data?.loaded_datasets?.length) {
      setActiveDatasetId(health.data.loaded_datasets[0]);
    }
  }, [activeDatasetId, health.data, setActiveDatasetId]);

  const datasets = datasetsData?.datasets ?? [];
  const activeDataset = datasets.find((d) => d.id === activeDatasetId);

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 z-30 flex h-screen w-60 flex-col border-r border-border bg-card transition-colors">
        {/* Brand */}
        <div className="flex items-center gap-2.5 border-b border-border px-5 py-4">
          <Shield size={22} className="text-primary" />
          <div>
            <h1 className="text-sm font-bold tracking-tight text-foreground">
              Credit Risk XAI
            </h1>
            <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Multi-dataset
            </span>
          </div>
        </div>

        {/* Dataset selector */}
        {datasets.length > 0 && (
          <div className="border-b border-border px-3 py-3">
            <label className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              <Database size={12} />
              Active dataset
            </label>
            <Select
              value={activeDatasetId ?? ""}
              onValueChange={(v) => setActiveDatasetId(v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder="Select a dataset…" />
              </SelectTrigger>
              <SelectContent>
                {datasets.map((ds) => (
                  <SelectItem key={ds.id} value={ds.id} className="text-xs">
                    {ds.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {activeDataset && (
              <p className="mt-1.5 text-[10px] text-muted-foreground">
                {activeDataset.n_features} features
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
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer: health + theme toggle */}
        <div className="flex items-center justify-between border-t border-border px-4 py-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span
              className={clsx(
                "h-2 w-2 rounded-full",
                health.isLoading
                  ? "animate-pulse bg-amber-400"
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
          <ThemeToggle />
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-60 flex-1 p-6 lg:p-8">
        <Outlet />
      </main>
    </div>
  );
}
