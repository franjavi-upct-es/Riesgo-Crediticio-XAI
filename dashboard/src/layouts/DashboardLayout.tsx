// dashboard/src/layouts/DashboardLayout.tsx

import { clsx } from "clsx";
import { Activity, BarChart3, Shield, UserSearch } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useHealth } from "@/hooks/useApi";

const NAV_ITEMS = [
  { to: "/", icon: BarChart3, label: "Global evaluation" },
  { to: "/predict", icon: UserSearch, label: "Local prediction" },
  { to: "/monitoring", icon: Activity, label: "Monitoring" },
] as const;

export default function DashboardLayout() {
  const health = useHealth();
  const isHealthy = health.data?.status == "healthy";

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 z-30 flex h-screen w-60 flex-col border-r border-slate-200 bg-white">
        {/* Brand */}
        <div className="flex items-center gap-2.5 border-b border-slate-200 px-5 py-4">
          <Shield size={12} className="text-violet-600" />
          <div>
            <h1 className="text-sm font-bold tracking-tight text-slate-900">
              Credit Risk XAI
            </h1>
            <span className="text-[10px] font-medium uppercase tracking-wider text-accent-muted">
              Dashboard
            </span>
          </div>
        </div>

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
              ? "Connecting..."
              : isHealthy
                ? `API healthy · v${health.data?.version}`
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
