// dashboard/src/App.tsx

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import DashboardLayout from "@/layouts/DashboardLayout";
import GlobalEvaluation from "@/pages/GlobalEvaluation";
import LocalPrediction from "@/pages/LocalPrediction";
import Monitoring from "@/pages/Monitoring";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route index element={<GlobalEvaluation />} />
            <Route path="predict" element={<LocalPrediction />} />
            <Route path="monitoring" element={<Monitoring />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
