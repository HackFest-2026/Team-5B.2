import { lazy, Suspense } from "react";
import { BrowserRouter, Link, Navigate, Route, Routes } from "react-router-dom";
import { Toaster } from "sonner";
import { AppLayout } from "@/components/layout/AppLayout";
import { EmptyState, Loading } from "@/components/common";
import { TooltipProvider } from "@/components/ui/overlays";
import { Button } from "@/components/ui/button";

const Dashboard = lazy(() => import("@/pages/business/Dashboard"));
const Domains = lazy(() => import("@/pages/business/Domains"));
const Strategy = lazy(() => import("@/pages/business/Strategy"));
const Simulator = lazy(() => import("@/pages/business/Simulator"));
const Versions = lazy(() => import("@/pages/business/Versions"));
const Decisions = lazy(() => import("@/pages/business/Decisions"));
const Audit = lazy(() => import("@/pages/business/Audit"));
const Portal = lazy(() => import("@/pages/portal/Portal"));

function NotFound() {
  return <EmptyState title="Page not found" description="That screen doesn't exist." action={<Button asChild><Link to="/business">Back to dashboard</Link></Button>} />;
}

export default function App() {
  return (
    <TooltipProvider delayDuration={200}>
      <BrowserRouter>
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<Navigate to="/business" replace />} />
            <Route path="/portal" element={<Portal />} />
            <Route path="/business" element={<AppLayout />}>
              <Route index element={<Dashboard />} />
              <Route path="domains" element={<Domains />} />
              <Route path="strategy" element={<Strategy />} />
              <Route path="simulator" element={<Simulator />} />
              <Route path="versions" element={<Versions />} />
              <Route path="decisions" element={<Decisions />} />
              <Route path="decisions/:decisionId" element={<Decisions />} />
              <Route path="audit" element={<Audit />} />
              <Route path="*" element={<NotFound />} />
            </Route>
            <Route path="*" element={<Navigate to="/business" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
      <Toaster richColors position="top-right" closeButton />
    </TooltipProvider>
  );
}
