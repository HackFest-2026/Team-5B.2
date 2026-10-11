import { Link } from "react-router-dom";
import { AlertTriangle, FileSearch, GitBranch, Layers, LayoutDashboard, ShieldAlert } from "lucide-react";
import { api, useQuery } from "@/api";
import { formatPrice, shortTime } from "@/engine/format";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/primitives";
import { EmptyState, ErrorBox, Loading, PageHeader, ProviderBadges, ShadowBadge, StatCard, StatusPill } from "@/components/common";
import { useUi } from "@/store/ui";

export default function Dashboard() {
  const domains = useQuery("domains", api.domains);
  const decisions = useQuery("decisions:all", () => api.decisions({ limit: 200 }));
  const setDomainId = useUi((s) => s.setDomainId);
  if (domains.error) return <ErrorBox error={domains.error} onRetry={domains.reload} />;
  if (!domains.data) return <Loading />;
  const ds = decisions.data ?? [];
  const noValid = ds.filter((d) => d.status === "NO_VALID_PRICE").length;
  const review = ds.filter((d) => d.validation_status === "REVIEW_REQUIRED" || d.validation_status === "FAIL");

  return (
    <div>
      <PageHeader icon={LayoutDashboard} title="Dashboard" description="One pricing engine, many domains. Domain behaviour lives in validated configuration; every price is computed by the same deterministic core." actions={<ProviderBadges />} />
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Domains" value={domains.data.length} hint={`${domains.data.filter((d) => d.active_version).length} published`} icon={Layers} />
        <StatCard label="Pricing decisions" value={ds.length} hint="most recent 200" icon={FileSearch} />
        <StatCard label="No valid price" value={noValid} hint="constraints refused every candidate" icon={AlertTriangle} tone={noValid ? "bad" : undefined} />
        <StatCard label="Decisions needing review" value={review.length} hint="shadow validation FAIL / REVIEW" icon={ShieldAlert} tone={review.length ? "bad" : "good"} />
      </div>
      <div className="grid gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-3">
          <CardHeader><CardTitle className="text-base">Domains</CardTitle><CardDescription>Same engine, different configuration.</CardDescription></CardHeader>
          <CardContent className="p-0 pb-2">
            <Table>
              <TableHeader><TableRow><TableHead>Domain</TableHead><TableHead>Currency</TableHead><TableHead>Active version</TableHead><TableHead>Draft</TableHead><TableHead /></TableRow></TableHeader>
              <TableBody>
                {domains.data.map((d) => (
                  <TableRow key={d.id}>
                    <TableCell><div className="font-medium">{d.name}</div><div className="max-w-[280px] truncate text-xs text-muted-foreground">{d.description}</div></TableCell>
                    <TableCell>{d.currency}</TableCell>
                    <TableCell>{d.active_version ? <StatusPill status="ACTIVE" /> : <span className="text-xs text-muted-foreground">not published</span>} <span className="text-xs">{d.active_version}</span></TableCell>
                    <TableCell>{d.draft_status && <StatusPill status={d.draft_status} />}</TableCell>
                    <TableCell className="text-right">
                      <Button asChild size="sm" variant="outline" onClick={() => setDomainId(d.id)}><Link to="/business/strategy">Open</Link></Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        <Card className="xl:col-span-2">
          <CardHeader><CardTitle className="text-base">Recent decisions</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {ds.length === 0 && <EmptyState icon={GitBranch} title="No quotes yet" description="Run a quote from the Simulator or the user portal." />}
            {ds.slice(0, 8).map((d) => (
              <Link key={d.decision_id} to={`/business/decisions/${d.decision_id}`} className="flex items-center justify-between rounded-md border p-2 text-sm hover:bg-muted/50">
                <div>
                  <div className="font-medium">{d.domain_id} · {d.product_id}</div>
                  <div className="text-xs text-muted-foreground">{shortTime(d.created_at)} · {d.created_by}</div>
                </div>
                <div className="text-right">
                  <div className="tabular font-medium">{d.final_price === null ? "—" : formatPrice(d.final_price, d.unit_prefix == null ? { currency: d.currency, decimals: d.decimals } : d)}</div>
                  <ShadowBadge status={d.validation_status} />
                </div>
              </Link>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
