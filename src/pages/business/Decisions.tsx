import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, FileSearch, Search } from "lucide-react";
import { api, useQuery } from "@/api";
import { formatPrice, shortTime } from "@/engine/format";
import { Button } from "@/components/ui/button";
import { Input, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/primitives";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { EmptyState, ErrorBox, Loading, PageHeader, ShadowBadge, StatusPill } from "@/components/common";
import { DecisionDetail } from "@/components/pricing";

function DecisionPage({ id }: { id: string }) {
  const q = useQuery(`decision:${id}`, () => api.decision(id));
  const pending = q.data?.shadow_validation?.status === "PENDING";
  useQuery(pending ? `decision-poll:${id}` : null, async () => { q.reload(); return null; }, { pollMs: pending ? 1500 : undefined });
  return (
    <div className="space-y-4">
      <PageHeader icon={FileSearch} title="Decision" description={<span className="font-mono text-xs">{id}</span>}
        actions={<Button asChild variant="outline"><Link to="/business/decisions"><ArrowLeft /> All decisions</Link></Button>} />
      {q.error && <ErrorBox error={q.error} onRetry={q.reload} />}
      {!q.data && !q.error && <Loading />}
      {q.data && (
        <>
          <Card><CardContent className="grid gap-2 p-4 text-xs sm:grid-cols-2 lg:grid-cols-4">
            <div><span className="text-muted-foreground">Domain / {"product"}:</span> {q.data.domain_id} / {q.data.product_id}</div>
            <div><span className="text-muted-foreground">Pricing date:</span> {q.data.request.as_of}</div>
            <div><span className="text-muted-foreground">Customer:</span> {q.data.request.customer_id ?? "anonymous"}</div>
            <div><span className="text-muted-foreground">Created:</span> {shortTime(q.data.created_at)}</div>
            <div className="sm:col-span-2 lg:col-span-4"><span className="text-muted-foreground">Inputs:</span> <span className="font-mono">{JSON.stringify(q.data.request.attributes)}</span></div>
          </CardContent></Card>
          <DecisionDetail d={q.data} />
        </>
      )}
    </div>
  );
}

export default function Decisions() {
  const { decisionId } = useParams();
  const navigate = useNavigate();
  const [domain, setDomain] = useState("all");
  const [lookup, setLookup] = useState("");
  const domains = useQuery("domains", api.domains);
  const list = useQuery(decisionId ? null : `decisions:${domain}`, () => api.decisions({ domain_id: domain === "all" ? undefined : domain, limit: 200 }));
  if (decisionId) return <DecisionPage id={decisionId} />;

  return (
    <div>
      <PageHeader icon={FileSearch} title="Decisions" description="Every quote, traceable to its strategy version, inputs, rules, constraints, optimizer output and shadow validation." />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Select value={domain} onValueChange={setDomain}>
          <SelectTrigger className="w-48" aria-label="Filter by domain"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All domains</SelectItem>
            {(domains.data ?? []).map((d) => <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
          </SelectContent>
        </Select>
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (lookup.trim()) navigate(`/business/decisions/${lookup.trim()}`); }}>
          <Input className="w-64" placeholder="Look up decision id (D-…)" value={lookup} onChange={(e) => setLookup(e.target.value)} />
          <Button variant="outline" type="submit"><Search /> Open</Button>
        </form>
      </div>
      {list.error && <ErrorBox error={list.error} onRetry={list.reload} />}
      {!list.data ? <Loading /> : list.data.length === 0 ? <EmptyState title="No decisions yet" description="Quotes from the user portal or the API appear here." /> : (
        <Card><CardContent className="p-0 pb-2">
          <Table>
            <TableHeader><TableRow><TableHead>Decision</TableHead><TableHead>Domain · product</TableHead><TableHead>Version</TableHead><TableHead>Status</TableHead><TableHead className="text-right">Price</TableHead><TableHead>Validation</TableHead><TableHead>Created</TableHead></TableRow></TableHeader>
            <TableBody>
              {list.data.map((d) => (
                <TableRow key={d.decision_id} className="cursor-pointer" onClick={() => navigate(`/business/decisions/${d.decision_id}`)}>
                  <TableCell className="font-mono text-xs">{d.decision_id}</TableCell>
                  <TableCell>{d.domain_id} · {d.product_id}</TableCell>
                  <TableCell className="text-xs">{d.version_id}</TableCell>
                  <TableCell><StatusPill status={d.status} /></TableCell>
                  <TableCell className="tabular text-right font-medium">{d.final_price === null ? "—" : formatPrice(d.final_price, d.unit_prefix == null ? { currency: d.currency, decimals: d.decimals } : d)}</TableCell>
                  <TableCell><ShadowBadge status={d.validation_status} /></TableCell>
                  <TableCell className="text-xs">{shortTime(d.created_at)}<div className="text-muted-foreground">{d.created_by}</div></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent></Card>
      )}
    </div>
  );
}
