import { useState } from "react";
import { ScrollText } from "lucide-react";
import { api, useQuery } from "@/api";
import { shortTime } from "@/engine/format";
import { Input, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/primitives";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { EmptyState, ErrorBox, Loading, PageHeader, StatusPill } from "@/components/common";

const TYPES = ["all", "domain", "strategy", "version", "decision"];

export default function Audit() {
  const [type, setType] = useState("all");
  const [actor, setActor] = useState("");
  const [resource, setResource] = useState("");
  const key = `audit:${type}:${actor}:${resource}`;
  const q = useQuery(key, () => api.auditLogs({ resource_type: type === "all" ? undefined : type, actor: actor || undefined, resource_id: resource || undefined, limit: 300 }));
  return (
    <div>
      <PageHeader icon={ScrollText} title="Audit log" description="Who did what, to which resource and version, and whether it succeeded. Failed attempts are recorded too." />
      <div className="mb-4 flex flex-wrap gap-2">
        <Select value={type} onValueChange={setType}>
          <SelectTrigger className="w-40" aria-label="Resource type"><SelectValue /></SelectTrigger>
          <SelectContent>{TYPES.map((t) => <SelectItem key={t} value={t}>{t === "all" ? "All resources" : t}</SelectItem>)}</SelectContent>
        </Select>
        <Input className="w-48" placeholder="Actor" value={actor} onChange={(e) => setActor(e.target.value)} />
        <Input className="w-56" placeholder="Resource id (e.g. hospitality-v1)" value={resource} onChange={(e) => setResource(e.target.value)} />
      </div>
      {q.error && <ErrorBox error={q.error} onRetry={q.reload} />}
      {!q.data ? <Loading /> : q.data.length === 0 ? <EmptyState title="No audit entries match" /> : (
        <Card><CardContent className="p-0 pb-2">
          <Table>
            <TableHeader><TableRow><TableHead>When</TableHead><TableHead>Actor</TableHead><TableHead>Operation</TableHead><TableHead>Resource</TableHead><TableHead>Version</TableHead><TableHead>Outcome</TableHead><TableHead>Details</TableHead></TableRow></TableHeader>
            <TableBody>
              {q.data.map((e) => (
                <TableRow key={e.id}>
                  <TableCell className="whitespace-nowrap text-xs">{shortTime(e.at)}</TableCell>
                  <TableCell className="text-xs">{e.actor}<div className="text-muted-foreground">{e.role}</div></TableCell>
                  <TableCell className="text-xs font-medium">{e.operation}</TableCell>
                  <TableCell className="text-xs">{e.resource_type}<div className="text-muted-foreground">{e.resource_id}</div></TableCell>
                  <TableCell className="text-xs">{e.version ?? "—"}</TableCell>
                  <TableCell><StatusPill status={e.outcome} /></TableCell>
                  <TableCell className="max-w-[360px] truncate font-mono text-[11px] text-muted-foreground" title={JSON.stringify(e.details)}>{JSON.stringify(e.details)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent></Card>
      )}
    </div>
  );
}
