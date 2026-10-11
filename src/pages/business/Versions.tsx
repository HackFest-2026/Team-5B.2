import { useState } from "react";
import { Link } from "react-router-dom";
import { GitBranch, GitCompare, RotateCcw, Rocket, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { api, invalidate, useQuery } from "@/api";
import { useCurrentDomain } from "@/api/hooks";
import type { StrategyDiff, VersionView } from "@/types";
import { shortTime } from "@/engine/format";
import { useUi } from "@/store/ui";
import { Button } from "@/components/ui/button";
import { Badge, Input, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/overlays";
import { AsyncButton, ConfirmDialog, EmptyState, ErrorBox, Loading, PageHeader, StatusPill, toastError } from "@/components/common";

function DiffView({ d }: { d: StrategyDiff }) {
  if (!d.has_changes) return <div className="text-sm text-muted-foreground">No differences.</div>;
  return (
    <div className="space-y-2 text-sm">
      {d.added.map((r) => <div key={r.id} className="rounded-md border-l-4 border-success bg-success/5 p-2">+ Added <b>{r.name}</b> ({r.id})</div>)}
      {d.removed.map((r) => <div key={r.id} className="rounded-md border-l-4 border-destructive bg-destructive/5 p-2">− Removed <b>{r.name}</b> ({r.id})</div>)}
      {d.changed.map((c) => (
        <div key={c.after.id} className="rounded-md border-l-4 border-warning bg-warning/5 p-2">
          ~ Changed <b>{c.after.name}</b> ({c.after.id}): {c.fields.join(", ")}
          {c.fields.includes("action") && <div className="text-xs text-muted-foreground">action {JSON.stringify(c.before.action)} → {JSON.stringify(c.after.action)}</div>}
        </div>
      ))}
      {d.constraints_changed && <div className="rounded-md border-l-4 border-warning bg-warning/5 p-2">~ Constraints: {JSON.stringify(d.constraints_before)} → {JSON.stringify(d.constraints_after)}</div>}
      {d.settings.map((s) => <div key={s.field} className="rounded-md border-l-4 border-warning bg-warning/5 p-2">~ {s.field}</div>)}
    </div>
  );
}

export default function Versions() {
  const { id } = useCurrentDomain();
  const role = useUi((s) => s.role);
  const user = useUi((s) => s.users[s.role]);
  const versions = useQuery(id ? `versions:${id}` : null, () => api.versions(id!));
  const draft = useQuery(id ? `draft:${id}` : null, () => api.draft(id!));
  const [notes, setNotes] = useState("");
  const [confirm, setConfirm] = useState<{ v: VersionView; op: "publish" | "rollback" } | null>(null);
  const [a, setA] = useState("draft");
  const [b, setB] = useState("");
  const [diff, setDiff] = useState<StrategyDiff | null>(null);
  if (!id) return <EmptyState title="No domain" />;
  if (versions.error) return <ErrorBox error={versions.error} onRetry={versions.reload} />;
  if (!versions.data || !draft.data) return <Loading />;

  const canApprove = role !== "analyst";
  const selfEdited = draft.data.updated_by === user;
  const approveBlock = !canApprove ? "Only a manager or admin can approve." : selfEdited ? `You (${user}) last edited this draft; maker-checker needs a different approver.` : draft.data.status !== "VALIDATED" ? "Validate the draft first (Strategy Studio)." : null;
  const refresh = () => { invalidate(`versions:${id}`); invalidate(`draft:${id}`); invalidate("domains"); invalidate("audit"); };

  return (
    <div className="space-y-4">
      <PageHeader icon={GitBranch} title="Versions & Approvals" description="DRAFT → VALIDATED → APPROVED → ACTIVE → RETIRED. Approved versions are immutable; every decision records the exact version that priced it." />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">Current draft <StatusPill status={draft.data.status} /></CardTitle>
          <CardDescription>Last edited by {draft.data.updated_by} at {shortTime(draft.data.updated_at)} · hash {draft.data.content_hash}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2">
          <Input className="max-w-md" placeholder="Approval notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} />
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <AsyncButton disabled={!!approveBlock} onClick={async () => {
                  try { const v = await api.approve(id, notes); setNotes(""); refresh(); toast.success(`${v.id} approved. Publish it to make it live.`); } catch (e) { toastError(e); }
                }}><ShieldCheck /> Approve draft</AsyncButton>
              </span>
            </TooltipTrigger>
            {approveBlock && <TooltipContent>{approveBlock}</TooltipContent>}
          </Tooltip>
          {approveBlock && <span className="text-xs text-muted-foreground">{approveBlock}</span>}
          {draft.data.status !== "VALIDATED" && <Button asChild variant="link" size="sm"><Link to="/business/strategy">Go to Strategy Studio</Link></Button>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">Versions</CardTitle></CardHeader>
        <CardContent className="p-0 pb-2">
          {versions.data.length === 0 ? <div className="p-5"><EmptyState title="No versions yet" description="Approve a validated draft to create v1." /></div> : (
            <Table>
              <TableHeader><TableRow><TableHead>Version</TableHead><TableHead>Status</TableHead><TableHead>Rules</TableHead><TableHead>Authored / approved</TableHead><TableHead>Published</TableHead><TableHead>Notes</TableHead><TableHead /></TableRow></TableHeader>
              <TableBody>
                {versions.data.map((v) => (
                  <TableRow key={v.id}>
                    <TableCell className="font-medium">{v.id}<div className="text-[11px] text-muted-foreground">{v.content_hash}</div></TableCell>
                    <TableCell><StatusPill status={v.status} /></TableCell>
                    <TableCell>{v.strategy.rules.length}</TableCell>
                    <TableCell className="text-xs">{v.created_by} / {v.approved_by}<div className="text-muted-foreground">{shortTime(v.approved_at)}</div></TableCell>
                    <TableCell className="text-xs">{shortTime(v.published_at)}</TableCell>
                    <TableCell className="max-w-[200px] truncate text-xs">{v.notes}</TableCell>
                    <TableCell className="whitespace-nowrap text-right">
                      {v.status === "APPROVED" && <Button size="sm" disabled={!canApprove} onClick={() => setConfirm({ v, op: "publish" })}><Rocket /> Publish</Button>}
                      {v.status === "RETIRED" && <Button size="sm" variant="outline" disabled={!canApprove} onClick={() => setConfirm({ v, op: "rollback" })}><RotateCcw /> Roll back to this</Button>}
                      {v.status === "ACTIVE" && <Badge variant="success">live</Badge>}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2 text-base"><GitCompare className="h-4 w-4" /> Compare</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            {[a, b].map((val, i) => (
              <Select key={i} value={val} onValueChange={i === 0 ? setA : setB}>
                <SelectTrigger className="w-48" aria-label={i === 0 ? "From" : "To"}><SelectValue placeholder="Choose…" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="draft">Current draft</SelectItem>
                  {(versions.data ?? []).map((v) => <SelectItem key={v.id} value={v.id}>{v.id}</SelectItem>)}
                </SelectContent>
              </Select>
            ))}
            <AsyncButton variant="outline" disabled={!b || a === b} onClick={async () => { try { setDiff(await api.diff(id, b, a)); } catch (e) { toastError(e); } }}>Show changes</AsyncButton>
          </div>
          {diff && <DiffView d={diff} />}
        </CardContent>
      </Card>

      <ConfirmDialog
        open={!!confirm}
        onOpenChange={(o) => !o && setConfirm(null)}
        title={confirm?.op === "publish" ? `Publish ${confirm.v.id}?` : `Roll back to ${confirm?.v.id}?`}
        description="The current active version is retired in the same transaction. New quotes immediately use this version; past decisions keep the version they were priced with."
        confirmLabel={confirm?.op === "publish" ? "Publish" : "Roll back"}
        onConfirm={async () => {
          if (!confirm) return;
          try {
            await (confirm.op === "publish" ? api.publish(id, confirm.v.id) : api.rollback(id, confirm.v.id));
            refresh();
            toast.success(`${confirm.v.id} is now live`);
          } catch (e) { toastError(e); }
        }}
      />
    </div>
  );
}
