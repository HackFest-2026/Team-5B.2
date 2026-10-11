import { useEffect, useState } from "react";
import { FileJson, Layers, Plus, Save, Sparkles, Upload } from "lucide-react";
import { toast } from "sonner";
import { api, invalidate, useQuery, type ApiError } from "@/api";
import { useCurrentDomain } from "@/api/hooks";
import type { DomainConfig } from "@/types";
import { Button } from "@/components/ui/button";
import { Badge, Textarea } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/overlays";
import { AsyncButton, EmptyState, ErrorBox, Loading, PageHeader, StatusPill, toastError } from "@/components/common";
import { useUi } from "@/store/ui";
import { BLANK_DOMAIN, DomainEditor } from "./DomainEditor";

function ImportDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<ApiError | null>(null);
  const setDomainId = useUi((s) => s.setDomainId);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Import a domain configuration</DialogTitle>
          <DialogDescription>Paste a DomainConfig, or a {"{ domain, strategy }"} bundle (see the files in <code>examples/domains/</code>). No engine code changes are needed.</DialogDescription>
        </DialogHeader>
        <input type="file" accept="application/json" className="text-sm" onChange={async (e) => { const f = e.target.files?.[0]; if (f) setText(await f.text()); }} />
        <Textarea className="h-72 font-mono text-xs" value={text} onChange={(e) => setText(e.target.value)} placeholder='{"domain": {...}, "strategy": {...}}' />
        {error && <ErrorBox error={error} />}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <AsyncButton onClick={async () => {
            setError(null);
            let payload: unknown;
            try { payload = JSON.parse(text); } catch { toast.error("That isn't valid JSON."); return; }
            try {
              const d = await api.createDomain(payload);
              invalidate("domain");
              setDomainId(d.id);
              toast.success(`Domain "${d.name}" created. Validate and publish its strategy next.`);
              onOpenChange(false);
              setText("");
            } catch (e) { setError(e as ApiError); }
          }}><Upload /> Import</AsyncButton>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function NewDomainDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [description, setDescription] = useState("");
  const [draft, setDraft] = useState<DomainConfig | null>(null);
  const [provider, setProvider] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const setDomainId = useUi((s) => s.setDomainId);
  const close = () => { onOpenChange(false); setDraft(null); setDescription(""); setError(null); };
  return (
    <Dialog open={open} onOpenChange={(o) => (o ? onOpenChange(o) : close())}>
      <DialogContent className="max-h-[90vh] max-w-6xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>New pricing domain</DialogTitle>
          <DialogDescription>Describe what you sell. The generator proposes attributes and products; you review every field before saving.</DialogDescription>
        </DialogHeader>
        {!draft ? (
          <div className="space-y-3">
            <Textarea className="h-28" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="e.g. Coworking desks rented per day. Demand varies by weekday, members are loyal, competitors nearby, people book in advance." />
            <div className="flex flex-wrap gap-2">
              <AsyncButton disabled={!description.trim()} onClick={async () => {
                try { const r = await api.suggestDomain(description); setDraft(r.domain); setProvider(r.provider); } catch (e) { toastError(e); }
              }}><Sparkles /> Suggest from description</AsyncButton>
              <Button variant="outline" onClick={() => { setDraft(structuredClone(BLANK_DOMAIN)); setProvider(null); }}>Start blank</Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {provider && <Badge variant={provider === "mock" ? "warning" : "info"}>{provider === "mock" ? "Drafted by the mock generator (no LLM configured)" : `Drafted by ${provider}`}</Badge>}
            <DomainEditor value={draft} onChange={setDraft} />
            {error && <ErrorBox error={error} />}
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={close}>Cancel</Button>
          {draft && (
            <AsyncButton onClick={async () => {
              setError(null);
              try {
                const d = await api.createDomain(draft);
                invalidate("domain");
                setDomainId(d.id);
                toast.success(`Domain "${d.name}" created. Next: generate a strategy in Strategy Studio.`);
                close();
              } catch (e) { setError(e as ApiError); }
            }}><Save /> Create domain</AsyncButton>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function Domains() {
  const domains = useQuery("domains", api.domains);
  const { id, domain, loading } = useCurrentDomain();
  const setDomainId = useUi((s) => s.setDomainId);
  const [edit, setEdit] = useState<DomainConfig | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [newOpen, setNewOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  useEffect(() => { setEdit(domain ? structuredClone(domain) : null); setError(null); }, [domain]);

  return (
    <div>
      <PageHeader icon={Layers} title="Domains" description="What is priced and which inputs exist. Industry vocabulary lives only here, in configuration."
        actions={<>
          <Button variant="outline" onClick={() => setImportOpen(true)}><FileJson /> Import JSON</Button>
          <Button onClick={() => setNewOpen(true)}><Plus /> New domain</Button>
        </>} />
      {domains.error && <ErrorBox error={domains.error} onRetry={domains.reload} />}
      <div className="mb-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {(domains.data ?? []).map((d) => (
          <button key={d.id} onClick={() => setDomainId(d.id)} className={`rounded-lg border p-4 text-left transition-colors hover:bg-muted/50 ${d.id === id ? "border-primary ring-1 ring-primary" : ""}`}>
            <div className="flex items-center justify-between"><div className="font-medium">{d.name}</div><Badge variant="outline">{d.currency}</Badge></div>
            <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">{d.description || "No description"}</div>
            <div className="mt-2 flex gap-1">{d.active_version ? <StatusPill status="ACTIVE" /> : <Badge variant="secondary">unpublished</Badge>}{d.draft_status && <StatusPill status={d.draft_status} />}</div>
          </button>
        ))}
      </div>
      {loading && !edit && <Loading />}
      {!loading && !edit && <EmptyState title="No domain selected" description="Create a domain or import a configuration." />}
      {edit && (
        <Card>
          <CardHeader className="flex-row items-start justify-between space-y-0">
            <div><CardTitle>{edit.name}</CardTitle><CardDescription>Changing the domain marks its strategy draft as needing validation again. Published versions keep their own frozen copy.</CardDescription></div>
            <AsyncButton onClick={async () => {
              setError(null);
              try { await api.updateDomain(edit.id, edit); invalidate("domain"); toast.success("Domain saved"); } catch (e) { setError(e as ApiError); }
            }}><Save /> Save domain</AsyncButton>
          </CardHeader>
          <CardContent className="space-y-3">
            {error && <ErrorBox error={error} />}
            <DomainEditor value={edit} onChange={setEdit} lockId />
          </CardContent>
        </Card>
      )}
      <NewDomainDialog open={newOpen} onOpenChange={setNewOpen} />
      <ImportDialog open={importOpen} onOpenChange={setImportOpen} />
    </div>
  );
}
