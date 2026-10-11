import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Info, Loader2, Tag } from "lucide-react";
import { api, useQuery } from "@/api";
import type { CatalogEntry, CustomerQuote, Scalar } from "@/types";
import { formatPrice } from "@/engine/format";
import { useUi } from "@/store/ui";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/overlays";
import { ErrorBox, Field, Loading, toastError } from "@/components/common";
import { AttributeInputs, cleanValues } from "@/components/pricing/AttributeInputs";

function QuoteResult({ q, entry }: { q: CustomerQuote; entry: CatalogEntry }) {
  if (q.status === "NO_VALID_PRICE") {
    return (
      <Card className="border-warning/50"><CardContent className="p-6 text-center">
        <div className="text-lg font-medium">This option isn't available for these details.</div>
        <div className="mt-1 text-sm text-muted-foreground">Try different dates or another {entry.product_noun.toLowerCase()}.</div>
      </CardContent></Card>
    );
  }
  return (
    <Card data-testid="customer-quote">
      <CardContent className="space-y-4 p-6">
        <div>
          <div className="text-sm text-muted-foreground">Your price</div>
          <div className="tabular text-4xl font-semibold">{formatPrice(q.final_price, q, true)}</div>
          <div className="text-sm text-muted-foreground">{q.unit_label}</div>
        </div>
        <div>
          <div className="mb-1 text-sm font-medium">What shaped this price</div>
          <ul className="space-y-1 text-sm">{q.factors.map((f, i) => <li key={i} className="flex items-center gap-2"><Tag className="h-3.5 w-3.5 text-primary" /> {f}</li>)}</ul>
        </div>
        {q.alternatives && (
          <div>
            <div className="mb-1 text-sm font-medium">Other available prices</div>
            <div className="flex flex-wrap gap-2">{q.alternatives.map((a) => <span key={a.price} className="rounded-md border px-3 py-1 text-sm tabular">{formatPrice(a.price, q)}</span>)}</div>
          </div>
        )}
        <div className="flex items-start gap-2 rounded-md bg-muted p-3 text-xs text-muted-foreground"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {q.notice} Reference: <span className="font-mono">{q.decision_id}</span></div>
      </CardContent>
    </Card>
  );
}

export default function Portal() {
  const catalog = useQuery("catalog", api.catalog);
  const { users, setUser } = useUi();
  const [domainId, setDomainId] = useState("");
  const [productId, setProductId] = useState("");
  const [values, setValues] = useState<Record<string, Scalar>>({});
  const [quote, setQuote] = useState<CustomerQuote | null>(null);
  const [busy, setBusy] = useState(false);
  const entry = useMemo(() => catalog.data?.find((c) => c.domain_id === domainId) ?? catalog.data?.[0], [catalog.data, domainId]);
  useEffect(() => { if (entry) { setDomainId(entry.domain_id); setProductId(entry.products[0]?.id ?? ""); setValues({}); setQuote(null); } }, [entry?.domain_id]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="min-h-screen bg-gradient-to-b from-accent/40 to-background">
      <header className="border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex min-h-14 max-w-4xl flex-wrap items-center gap-x-3 gap-y-2 px-4 py-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">M</div>
          <div className="font-semibold">Get a price</div>
          <div className="ml-auto flex min-w-0 items-center gap-2 text-xs">
            <span className="hidden text-muted-foreground sm:inline">Signed in as</span>
            <Input className="h-8 w-28 sm:w-36" value={users.customer} onChange={(e) => setUser("customer", e.target.value || "guest")} aria-label="Customer id" />
            <Button asChild variant="ghost" size="sm"><Link to="/business"><ArrowLeft /> <span className="hidden sm:inline">Business portal</span></Link></Button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-4xl space-y-4 p-4 lg:py-8">
        {catalog.error && <ErrorBox error={catalog.error} onRetry={catalog.reload} />}
        {!catalog.data && !catalog.error && <Loading />}
        {catalog.data && catalog.data.length === 0 && <Card><CardContent className="p-6 text-sm">Nothing is on sale yet.</CardContent></Card>}
        {entry && (
          <Card>
            <CardHeader><CardTitle>{entry.name}</CardTitle><CardDescription>{entry.description}</CardDescription></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Service">
                  <Select value={entry.domain_id} onValueChange={setDomainId}>
                    <SelectTrigger aria-label="Service"><SelectValue /></SelectTrigger>
                    <SelectContent>{catalog.data!.map((c) => <SelectItem key={c.domain_id} value={c.domain_id}>{c.name}</SelectItem>)}</SelectContent>
                  </Select>
                </Field>
                <Field label={entry.product_noun}>
                  <Select value={productId} onValueChange={(v) => { setProductId(v); setQuote(null); }}>
                    <SelectTrigger aria-label={entry.product_noun}><SelectValue /></SelectTrigger>
                    <SelectContent>{entry.products.map((p) => <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>)}</SelectContent>
                  </Select>
                </Field>
              </div>
              <AttributeInputs attributes={entry.inputs} values={values} onChange={(k, v) => { setValues({ ...values, [k]: v }); setQuote(null); }} />
              <Button size="lg" disabled={busy || !productId} onClick={async () => {
                setBusy(true);
                try { setQuote(await api.customerQuote({ domain_id: entry.domain_id, product_id: productId, attributes: cleanValues(values) })); } catch (e) { toastError(e, "Couldn't get a price"); } finally { setBusy(false); }
              }}>{busy && <Loader2 className="animate-spin" />} Get price</Button>
            </CardContent>
          </Card>
        )}
        {quote && entry && <QuoteResult q={quote} entry={entry} />}
      </main>
    </div>
  );
}
