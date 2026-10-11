import { ExternalLink, Globe2, Wand2 } from "lucide-react";
import type { DomainConfig, MarketContext } from "@/types";
import { formatUnit } from "@/engine/format";
import { Badge, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/primitives";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AsyncButton } from "@/components/common";

const POSITION: Record<string, { label: string; variant: "success" | "warning" | "info" }> = {
  below_market: { label: "below market", variant: "info" },
  in_line: { label: "in line", variant: "success" },
  above_market: { label: "above market", variant: "warning" },
};

/** Web market research behind a generated draft. Unverified signals, shown with their sources. */
export function MarketPanel({ market, observations, domain, onApplyCompetitorPrices }: {
  market: MarketContext;
  observations?: string[];
  domain: DomainConfig;
  onApplyCompetitorPrices?: () => Promise<void>;
}) {
  const unit = { ...domain.output_unit, suffix: domain.output_unit.suffix.startsWith(" ") ? "" : domain.output_unit.suffix };
  const fmt = (v: number | null) => formatUnit(v, unit);
  const competitorAttr = domain.attributes.find((a) => a.semantic === "competitorPrice" && a.source === "inventory");
  const canApply = !!competitorAttr && market.products.some((p) => p.median !== null) && !!onApplyCompetitorPrices;
  return (
    <Card data-testid="market-panel">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          <Globe2 className="h-4 w-4" /> Market research <Badge variant="outline">{market.provider} · {market.region.toUpperCase()}</Badge>
          <Badge variant="secondary">{market.searches_used} searches{market.cache_hits ? ` · ${market.cache_hits} cached` : ""}</Badge>
        </CardTitle>
        <CardDescription>{market.disclaimer}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{domain.product_noun}</TableHead><TableHead className="text-right">Our base</TableHead>
              <TableHead className="text-right">Market min</TableHead><TableHead className="text-right">Median</TableHead>
              <TableHead className="text-right">Max</TableHead><TableHead>Prices found</TableHead><TableHead>Position</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {market.products.map((p) => (
              <TableRow key={p.product_id}>
                <TableCell className="font-medium">{p.name}<div className="text-[11px] text-muted-foreground">“{p.query}”</div></TableCell>
                <TableCell className="tabular text-right">{fmt(p.base_price)}</TableCell>
                <TableCell className="tabular text-right">{fmt(p.min)}</TableCell>
                <TableCell className="tabular text-right font-medium">{fmt(p.median)}</TableCell>
                <TableCell className="tabular text-right">{fmt(p.max)}</TableCell>
                <TableCell><span className="tabular">{p.n}</span> <Badge variant={p.confidence === "high" ? "success" : p.confidence === "none" ? "secondary" : "warning"}>{p.confidence}</Badge></TableCell>
                <TableCell>
                  {p.position ? <Badge variant={POSITION[p.position].variant}>{POSITION[p.position].label} {p.base_vs_median_pct !== null && `(${p.base_vs_median_pct > 0 ? "+" : ""}${p.base_vs_median_pct}%)`}</Badge> : <span className="text-xs text-muted-foreground">no data</span>}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {canApply && (
          <div className="flex flex-wrap items-center gap-3 rounded-md border bg-muted/30 p-3 text-sm">
            <span>Use the observed medians as each {domain.product_noun.toLowerCase()}'s <b>{competitorAttr!.label}</b> (you can edit them in Domains).</span>
            <AsyncButton size="sm" variant="outline" onClick={onApplyCompetitorPrices!}><Wand2 /> Apply medians</AsyncButton>
          </div>
        )}
        {observations && observations.length > 0 && (
          <div>
            <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">How the market shaped the rules</div>
            <ul className="list-disc space-y-1 pl-5 text-sm">{observations.map((o, i) => <li key={i}>{o}</li>)}</ul>
          </div>
        )}
        <details className="text-sm">
          <summary className="cursor-pointer text-xs font-medium uppercase text-muted-foreground">Sources ({market.products.reduce((n, p) => n + p.observations.length, 0)} prices, {market.insights.length} articles)</summary>
          <div className="mt-2 grid gap-3 lg:grid-cols-2">
            <ul className="space-y-1 text-xs">
              {market.products.flatMap((p) => p.observations.map((o, i) => (
                <li key={`${p.product_id}-${i}`} className="flex items-start gap-1">
                  <span className="tabular w-20 shrink-0 font-medium">{fmt(o.price)}</span>
                  <span className="text-muted-foreground">{o.kind} · {o.source || "web"} · {o.link ? <a href={o.link} target="_blank" rel="noreferrer noopener" className="underline">{o.title}</a> : o.title}</span>
                </li>
              )))}
            </ul>
            <ul className="space-y-2 text-xs">
              {market.insights.map((i, k) => (
                <li key={k}>
                  <a href={i.link} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 font-medium underline">{i.title} <ExternalLink className="h-3 w-3" /></a>
                  <div className="text-muted-foreground">{i.snippet}</div>
                </li>
              ))}
            </ul>
          </div>
        </details>
        {market.warnings.length > 0 && <ul className="list-disc pl-5 text-xs text-muted-foreground">{market.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
      </CardContent>
    </Card>
  );
}
