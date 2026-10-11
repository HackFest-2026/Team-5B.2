import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { AlertTriangle, FlaskConical, Inbox, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { ApiError, api, useQuery } from "@/api";
import type { ShadowStatus } from "@/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/primitives";
import { Button, type ButtonProps } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/overlays";

export function PageHeader({ title, description, actions, icon: Icon }: { title: string; description?: React.ReactNode; actions?: React.ReactNode; icon?: LucideIcon }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="flex items-start gap-3">
        {Icon && (
          <div className="mt-0.5 rounded-lg bg-accent p-2 text-accent-foreground">
            <Icon className="h-5 w-5" />
          </div>
        )}
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          {description && <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{description}</p>}
        </div>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function EmptyState({ title, description, action, icon: Icon = Inbox }: { title: string; description?: React.ReactNode; action?: React.ReactNode; icon?: LucideIcon }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed px-6 py-12 text-center">
      <Icon className="mb-3 h-8 w-8 text-muted-foreground" />
      <div className="font-medium">{title}</div>
      {description && <div className="mt-1 max-w-md text-sm text-muted-foreground">{description}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function StatCard({ label, value, hint, icon: Icon, tone }: { label: string; value: React.ReactNode; hint?: React.ReactNode; icon?: LucideIcon; tone?: "good" | "bad" }) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>{label}</span>
          {Icon && <Icon className="h-4 w-4" />}
        </div>
        <div className="tabular mt-2 text-2xl font-semibold">{value}</div>
        {hint && <div className={cn("mt-1 text-xs", tone === "good" ? "text-success" : tone === "bad" ? "text-destructive" : "text-muted-foreground")}>{hint}</div>}
      </CardContent>
    </Card>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const e = error as ApiError;
  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm">
      <div className="flex items-center gap-2 font-medium text-destructive">
        <AlertTriangle className="h-4 w-4" /> {e?.message ?? "Something went wrong"}
      </div>
      {e?.lines?.length ? (
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-destructive/90">
          {e.lines.slice(0, 8).map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      ) : null}
      {onRetry && (
        <Button size="sm" variant="outline" className="mt-3" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

/** Show an API error as a toast with its first detail lines. */
export function toastError(e: unknown, title?: string) {
  const err = e as ApiError;
  const lines = err?.lines ?? [];
  toast.error(title ?? err?.message ?? "Request failed", { description: lines.slice(0, 4).join(" · ") || (title ? err?.message : undefined) });
}

/** A button that shows a spinner while its async action runs. */
export function AsyncButton({ onClick, children, ...props }: Omit<ButtonProps, "onClick"> & { onClick: () => Promise<unknown> }) {
  const [busy, setBusy] = React.useState(false);
  return (
    <Button
      {...props}
      disabled={busy || props.disabled}
      onClick={async () => {
        setBusy(true);
        try {
          await onClick();
        } finally {
          setBusy(false);
        }
      }}
    >
      {busy && <Loader2 className="animate-spin" />}
      {children}
    </Button>
  );
}

export function SimulatedBadge({ label = "Simulated", title }: { label?: string; title?: string }) {
  return (
    <Badge variant="warning" title={title ?? "Synthetic / simulated output: not evidence about real customers."}>
      <FlaskConical className="h-3 w-3" /> {label}
    </Badge>
  );
}

const SHADOW_VARIANT: Record<ShadowStatus, React.ComponentProps<typeof Badge>["variant"]> = {
  PASS: "success", REVIEW_REQUIRED: "warning", FAIL: "destructive", ERROR: "destructive", SKIPPED: "secondary", PENDING: "info",
};
export const ShadowBadge = ({ status }: { status: ShadowStatus | null | undefined }) =>
  status ? <Badge variant={SHADOW_VARIANT[status]}>{status.replace("_", " ").toLowerCase()}</Badge> : <span className="text-xs text-muted-foreground">—</span>;

export const StatusPill = ({ status }: { status: string }) => {
  const v: React.ComponentProps<typeof Badge>["variant"] =
    status === "ACTIVE" || status === "VALIDATED" || status === "OK" || status === "SUCCESS" ? "success"
      : status === "APPROVED" ? "info"
      : status === "DRAFT" || status === "REJECTED" ? "warning"
      : status === "NO_VALID_PRICE" || status === "FAILED" ? "destructive" : "secondary";
  return <Badge variant={v}>{status.replace(/_/g, " ")}</Badge>;
};

/** Which LLM / optimizer / validation is live, straight from /health. */
export function ProviderBadges({ compact }: { compact?: boolean }) {
  const { data } = useQuery("health", api.health);
  if (!data) return null;
  const opt = data.optimizer;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant={data.llm.mock ? "warning" : "info"}>LLM: {data.llm.mock ? "mock generator" : `${data.llm.provider}${compact ? "" : ` (${data.llm.model})`}`}</Badge>
        </TooltipTrigger>
        <TooltipContent>{data.llm.mock ? "No LLM key configured: strategies come from a deterministic, clearly labelled mock." : "Strategies are drafted by a real LLM, then validated."}</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant={opt.kind === "ml" ? "info" : "warning"}>Optimizer: {opt.kind === "ml" ? opt.name.replace("ml_updated-", "ML ").toUpperCase() : "simulated"}</Badge>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">{opt.version}{opt.training_data === "synthetic" ? " · trained on synthetic data" : ""}{opt.fallback_reason ? ` · fallback: ${opt.fallback_reason}` : ""}</TooltipContent>
      </Tooltip>
      {opt.training_data === "synthetic" && <SimulatedBadge label="synthetic training data" />}
      <Badge variant={data.market?.enabled ? "info" : "secondary"} title="Web search used to ground generated strategies">
        Market: {data.market?.enabled ? data.market.provider : "off"}
      </Badge>
    </div>
  );
}

export function ConfirmDialog({
  open, onOpenChange, title, description, confirmLabel = "Confirm", destructive, onConfirm,
}: {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; description?: React.ReactNode; confirmLabel?: string;
  destructive?: boolean; onConfirm: () => void | Promise<void>;
}) {
  const [busy, setBusy] = React.useState(false);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>Cancel</Button>
          <Button
            variant={destructive ? "destructive" : "default"}
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await onConfirm();
                onOpenChange(false);
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy && <Loader2 className="animate-spin" />} {confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function Field({ label, hint, children, className }: { label: string; hint?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <label className={cn("block space-y-1.5", className)}>
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      {children}
      {hint && <span className="block text-[11px] text-muted-foreground">{hint}</span>}
    </label>
  );
}
