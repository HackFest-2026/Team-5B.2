import { useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { CalendarDays, ChevronDown, ExternalLink, Menu, RotateCcw, UserCircle2, X } from "lucide-react";
import { toast } from "sonner";
import { api, invalidate } from "@/api";
import { useCurrentDomain, useDomains } from "@/api/hooks";
import { ROLE_LABEL, useUi } from "@/store/ui";
import { cn } from "@/lib/utils";
import { ALL_NAV, NAV } from "./nav";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/primitives";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger,
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/overlays";
import { ConfirmDialog, ProviderBadges, toastError } from "@/components/common";

const ROLES = ["analyst", "manager", "admin"] as const;

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
      <div className="flex h-14 items-center gap-2 px-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">M</div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">Monetize360</div>
          <div className="text-[11px] text-sidebar-muted">Business portal</div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-3 pb-6">
        {NAV.map((s) => (
          <div key={s.section} className="mt-4">
            <div className="px-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-sidebar-muted">{s.section}</div>
            {s.items.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === "/business"}
                onClick={onNavigate}
                className={({ isActive }) =>
                  cn("flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors",
                    isActive ? "bg-white/10 font-medium text-white" : "text-sidebar-foreground/80 hover:bg-white/5 hover:text-white")
                }
              >
                <item.icon className="h-4 w-4" />
                <span className="flex-1">{item.label}</span>
              </NavLink>
            ))}
          </div>
        ))}
      </div>
      <div className="border-t border-white/10 p-3">
        <Link to="/portal" className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-sidebar-foreground/80 hover:bg-white/5 hover:text-white">
          <ExternalLink className="h-4 w-4" /> Open user portal
        </Link>
      </div>
    </nav>
  );
}

function TopBar({ onMenu }: { onMenu: () => void }) {
  const { data: domains } = useDomains();
  const { id } = useCurrentDomain();
  const { role, users, setRole, setDomainId, asOf, setAsOf } = useUi();
  const [confirmReset, setConfirmReset] = useState(false);
  const location = useLocation();
  const current = [...ALL_NAV].reverse().find((n) => location.pathname.startsWith(n.path));

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b bg-background/95 px-4 backdrop-blur lg:px-6">
      <Button variant="ghost" size="icon" className="lg:hidden" onClick={onMenu} aria-label="Open menu">
        <Menu />
      </Button>
      <div className="hidden text-sm text-muted-foreground xl:block">{current?.label}</div>
      <div className="hidden xl:block"><ProviderBadges compact /></div>
      <div className="ml-auto flex items-center gap-2">
        <Select value={id ?? ""} onValueChange={setDomainId}>
          <SelectTrigger className="h-8 w-[170px]" aria-label="Select domain">
            <SelectValue placeholder="No domains" />
          </SelectTrigger>
          <SelectContent>
            {(domains ?? []).map((d) => (
              <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="hidden items-center gap-1.5 md:flex" title="Pricing date sent with simulations and quotes (explicit for reproducibility)">
          <CalendarDays className="h-4 w-4 text-muted-foreground" />
          <Input type="date" value={asOf} onChange={(e) => e.target.value && setAsOf(e.target.value)} className="h-8 w-[140px]" aria-label="Pricing date" />
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="h-8" data-testid="role-menu">
              <UserCircle2 />
              <span className="hidden sm:inline">{ROLE_LABEL[role]}</span>
              <ChevronDown className="opacity-60" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Signed in as {users[role]} (dev token)</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {ROLES.map((r) => (
              <DropdownMenuItem key={r} onSelect={() => { setRole(r); invalidate(); toast.success(`Switched to ${ROLE_LABEL[r]} (${users[r]})`); }}>
                <span className={cn("h-2 w-2 rounded-full", r === role ? "bg-primary" : "bg-transparent")} />
                {ROLE_LABEL[r]} · {users[r]}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem disabled={role !== "admin"} onSelect={() => setConfirmReset(true)} className="text-destructive">
              <RotateCcw className="h-4 w-4" /> Reset demo data {role !== "admin" && "(admin only)"}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <ConfirmDialog
        open={confirmReset}
        onOpenChange={setConfirmReset}
        title="Reset demo data?"
        description="Every domain, draft, version, decision and audit entry is replaced with the seeded demo data. This cannot be undone."
        confirmLabel="Reset"
        destructive
        onConfirm={async () => {
          try {
            const r = await api.reseed();
            invalidate();
            toast.success(`Demo data reset: ${r.seeded.join(", ")}`);
          } catch (e) {
            toastError(e);
          }
        }}
      />
    </header>
  );
}

export function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <div className="min-h-screen">
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-60 lg:block">
        <Sidebar />
      </aside>
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setMobileOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-60">
            <Sidebar onNavigate={() => setMobileOpen(false)} />
            <Button variant="ghost" size="icon" className="absolute right-2 top-3 text-white" onClick={() => setMobileOpen(false)} aria-label="Close menu">
              <X />
            </Button>
          </div>
        </div>
      )}
      <div className="lg:pl-60">
        <TopBar onMenu={() => setMobileOpen(true)} />
        <main className="mx-auto max-w-[1400px] p-4 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
