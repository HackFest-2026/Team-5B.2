import type { LucideIcon } from "lucide-react";
import { Calculator, FileSearch, GitBranch, Layers, LayoutDashboard, ScrollText, Sparkles } from "lucide-react";

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
}

export const NAV: { section: string; items: NavItem[] }[] = [
  { section: "Overview", items: [{ path: "/business", label: "Dashboard", icon: LayoutDashboard }] },
  {
    section: "Configure",
    items: [
      { path: "/business/domains", label: "Domains", icon: Layers },
      { path: "/business/strategy", label: "Strategy Studio", icon: Sparkles },
    ],
  },
  { section: "Analyse", items: [{ path: "/business/simulator", label: "Simulator", icon: Calculator }] },
  {
    section: "Govern",
    items: [
      { path: "/business/versions", label: "Versions & Approvals", icon: GitBranch },
      { path: "/business/decisions", label: "Decisions", icon: FileSearch },
      { path: "/business/audit", label: "Audit Log", icon: ScrollText },
    ],
  },
];

export const ALL_NAV = NAV.flatMap((s) => s.items);
