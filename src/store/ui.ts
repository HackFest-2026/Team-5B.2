import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { Role } from "@/types";

/** Demo users per role. The backend's dev auth trusts these names (development only). */
export const USERS: Record<Role, string> = {
  analyst: "priya.analyst",
  manager: "marco.manager",
  admin: "ada.admin",
  customer: "guest-gold",
};

interface UiState {
  role: Exclude<Role, "customer">;
  users: Record<Role, string>;
  domainId: string;
  /** Pricing date sent with simulations and quotes. Explicit so results are reproducible. */
  asOf: string;
  setRole: (r: Exclude<Role, "customer">) => void;
  setUser: (r: Role, user: string) => void;
  setDomainId: (id: string) => void;
  setAsOf: (d: string) => void;
}

export const useUi = create<UiState>()(
  persist(
    (set) => ({
      role: "analyst",
      users: USERS,
      domainId: "",
      asOf: new Date().toISOString().slice(0, 10),
      setRole: (role) => set({ role }),
      setUser: (r, user) => set((s) => ({ users: { ...s.users, [r]: user } })),
      setDomainId: (domainId) => set({ domainId }),
      setAsOf: (asOf) => set({ asOf }),
    }),
    { name: "m360-ui-v2", storage: createJSONStorage(() => localStorage) },
  ),
);

export const ROLE_LABEL: Record<Role, string> = { analyst: "Analyst", manager: "Manager", admin: "Admin", customer: "Customer" };
