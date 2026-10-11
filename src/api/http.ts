// The only way the UI reaches data: HTTP calls to the FastAPI backend (/api/v1, proxied by Vite).
import type { Role } from "@/types";
import { useUi } from "@/store/ui";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details?: unknown) {
    super(message);
  }
  /** Flatten details into readable lines for toasts and inline errors. */
  get lines(): string[] {
    const d = this.details;
    if (!d) return [];
    if (Array.isArray(d)) {
      return d.map((x) => (typeof x === "string" ? x : x && typeof x === "object" ? String((x as { message?: string }).message ?? JSON.stringify(x)) : String(x)));
    }
    if (typeof d === "object" && Array.isArray((d as { errors?: unknown[] }).errors)) return (d as { errors: string[] }).errors;
    return [];
  }
}

export function authHeader(role?: Role): string {
  const s = useUi.getState();
  const r = role ?? s.role;
  return `Bearer dev:${r}:${s.users[r]}`;
}

export async function request<T>(method: string, path: string, body?: unknown, role?: Role): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/v1${path}`, {
      method,
      headers: { "Content-Type": "application/json", Authorization: authHeader(role) },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "NETWORK", "Cannot reach the backend. Is it running on port 8010?");
  }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const e = data?.error ?? {};
    throw new ApiError(res.status, e.code ?? "HTTP_" + res.status, e.message ?? res.statusText, e.details);
  }
  return data as T;
}
