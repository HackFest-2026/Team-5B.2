// A tiny query cache: components read with useQuery(key, fn) and writes call invalidate(prefix).
import { useCallback, useEffect, useRef, useState } from "react";

type Listener = () => void;
const listeners = new Map<string, Set<Listener>>();
const cache = new Map<string, unknown>();

export function invalidate(prefix = "") {
  for (const k of [...cache.keys()]) if (k.startsWith(prefix)) cache.delete(k);
  for (const [k, set] of listeners) if (k.startsWith(prefix)) set.forEach((l) => l());
}

export interface QueryState<T> {
  data: T | undefined;
  error: Error | null;
  loading: boolean;
  reload: () => void;
}

export function useQuery<T>(key: string | null, fn: () => Promise<T>, opts: { pollMs?: number } = {}): QueryState<T> {
  const [data, setData] = useState<T | undefined>(() => (key ? (cache.get(key) as T | undefined) : undefined));
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(!!key && !cache.has(key));
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const seq = useRef(0);

  const load = useCallback(() => {
    if (!key) return;
    const my = ++seq.current;
    setLoading(true);
    fnRef
      .current()
      .then((d) => {
        if (my !== seq.current) return;
        cache.set(key, d);
        setData(d);
        setError(null);
      })
      .catch((e: Error) => my === seq.current && setError(e))
      .finally(() => my === seq.current && setLoading(false));
  }, [key]);

  useEffect(() => {
    if (!key) {
      setData(undefined);
      return;
    }
    setData(cache.get(key) as T | undefined);
    load();
    const set = listeners.get(key) ?? new Set();
    set.add(load);
    listeners.set(key, set);
    return () => {
      set.delete(load);
    };
  }, [key, load]);

  useEffect(() => {
    if (!opts.pollMs || !key) return;
    const t = setInterval(load, opts.pollMs);
    return () => clearInterval(t);
  }, [opts.pollMs, key, load]);

  return { data, error, loading, reload: load };
}
