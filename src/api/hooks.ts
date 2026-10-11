import { useEffect } from "react";
import { api } from "./index";
import { useQuery } from "./query";
import { useUi } from "@/store/ui";

export const useDomains = () => useQuery("domains", api.domains);

/** The domain picked in the top bar (falls back to the first one). Loads its full config. */
export function useCurrentDomain() {
  const { data: domains, loading: listLoading, error: listError } = useDomains();
  const domainId = useUi((s) => s.domainId);
  const setDomainId = useUi((s) => s.setDomainId);
  const id = domains?.find((d) => d.id === domainId)?.id ?? domains?.[0]?.id ?? null;
  useEffect(() => {
    if (id && id !== domainId) setDomainId(id);
  }, [id, domainId, setDomainId]);
  const detail = useQuery(id ? `domain:${id}` : null, () => api.domain(id!));
  return {
    id,
    summary: domains?.find((d) => d.id === id),
    domain: detail.data,
    loading: listLoading || detail.loading,
    error: listError ?? detail.error,
    reload: detail.reload,
  };
}
