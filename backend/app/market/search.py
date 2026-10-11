"""Web search clients (SerpApi or Serper) with a small on-disk cache to save search credits."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Protocol

import httpx

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    title: str
    snippet: str = ""
    source: str = ""
    link: str = ""
    price: Optional[float] = None       # structured price when the engine returns one (shopping)
    price_text: str = ""


class SearchError(RuntimeError):
    pass


class WebSearch(Protocol):
    name: Literal["serpapi", "serper"]

    def shopping(self, query: str, region: str) -> list[SearchHit]: ...

    def web(self, query: str, region: str, num: int = 8) -> list[SearchHit]: ...


class DiskCache:
    """JSON file cache keyed by (provider, kind, region, query). Thread-safe, TTL in hours."""

    def __init__(self, path: str, ttl_h: float):
        self.path = Path(path)
        self.ttl = ttl_h * 3600
        self.lock = threading.Lock()
        self.hits = 0

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> Optional[list[SearchHit]]:
        with self.lock:
            entry = self._load().get(key)
        if entry and time.time() - entry["at"] < self.ttl:
            self.hits += 1
            return [SearchHit(**h) for h in entry["hits"]]
        return None

    def put(self, key: str, hits: list[SearchHit]) -> None:
        with self.lock:
            data = self._load()
            data[key] = {"at": time.time(), "hits": [asdict(h) for h in hits]}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data), encoding="utf-8")


class _Base:
    name: Literal["serpapi", "serper"]

    def __init__(self, api_key: str, cache: Optional[DiskCache] = None, client: Optional[httpx.Client] = None, timeout_s: float = 30):
        if not api_key:
            raise SearchError(f"{self.name} API key is not set")
        self.key = api_key
        self.cache = cache
        self.client = client or httpx.Client(timeout=timeout_s)
        self.calls = 0

    def _cached(self, kind: str, query: str, region: str, fetch) -> list[SearchHit]:
        ck = f"{self.name}|{kind}|{region}|{query.lower().strip()}"
        if self.cache and (hit := self.cache.get(ck)) is not None:
            return hit
        for attempt in (1, 2):  # one retry on timeouts / transient network errors
            try:
                hits = fetch()
                break
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt == 2:
                    raise SearchError(f"{self.name} request failed: {e}") from None
            except httpx.HTTPError as e:
                raise SearchError(f"{self.name} request failed: {e}") from None
        self.calls += 1
        if self.cache:
            self.cache.put(ck, hits)
        return hits


class SerpApiClient(_Base):
    """serpapi.com: engines google_shopping (structured prices) and google (organic results)."""

    name = "serpapi"
    URL = "https://serpapi.com/search.json"

    def _get(self, params: dict) -> dict:
        r = self.client.get(self.URL, params={**params, "api_key": self.key})
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200 or data.get("error"):
            raise SearchError(f"SerpApi error {r.status_code}: {data.get('error', r.text[:120])}")
        return data

    def shopping(self, query: str, region: str) -> list[SearchHit]:
        def fetch():
            data = self._get({"engine": "google_shopping", "q": query, "gl": region, "hl": "en"})
            return [SearchHit(title=x.get("title", ""), source=x.get("source", ""), link=x.get("product_link") or x.get("link", ""),
                              price=x.get("extracted_price"), price_text=str(x.get("price", "")))
                    for x in data.get("shopping_results", [])[:20]]
        return self._cached("shopping", query, region, fetch)

    def web(self, query: str, region: str, num: int = 8) -> list[SearchHit]:
        def fetch():
            data = self._get({"engine": "google", "q": query, "gl": region, "hl": "en", "num": num})
            return [SearchHit(title=x.get("title", ""), snippet=x.get("snippet", ""), source=x.get("source", ""), link=x.get("link", ""))
                    for x in data.get("organic_results", [])[:num]]
        return self._cached("web", query, region, fetch)


class SerperClient(_Base):
    """google.serper.dev: /shopping and /search."""

    name = "serper"
    URL = "https://google.serper.dev"

    def _post(self, path: str, body: dict) -> dict:
        r = self.client.post(f"{self.URL}/{path}", headers={"X-API-KEY": self.key}, json=body)
        if r.status_code != 200:
            raise SearchError(f"Serper error {r.status_code}: {r.text[:120]}")
        return r.json()

    def shopping(self, query: str, region: str) -> list[SearchHit]:
        def fetch():
            data = self._post("shopping", {"q": query, "gl": region})
            return [SearchHit(title=x.get("title", ""), source=x.get("source", ""), link=x.get("link", ""), price_text=str(x.get("price", "")))
                    for x in data.get("shopping", [])[:20]]
        return self._cached("shopping", query, region, fetch)

    def web(self, query: str, region: str, num: int = 8) -> list[SearchHit]:
        def fetch():
            data = self._post("search", {"q": query, "gl": region, "num": num})
            return [SearchHit(title=x.get("title", ""), snippet=x.get("snippet", ""), source=x.get("source", ""), link=x.get("link", ""))
                    for x in data.get("organic", [])[:num]]
        return self._cached("web", query, region, fetch)


def build_search(provider: str, serpapi_key: str, serper_key: str, cache: Optional[DiskCache]) -> Optional[WebSearch]:
    """auto: whichever key is set. A 64-hex key in SERPER_API_KEY is treated as a SerpApi key (common mix-up)."""
    if provider == "none":
        return None
    if provider == "auto":
        if serpapi_key:
            provider = "serpapi"
        elif serper_key:
            provider, serpapi_key = ("serpapi", serper_key) if len(serper_key) == 64 else ("serper", "")
        else:
            return None
    try:
        return SerpApiClient(serpapi_key, cache) if provider == "serpapi" else SerperClient(serper_key, cache)
    except SearchError as e:
        log.warning("market search disabled: %s", e)
        return None
