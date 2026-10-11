"""Market research (web search) that grounds LLM strategy drafts in observed competitor prices."""

from __future__ import annotations

from typing import Optional

from app.config import Settings

from .research import MarketResearcher, compact_for_prompt
from .search import DiskCache, SearchError, WebSearch, build_search


def build_researcher(settings: Settings) -> Optional[MarketResearcher]:
    cache = DiskCache(settings.market_cache_file, settings.market_cache_ttl_h)
    search = build_search(settings.search_provider, settings.serpapi_api_key, settings.serper_api_key, cache)
    return MarketResearcher(search, settings.market_max_products) if search else None


__all__ = ["MarketResearcher", "SearchError", "WebSearch", "build_researcher", "compact_for_prompt"]
