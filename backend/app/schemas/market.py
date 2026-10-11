"""Market research context gathered from web search to ground LLM strategy generation.

Everything here is unverified public web data: noisy, possibly stale, and never used as a price by
itself. It only informs the draft rules a business user then reviews.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .common import StrictModel


class PriceObservation(StrictModel):
    price: float
    title: str
    source: str = ""
    link: str = ""
    kind: Literal["shopping", "web"]


class ProductMarket(StrictModel):
    product_id: str
    name: str
    base_price: float
    query: str
    observations: list[PriceObservation] = Field(default_factory=list)
    n: int = 0
    min: Optional[float] = None
    median: Optional[float] = None
    max: Optional[float] = None
    confidence: Literal["none", "low", "medium", "high"] = "none"
    position: Optional[Literal["below_market", "in_line", "above_market"]] = None
    base_vs_median_pct: Optional[float] = None


class MarketInsight(StrictModel):
    title: str
    snippet: str
    source: str = ""
    link: str = ""


class MarketContext(StrictModel):
    provider: Literal["serpapi", "serper", "none"]
    region: str
    currency: str
    queried_at: str
    products: list[ProductMarket] = Field(default_factory=list)
    insights: list[MarketInsight] = Field(default_factory=list)
    searches_used: int = 0
    cache_hits: int = 0
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = "Unverified public web search results. Signals for drafting rules only, never used as prices."
