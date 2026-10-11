"""Market research for any domain: competitor price signals per product plus a few market snippets.

Queries are built only from the domain's own configuration (names, product names, currency, unit),
so the same code works for every industry. Search use is bounded per run and cached on disk.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.db import utcnow
from app.schemas.domain import DomainConfig, Product
from app.schemas.market import MarketContext, MarketInsight, PriceObservation, ProductMarket

from .prices import extract_prices, in_band, percent_values, region_for, summarize
from .search import SearchError, WebSearch

log = logging.getLogger(__name__)


def _is_percent_unit(domain: DomainConfig) -> bool:
    return domain.output_unit.suffix.strip().startswith("%") or "%" in domain.output_unit.label


def _product_query(domain: DomainConfig, p: Product) -> str:
    unit = domain.output_unit.label.split(" per ")[-1] if " per " in domain.output_unit.label else ""
    return " ".join(x for x in (p.name, domain.name if domain.name.lower() not in p.name.lower() else "", unit, "price") if x)


class MarketResearcher:
    def __init__(self, search: WebSearch, max_products: int = 3):
        self.search = search
        self.max_products = max_products

    def research(self, domain: DomainConfig, description: str = "") -> MarketContext:
        region = region_for(domain.currency)
        ctx = MarketContext(provider=self.search.name, region=region, currency=domain.currency, queried_at=utcnow())
        calls_before = getattr(self.search, "calls", 0)
        cache = getattr(self.search, "cache", None)
        hits_before = cache.hits if cache else 0
        products = domain.products[: self.max_products]
        if len(domain.products) > self.max_products:
            ctx.warnings.append(f"Researched the first {self.max_products} of {len(domain.products)} products to save search credits.")
        percent = _is_percent_unit(domain)

        with ThreadPoolExecutor(max_workers=4) as pool:
            markets = list(pool.map(lambda p: self._product(domain, p, region, percent, ctx.warnings), products))
            insight_q = f"{domain.name} {domain.product_noun.lower()} pricing market {domain.currency}"
            try:
                web = self.search.web(insight_q, region, num=6)
                ctx.insights = [MarketInsight(title=h.title[:200], snippet=h.snippet[:400], source=h.source, link=h.link)
                                for h in web if h.snippet][:6]
            except SearchError as e:
                ctx.warnings.append(str(e))
        ctx.products = markets
        ctx.searches_used = getattr(self.search, "calls", 0) - calls_before
        ctx.cache_hits = (cache.hits - hits_before) if cache else 0
        if not any(m.n for m in markets):
            ctx.warnings.append("No usable competitor prices were found; the strategy relies on the business's own inputs.")
        return ctx

    def _product(self, domain: DomainConfig, p: Product, region: str, percent: bool, warnings: list[str]) -> ProductMarket:
        query = _product_query(domain, p)
        obs: list[PriceObservation] = []
        try:
            if not percent:  # rates and percentages aren't sold in shopping results
                for h in self.search.shopping(query, region):
                    price = h.price if h.price else next(iter(extract_prices(h.price_text, domain.currency)), None)
                    if price and in_band(price, p.base_price):
                        obs.append(PriceObservation(price=price, title=h.title[:160], source=h.source, link=h.link, kind="shopping"))
            if len(obs) < 3:  # services or sparse shopping data: read prices out of web snippets
                for h in self.search.web(query, region, num=8):
                    text = f"{h.title} {h.snippet}"
                    values = percent_values(text) if percent else extract_prices(text, domain.currency)
                    for v in values:
                        if in_band(v, p.base_price):
                            obs.append(PriceObservation(price=v, title=h.title[:160], source=h.source, link=h.link, kind="web"))
        except SearchError as e:
            warnings.append(f"{p.name}: {e}")
        stats = summarize([o.price for o in obs])
        m = ProductMarket(product_id=p.id, name=p.name, base_price=p.base_price, query=query, observations=obs[:12], **stats)
        if m.median:
            gap = (p.base_price - m.median) / m.median * 100
            m.base_vs_median_pct = round(gap, 1)
            m.position = "above_market" if gap > 5 else "below_market" if gap < -5 else "in_line"
        return m


def compact_for_prompt(ctx: Optional[MarketContext]) -> Optional[dict]:
    """The slice of market data the LLM sees (no links, bounded size)."""
    if ctx is None:
        return None
    return {
        "region": ctx.region, "currency": ctx.currency, "disclaimer": ctx.disclaimer,
        "products": [{"product_id": m.product_id, "name": m.name, "our_base_price": m.base_price, "observed_prices": m.n,
                      "market_min": m.min, "market_median": m.median, "market_max": m.max, "confidence": m.confidence,
                      "our_position": m.position, "base_vs_median_pct": m.base_vs_median_pct,
                      "examples": [f"{o.price:g} — {o.source or o.title[:40]}" for o in m.observations[:4]]}
                     for m in ctx.products],
        "market_snippets": [f"{i.title}: {i.snippet}" for i in ctx.insights[:5]],
        "warnings": ctx.warnings,
    }
