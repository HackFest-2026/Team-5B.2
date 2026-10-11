"""Strategy generation: LLM providers (Gemini, Grok) and a clearly labelled mock fallback."""

from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings

from .base import JsonLlm, LlmError

log = logging.getLogger(__name__)


def build_llm(settings: Settings, provider: Optional[str] = None) -> Optional[JsonLlm]:
    """Return the configured JSON LLM client, or None (meaning: use the mock generator)."""
    name = provider or settings.llm_provider
    try:
        if name == "groq":
            from .providers import GroqClient

            return GroqClient(settings.groq_api_key, settings.groq_model, max(settings.llm_timeout_s, 60))
        if name == "gemini":
            from .providers import GeminiClient

            return GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s)
        if name == "grok":
            from .providers import GrokClient

            return GrokClient(settings.xai_api_key, settings.grok_model, settings.llm_timeout_s)
    except (LlmError, ImportError) as e:
        log.warning("LLM provider %s unavailable (%s); using the mock generator", name, e)
    return None


__all__ = ["JsonLlm", "LlmError", "build_llm"]
