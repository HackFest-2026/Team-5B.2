"""LLM provider boundary. Providers only return JSON; they never price anything and never run code."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel

ProviderName = Literal["groq", "gemini", "grok", "mock"]


class LlmError(RuntimeError):
    """The provider failed (network, auth, quota, malformed JSON)."""


class JsonLlm(Protocol):
    name: ProviderName
    model: str

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> Any:
        """Return parsed JSON intended to match ``schema``. The caller validates it."""
        ...
