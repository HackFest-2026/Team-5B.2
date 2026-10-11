"""Gemini and Grok clients behind the JsonLlm interface.

Both request JSON constrained by the schema and both outputs are re-validated by Pydantic and the
strategy validator afterwards, so provider quirks can't slip unvalidated data into a strategy.
These calls need API keys; without keys the mock generator is used and labelled as such.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .base import LlmError


def _parse(text: str | None) -> Any:
    if not text:
        raise LlmError("empty response from provider")
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise LlmError(f"provider returned malformed JSON: {e}") from None


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: float = 45.0):
        if not api_key:
            raise LlmError("GEMINI_API_KEY is not set")
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key, http_options={"timeout": int(timeout_s * 1000)})

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> Any:
        from google.genai import types

        base = dict(system_instruction=system, response_mime_type="application/json", temperature=0.2)
        try:
            try:
                cfg = types.GenerateContentConfig(**base, response_json_schema=schema.model_json_schema())
            except Exception:  # older SDKs: no response_json_schema → schema goes in the prompt only
                cfg = types.GenerateContentConfig(**base)
            resp = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
        except Exception as e:  # noqa: BLE001 - surface provider failures uniformly
            raise LlmError(f"Gemini request failed: {e}") from e
        return _parse(getattr(resp, "text", None))


class GrokClient:
    name = "grok"

    def __init__(self, api_key: str, model: str, timeout_s: float = 45.0):
        if not api_key:
            raise LlmError("XAI_API_KEY is not set")
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=timeout_s)

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        formats = [
            {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
            {"type": "json_object"},
        ]
        last: Exception | None = None
        for fmt in formats:
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.2,
                                                           response_format=fmt)
                return _parse(resp.choices[0].message.content)
            except LlmError:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
        raise LlmError(f"Grok request failed: {last}")


class GroqClient:
    """Groq (OpenAI-compatible API). Default model openai/gpt-oss-120b, which follows JSON schemas well."""

    name = "groq"

    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0):
        if not api_key:
            raise LlmError("GROQ_API_KEY is not set")
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=timeout_s, max_retries=2)

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        extra = {"reasoning_effort": "medium"} if "gpt-oss" in self.model else {}
        formats = [
            {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}},
            {"type": "json_object"},
        ]
        last: Exception | None = None
        for fmt in formats:
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.2,
                                                           response_format=fmt, max_completion_tokens=6000, extra_body=extra)
                return _parse(resp.choices[0].message.content)
            except LlmError as e:
                last = e
            except Exception as e:  # noqa: BLE001 - try the next response format, then give up
                last = e
        raise LlmError(f"Groq request failed: {last}")
