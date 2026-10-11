"""Runtime settings from environment variables / backend/.env (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'monetize360.db').as_posix()}"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    llm_provider: Literal["groq", "gemini", "grok", "mock"] = "mock"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    xai_api_key: str = ""
    grok_model: str = "grok-4"
    llm_timeout_s: float = 45.0
    persona_text_provider: Literal["groq", "gemini", "grok", "none"] = "none"

    # Market research (web search) used to ground generated strategies.
    search_provider: Literal["auto", "serpapi", "serper", "none"] = "auto"
    serpapi_api_key: str = ""
    serper_api_key: str = ""
    market_max_products: int = 3
    market_cache_ttl_h: float = 24.0
    market_cache_file: str = str(BACKEND_DIR / "data" / "market_cache.json")

    optimizer_provider: Literal["auto", "ml", "simulated", "http"] = "auto"
    optimizer_model: Literal["gbt", "lr"] = "gbt"
    ml_artifacts_dir: str = str(REPO_DIR / "ml_updated" / "artifacts")
    ml_artifact_sha256: str = ""
    optimizer_url: str = ""

    validation_mode: Literal["sync", "background", "off"] = "background"
    validation_pop_sim: int = 1000
    validation_pop_quote: int = 250
    validation_seed: int = 42

    seed_on_startup: bool = True

    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
