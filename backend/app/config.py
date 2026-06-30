"""Central configuration, loaded from environment / .env file.

Nothing here is required to run the data clients; the NCBI fields just unlock
a higher rate limit, and the LLM fields are unused until a provider is chosen.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # NCBI E-utilities
    ncbi_api_key: str | None = None
    ncbi_tool: str = "ai-drug-repurposing"
    ncbi_email: str | None = None

    # LLM (provider-agnostic; "none" until we pick one)
    llm_provider: str = "none"
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None


settings = Settings()
