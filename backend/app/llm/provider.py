"""Provider-agnostic LLM layer.

The app only depends on the LLMProvider interface, so swapping Gemini / Ollama /
a placeholder is a one-line change. Each provider exposes `.live` — True only
when a real model will answer. When no live provider is configured we fall back
to MockProvider, which returns clearly-labeled placeholder text (and never
fabricates medical facts) so every UI flow is demoable without an API key.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import settings

PLACEHOLDER_BANNER = "⚠️ Placeholder AI (no live key). Add a valid Gemini key to enable real analysis."


@runtime_checkable
class LLMProvider(Protocol):
    live: bool
    kind: str

    def complete(self, system: str, prompt: str, temperature: float = 0.3) -> str: ...


class NullProvider:
    live = False
    kind = "none"

    def complete(self, system: str, prompt: str, temperature: float = 0.3) -> str:
        return ""


class MockProvider:
    """Placeholder used until a real key is set. Echoes the structured context
    it is given rather than inventing facts, prefixed with a clear banner."""

    live = False
    kind = "placeholder"

    def complete(self, system: str, prompt: str, temperature: float = 0.3) -> str:
        return (
            f"{PLACEHOLDER_BANNER}\n\n"
            "Once a live model is connected it will read the evidence below and "
            "write a cited, science-backed answer here. For now, see the raw "
            "structured data and the linked studies."
        )


class GeminiProvider:
    """Google Gemini via REST (no SDK dependency). Key from aistudio.google.com/apikey."""

    live = True
    kind = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        from app.clients.base import make_client

        self._http = make_client(base_url="https://generativelanguage.googleapis.com")
        self._api_key = api_key
        self._model = model

    def complete(self, system: str, prompt: str, temperature: float = 0.3) -> str:
        r = self._http.post(
            f"/v1beta/models/{self._model}:generateContent",
            headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
        )
        r.raise_for_status()
        candidates = r.json().get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


class GroqProvider:
    """Groq (OpenAI-compatible chat completions, very fast). Key starts 'gsk_'.
    Free key from https://console.groq.com/keys."""

    live = True
    kind = "groq"

    def __init__(self, api_key: str, model: str) -> None:
        from app.clients.base import make_client

        self._http = make_client(base_url="https://api.groq.com")
        self._api_key = api_key
        self._model = model

    def complete(self, system: str, prompt: str, temperature: float = 0.3) -> str:
        r = self._http.post(
            "/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={
                "model": self._model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        r.raise_for_status()
        choices = r.json().get("choices", [])
        return choices[0]["message"]["content"] if choices else ""


class OpenRouterProvider:
    """OpenRouter (OpenAI-compatible gateway to many models). Key starts 'sk-or-'.
    Pick any model via LLM_MODEL, e.g. 'anthropic/claude-3.5-sonnet', 'google/gemini-2.0-flash-001',
    'deepseek/deepseek-chat'. Buy credits at https://openrouter.ai/credits."""

    live = True
    kind = "openrouter"

    def __init__(self, api_key: str, model: str) -> None:
        from app.clients.base import make_client

        self._http = make_client(base_url="https://openrouter.ai")
        self._api_key = api_key
        self._model = model

    def complete(self, system: str, prompt: str, temperature: float = 0.3) -> str:
        r = self._http.post(
            "/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # OpenRouter uses these for attribution/rankings (optional but recommended)
                "HTTP-Referer": "https://repurposepharma.com",
                "X-Title": "RePurpose Pharma",
            },
            json={
                "model": self._model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        r.raise_for_status()
        choices = r.json().get("choices", [])
        return choices[0]["message"]["content"] if choices else ""


class OllamaProvider:
    """Local models via Ollama (free, no key). Active when LLM_PROVIDER=ollama."""

    live = True
    kind = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        from app.clients.base import make_client

        self._http = make_client(base_url=base_url)
        self._model = model

    def complete(self, system: str, prompt: str, temperature: float = 0.3) -> str:
        r = self._http.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")


def _looks_like_real_gemini_key(key: str | None) -> bool:
    # Real AI Studio keys start with "AIza" and are ~39 chars.
    return bool(key) and key.startswith("AIza") and len(key) >= 30


def get_provider() -> LLMProvider:
    provider = (settings.llm_provider or "none").lower()
    if provider == "openrouter" and (settings.llm_api_key or "").startswith("sk-or-"):
        return OpenRouterProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model or "google/gemini-2.0-flash-001",
        )
    if provider == "groq" and (settings.llm_api_key or "").startswith("gsk_"):
        return GroqProvider(api_key=settings.llm_api_key, model=settings.llm_model or "llama-3.3-70b-versatile")
    if provider == "gemini" and _looks_like_real_gemini_key(settings.llm_api_key):
        return GeminiProvider(api_key=settings.llm_api_key, model=settings.llm_model or "gemini-2.0-flash")
    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.llm_base_url or "http://localhost:11434",
            model=settings.llm_model or "llama3.1",
        )
    if provider == "none":
        return NullProvider()
    # provider requested but not usable yet (e.g. invalid/placeholder key)
    return MockProvider()
