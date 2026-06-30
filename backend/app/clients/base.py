"""Shared HTTP plumbing for all external API clients.

Synchronous httpx is used deliberately: it keeps the client code easy to read
and debug, and FastAPI runs sync endpoint handlers in a threadpool so we don't
block the event loop.
"""
from __future__ import annotations

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
USER_AGENT = "ai-drug-repurposing/0.1 (research assistant)"


def make_client(base_url: str = "", headers: dict | None = None) -> httpx.Client:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    return httpx.Client(base_url=base_url, headers=merged, timeout=DEFAULT_TIMEOUT, follow_redirects=True)


class ApiError(RuntimeError):
    """Raised when an upstream API returns an error or unexpected payload."""
