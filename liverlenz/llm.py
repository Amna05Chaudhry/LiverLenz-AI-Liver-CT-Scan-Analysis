"""Tiny OpenAI-compatible LLM client.

Configure with ``.env`` (see ``.env.example``)::

    LLM_BASE_URL=https://api.groq.com/openai/v1   # Groq (default), OpenAI, or http://localhost:11434/v1 for Ollama
    LLM_API_KEY=...
    LLM_MODEL=llama-3.3-70b-versatile

With no key configured (or if the call fails) ``chat`` returns ``None`` and the
caller falls back to a templated, rule-based answer. ``status()`` tells the UI
whether to show the "LLM offline" badge.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _config() -> tuple[str, str, str]:
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    # Local Ollama does not need a key; the client still wants a non-empty string.
    if not api_key and ("localhost" in base_url or "127.0.0.1" in base_url):
        api_key = "ollama"
    return base_url, api_key, model


def is_configured() -> bool:
    return bool(_config()[1])


def status() -> dict:
    base_url, _, model = _config()
    return {"online": is_configured(), "model": model if is_configured() else None, "base_url": base_url}


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        base_url, api_key, _ = _config()
        _client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=1)
    return _client


def chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 700) -> str | None:
    """Send chat messages; return the reply text, or ``None`` if unavailable."""
    if not is_configured():
        return None
    try:
        resp = _get_client().chat.completions.create(
            model=_config()[2], messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:  # network, auth, rate limit: degrade gracefully
        log.warning("LLM call failed, using template fallback: %s", exc)
        return None
