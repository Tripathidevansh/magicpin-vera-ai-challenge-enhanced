#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — LLM Provider Abstraction (Phase 3)
==============================================================
Provides resilient, low-latency LLM completions with automatic fallback:
- Primary : Google Gemini 2.5 Flash
- Fallback: Groq Llama 3.3 70B

Configured via environment variables:
- GEMINI_API_KEY
- GROQ_API_KEY

Enforces:
- Strict per-call timeouts
- temperature = 0.0 for determinism
- Structured JSON output format
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import httpx

TIMEOUT_LLM_CALL = 4.5

def get_llm_timeout() -> float:
    return float(os.environ.get("LLM_CALL_TIMEOUT", str(TIMEOUT_LLM_CALL)))


class LLMProvider(ABC):
    """Abstract base provider for LLM completion calls."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name and model."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the provider has necessary API keys configured."""
        pass

    @abstractmethod
    async def complete(
        self, prompt: str, system_prompt: str, timeout: Optional[float] = None
    ) -> str:
        """Execute chat completion and return raw response text."""
        pass


class GeminiProvider(LLMProvider):
    """
    Google Gemini 2.5 Flash provider using the Generative Language REST API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model

    def name(self) -> str:
        return f"Gemini ({self.model})"

    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    async def complete(
        self, prompt: str, system_prompt: str, timeout: Optional[float] = None
    ) -> str:
        t_limit = timeout if timeout is not None else get_llm_timeout()
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Gemini candidate has no content parts")

        return parts[0].get("text", "")


class GroqProvider(LLMProvider):
    """
    Groq provider running Llama 3.3 70B Versatile on high-speed LPUs.
    OpenAI-compatible chat completions interface.
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model

    def name(self) -> str:
        return f"Groq ({self.model})"

    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    async def complete(
        self, prompt: str, system_prompt: str, timeout: Optional[float] = None
    ) -> str:
        t_limit = timeout if timeout is not None else get_llm_timeout()
        if not self.is_configured():
            raise ValueError("GROQ_API_KEY is not configured")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Groq returned no choices")

        return choices[0].get("message", {}).get("content", "")


class MockProvider(LLMProvider):
    """
    Mock provider for automated deterministic testing without real credentials.
    """

    def __init__(
        self,
        provider_name: str = "MockProvider",
        response_text: str = "",
        should_fail: bool = False,
        should_timeout: bool = False,
        delay_seconds: float = 0.0,
    ) -> None:
        self.provider_name = provider_name
        self.response_text = response_text
        self.should_fail = should_fail
        self.should_timeout = should_timeout
        self.delay_seconds = delay_seconds

    def name(self) -> str:
        return self.provider_name

    def is_configured(self) -> bool:
        return True

    async def complete(
        self, prompt: str, system_prompt: str, timeout: Optional[float] = None
    ) -> str:
        t_limit = timeout if timeout is not None else get_llm_timeout()
        if self.delay_seconds > 0:
            import asyncio
            await asyncio.sleep(self.delay_seconds)

        if self.should_timeout:
            import asyncio
            await asyncio.sleep(t_limit + 0.05)

        if self.should_fail:
            raise RuntimeError(f"{self.provider_name} simulated execution failure")

        return self.response_text


class LLMManager:
    """
    Manages primary and fallback LLM completions.
    Tries Primary (Gemini) -> Fallback (Groq) -> Returns (text, provider_name).
    """

    def __init__(
        self,
        primary_provider: Optional[LLMProvider] = None,
        fallback_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.primary = primary_provider or GeminiProvider()
        self.fallback = fallback_provider or GroqProvider()

    def has_any_provider(self) -> bool:
        return self.primary.is_configured() or self.fallback.is_configured()

    async def complete_with_fallback(
        self, prompt: str, system_prompt: str, timeout: Optional[float] = None
    ) -> Tuple[Optional[str], str]:
        """
        Executes completion with resilient fallback.
        Returns:
            (response_text: Optional[str], provider_used: str)
        """
        t_limit = timeout if timeout is not None else get_llm_timeout()
        # 1. Try Primary Provider (Gemini 2.5 Flash)
        if self.primary.is_configured():
            try:
                res = await self.primary.complete(prompt, system_prompt, timeout=t_limit)
                if res and res.strip():
                    return res, self.primary.name()
            except Exception as e:
                # Log provider failure and try fallback
                pass

        # 2. Try Fallback Provider (Groq / Llama 3.3 70B)
        if self.fallback.is_configured():
            try:
                res = await self.fallback.complete(prompt, system_prompt, timeout=t_limit)
                if res and res.strip():
                    return res, self.fallback.name()
            except Exception as e:
                pass

        return None, "all_providers_failed"
