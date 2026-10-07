"""Optional Groq-backed narration for the Overview Agent.

The template summaries are precise but dense ("wall of text with numbers"). When
a Groq API key is present, the agent asks a small hosted model for a **short,
plain-language** version of the same facts; the numbers stay in the payload for
the panels.

Configuration (environment):

``GROQ_API_KEY``     enables the narration when set
``BHEL_LLM``         ``off`` disables it even with a key present
``BHEL_LLM_MODEL``   model id, default ``llama-3.1-8b-instant``
``BHEL_LLM_TIMEOUT`` request timeout in seconds, default 6
``BHEL_LLM_ENDPOINT`` OpenAI-compatible endpoint (defaults to Groq's)

Design rules: never block the demo for long, never call more than once per
few seconds, cache identical situations, and **always** fall back to the
template text when anything goes wrong (missing key, timeout, bad response).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

DEFAULT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TIMEOUT = 6.0
#: Minimum seconds between two network calls (the UI polls every few seconds).
MIN_INTERVAL_S = 5.0
#: How many rendered summaries to remember (keyed by the facts they describe).
CACHE_SIZE = 64

SYSTEM_PROMPT = (
    "You are the shift-supervisor assistant for a thermal power plant boiler. "
    "Summarise the situation for an operator in at most two short sentences of "
    "plain English. Say what is happening and where. Avoid numbers unless one is "
    "essential; never invent readings, never speculate about causes beyond the "
    "'likely_pattern' given, and do not greet or explain yourself. If every part "
    "is nominal, say so in one short sentence."
)


@dataclass
class GroqNarrator:
    """Turns the agent's facts into a short narrative, with safe fallbacks."""

    api_key: str | None = field(default_factory=lambda: os.environ.get("GROQ_API_KEY") or None)
    model: str = field(default_factory=lambda: os.environ.get("BHEL_LLM_MODEL", DEFAULT_MODEL))
    endpoint: str = field(default_factory=lambda: os.environ.get("BHEL_LLM_ENDPOINT", DEFAULT_ENDPOINT))
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("BHEL_LLM_TIMEOUT", DEFAULT_TIMEOUT))
    )
    min_interval_s: float = MIN_INTERVAL_S
    enabled: bool = field(
        default_factory=lambda: (os.environ.get("BHEL_LLM", "groq").lower() != "off")
    )
    last_error: str | None = None
    calls: int = 0
    _cache: dict[str, str] = field(default_factory=dict)
    _last_call: float = 0.0

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.api_key)

    # ------------------------------------------------------------------
    def summarise(self, facts: dict) -> tuple[str | None, str]:
        """Return ``(text, source)``; ``source`` is ``groq``/``template``."""
        if not self.available:
            return None, "template"
        signature = json.dumps(facts, sort_keys=True, default=str)
        cached = self._cache.get(signature)
        if cached:
            return cached, "groq"
        now = time.monotonic()
        if now - self._last_call < self.min_interval_s:
            return None, "throttled"
        self._last_call = now
        try:
            text = self._request(facts)
        except Exception as exc:  # network, auth, malformed response — never fatal
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return None, "template"
        if not text:
            self.last_error = "empty completion"
            return None, "template"
        self.last_error = None
        if len(self._cache) >= CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[signature] = text
        return text, "groq"

    # ------------------------------------------------------------------
    def _request(self, facts: dict) -> str:
        import requests

        response = requests.post(
            self.endpoint,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 120,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(facts, default=str)},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        self.calls += 1
        return (payload["choices"][0]["message"]["content"] or "").strip()
