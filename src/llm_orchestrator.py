"""Phase III — multi-tier LLM extraction engine.

Fallback chain: OpenRouter (gpt-4o-mini, user's preferred cheap tier)
-> Gemini Flash -> Groq Llama 3 -> DeepSeek.
(The chain order is by cost first, per the user's preference for
OpenRouter: gpt-4o-mini is the cheapest capable tier, Gemini Flash is
the fast fallback, Groq is the low-latency fallback, DeepSeek is the
capability fallback. The OpenRouter model is configurable via
OPENROUTER_MODEL, default openai/gpt-4o-mini.)

413 (Payload Too Large) handling — never retry the same payload:
  1. Estimate tokens (~4 chars/token, conservative).
  2. If the text exceeds a tier's budget, split it into semantically
     dense chunks (paragraph-packed, each chunk keeps the document title
     as context) and extract per chunk, then merge the JSON results.
  3. If a tier still returns 413, shrink the payload by half (drop the
     lowest-priority middle sections, keep head + tail) and retry once.
  4. Then move to the next tier.

429 (Too Many Requests) handling:
  exponential backoff with full jitter, max 3 attempts per tier, plus a
  per-tier circuit breaker: 3 consecutive failures opens the breaker and
  the tier is skipped for the rest of the run.

Every attempt is logged (tier, model, tokens in/out, latency, outcome).
LLM output is never trusted blindly: it must parse as JSON and pass the
caller-supplied validator, or it is a failure and the chain moves on.
One repair attempt per tier is allowed (re-prompt with the parse error).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field

import httpx

from .common import log

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Tier:
    name: str
    model: str
    url: str
    api_key_env: str
    max_input_tokens: int
    style: str  # "gemini" or "openai"
    failures: int = 0
    breaker_open: bool = False


TIERS = [
    Tier("openrouter", os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
         "https://openrouter.ai/api/v1/chat/completions",
         "OPENROUTER_API_KEY", 8000, "openai"),
    Tier("gemini", "gemini-2.0-flash",
         "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
         "GEMINI_API_KEY", 8000, "gemini"),
    Tier("groq", "llama-3.3-70b-versatile",
         "https://api.groq.com/openai/v1/chat/completions",
         "GROQ_API_KEY", 6000, "openai"),
    Tier("deepseek", "deepseek-chat",
         "https://api.deepseek.com/chat/completions",
         "DEEPSEEK_API_KEY", 16000, "openai"),
]


def split_chunks(text: str, max_tokens: int, title: str = "") -> list[str]:
    """Paragraph-packed chunks, each under max_tokens, title kept as context."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]
    chunks, current = [], (f"{title}\n\n" if title else "")
    for p in paras:
        if estimate_tokens(current + "\n\n" + p) > max_tokens and current.strip():
            chunks.append(current)
            current = (f"{title}\n\n" if title else "") + p
        else:
            current = (current + "\n\n" + p) if current.strip() else current + p
    if current.strip():
        chunks.append(current)
    return chunks


def shrink(text: str, target_tokens: int) -> str:
    """Keep head + tail (the semantically densest parts), drop the middle."""
    marker = "\n\n[... middle truncated to fit context window ...]\n\n"
    budget_chars = target_tokens * CHARS_PER_TOKEN - len(marker)
    if len(text) <= budget_chars:
        return text
    head = int(budget_chars * 0.6)
    tail = budget_chars - head
    return text[:head] + marker + text[-tail:]


def _extract_json(raw: str) -> dict | list:
    """Pull the first JSON object/array out of model output (incl. fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
    if m:
        return json.loads(m.group(1))
    return json.loads(raw)


def _merge(results: list[dict | list]) -> dict | list:
    if all(isinstance(r, dict) for r in results):
        merged: dict = {}
        for r in results:
            for k, v in r.items():
                if isinstance(v, list) and isinstance(merged.get(k), list):
                    merged[k].extend(v)
                elif k not in merged:
                    merged[k] = v
        return merged
    if all(isinstance(r, list) for r in results):
        out: list = []
        for r in results:
            out.extend(r)
        return out
    return results[0] if results else {}


class LLMOrchestrator:
    def __init__(self, client: httpx.AsyncClient | None = None, tiers: list[Tier] | None = None):
        self.client = client
        self.tiers = tiers if tiers is not None else TIERS
        self.attempt_log: list[dict] = []

    async def _call_tier(self, tier: Tier, prompt: str) -> str:
        key = os.environ.get(tier.api_key_env, "")
        if tier.style == "gemini":
            url = f"{tier.url}?key={key}"
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"}}
            headers = {"Content-Type": "application/json"}
        else:
            url, headers = tier.url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            if tier.name == "openrouter":
                headers["HTTP-Referer"] = "https://github.com/Rajendra-Pd-Joshi"
                headers["X-Title"] = "FrontierAtlas"
            body = {"model": tier.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}, "temperature": 0}
        assert self.client is not None
        r = await self.client.post(url, json=body, headers=headers, timeout=120)
        if r.status_code == 413:
            raise _PayloadTooLarge(f"413 from {tier.name}")
        if r.status_code == 429:
            raise _RateLimited(f"429 from {tier.name}")
        r.raise_for_status()
        data = r.json()
        if tier.style == "gemini":
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return data["choices"][0]["message"]["content"]

    async def _attempt(self, tier: Tier, prompt: str, validator) -> dict | list:
        """One tier, up to 3 attempts (429 backoff, one 413 shrink)."""
        shrunk = False
        for attempt in range(3):
            t0 = time.time()
            try:
                raw = await self._call_tier(tier, prompt)
                parsed = _extract_json(raw)
                if validator is not None:
                    validator(parsed)
                self.attempt_log.append({"tier": tier.name, "model": tier.model,
                                         "attempt": attempt, "outcome": "ok",
                                         "ms": round((time.time() - t0) * 1000),
                                         "tokens_in": estimate_tokens(prompt)})
                tier.failures = 0
                return parsed
            except _PayloadTooLarge:
                self.attempt_log.append({"tier": tier.name, "outcome": "413", "attempt": attempt})
                if shrunk:
                    raise
                prompt = shrink(prompt, tier.max_input_tokens // 2)
                shrunk = True
            except _RateLimited:
                self.attempt_log.append({"tier": tier.name, "outcome": "429", "attempt": attempt})
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
            except Exception as e:  # noqa: BLE001 - parse/validation failure: repair once
                self.attempt_log.append({"tier": tier.name, "outcome": f"error:{type(e).__name__}",
                                         "attempt": attempt, "detail": str(e)[:200]})
                if attempt == 0:
                    prompt = prompt + f"\n\nYour previous output failed validation: {e}. Return valid JSON only."
                else:
                    raise
        raise RuntimeError(f"tier {tier.name} exhausted")

    async def extract(self, text: str, instruction: str, validator=None,
                      title: str = "") -> tuple[dict | list, dict]:
        """Extract structured JSON from text. Returns (result, report)."""
        for tier in self.tiers:
            if tier.breaker_open or not os.environ.get(tier.api_key_env):
                if not os.environ.get(tier.api_key_env):
                    self.attempt_log.append({"tier": tier.name, "outcome": "skipped:no_key"})
                continue
            prompt = f"{instruction}\n\n---\n\n{text}"
            try:
                if estimate_tokens(prompt) > tier.max_input_tokens:
                    chunks = split_chunks(text, tier.max_input_tokens // 2, title=title)
                    parts = []
                    for c in chunks:
                        parts.append(await self._attempt(
                            tier, f"{instruction}\n\n---\n\n{c}", validator))
                    return _merge(parts), {"tier": tier.name, "chunked": len(chunks),
                                           "attempts": self.attempt_log}
                result = await self._attempt(tier, prompt, validator)
                return result, {"tier": tier.name, "chunked": 0, "attempts": self.attempt_log}
            except Exception as e:  # noqa: BLE001 - move to next tier
                tier.failures += 1
                if tier.failures >= 3:
                    tier.breaker_open = True
                    log.warning("circuit breaker opened for %s", tier.name)
                log.warning("tier %s failed (%s); falling back", tier.name, e)
                continue
        raise RuntimeError("all LLM tiers failed or have no API key")


class _PayloadTooLarge(Exception):
    pass


class _RateLimited(Exception):
    pass
