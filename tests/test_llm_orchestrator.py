"""Mock-transport tests for the LLM orchestrator — no API keys, no network.

Run: .venv/bin/python -m tests.test_llm_orchestrator
Scenarios:
  A. Gemini 429s three times -> falls back to Groq, which succeeds.
  B. Oversize text -> chunked extraction, results merged.
  C. Gemini 413 once -> payload shrinks, retry succeeds on the same tier.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

os.environ.update({"GEMINI_API_KEY": "fake", "GROQ_API_KEY": "fake", "DEEPSEEK_API_KEY": "fake",
                   "OPENROUTER_API_KEY": "fake"})

from src.llm_orchestrator import LLMOrchestrator, Tier, estimate_tokens, shrink, split_chunks, TIERS  # noqa: E402

# Tests pin the classic chain (Gemini -> Groq -> DeepSeek) so they do not
# depend on the OpenRouter-first default used in production.
CLASSIC = [t for t in TIERS if t.name in ("gemini", "groq", "deepseek")]


def _json_resp(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _gemini_ok(text: str) -> httpx.Response:
    return _json_resp({"candidates": [{"content": {"parts": [{"text": text}]}}]})


def _openai_ok(text: str) -> httpx.Response:
    return _json_resp({"choices": [{"message": {"content": text}}]})


def test_429_fallback():
    calls = {"gemini": 0, "groq": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "generativelanguage" in str(request.url):
            calls["gemini"] += 1
            return httpx.Response(429, json={"error": "rate limited"})
        calls["groq"] += 1
        return _openai_ok('{"name": "OpenAI", "employees": 1000}')

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            orch = LLMOrchestrator(client, tiers=CLASSIC)
            result, report = await orch.extract(
                "OpenAI is an AI company with 1000 employees.",
                "Extract the company as JSON with keys name, employees.")
            assert result == {"name": "OpenAI", "employees": 1000}, result
            assert report["tier"] == "groq", report
            assert calls["gemini"] == 3, calls  # 3 attempts, then fallback
            assert calls["groq"] == 1, calls
    asyncio.run(run())
    print("A 429 fallback: PASS")


def test_chunking():
    calls = {"gemini": 0}
    paras = [f"Paragraph {i} about AI startup number {i} and its funding round." for i in range(2000)]
    text = "\n\n".join(paras)
    assert estimate_tokens(text) > 8000

    def handler(request: httpx.Request) -> httpx.Response:
        calls["gemini"] += 1
        return _gemini_ok(json.dumps({"items": [f"item-{calls['gemini']}"]}))

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            orch = LLMOrchestrator(client, tiers=CLASSIC)
            result, report = await orch.extract(text, "Extract items as JSON {\"items\": [...]}.",
                                                title="Test doc")
            assert report["chunked"] > 1, report
            assert len(result["items"]) == report["chunked"], result
    asyncio.run(run())
    print(f"B chunking: PASS ({calls['gemini']} chunks)")


def test_413_shrink():
    calls = {"gemini": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["gemini"] += 1
        if calls["gemini"] == 1:
            return httpx.Response(413, json={"error": "too large"})
        return _gemini_ok('{"ok": true}')

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            orch = LLMOrchestrator(client, tiers=CLASSIC)
            # Force the 413 path by calling _attempt directly with a big prompt.
            result = await orch._attempt(orch.tiers[0], "x" * 100, None)
            assert result == {"ok": True}, result
            assert calls["gemini"] == 2, calls
    asyncio.run(run())
    print("C 413 shrink-and-retry: PASS")


def test_helpers():
    assert estimate_tokens("abcd" * 100) == 100
    assert len(shrink("a" * 10000, 100)) <= 450  # ~400 chars + marker
    chunks = split_chunks("p1\n\np2\n\np3", 10)
    assert len(chunks) >= 1
    print("helpers: PASS")


def test_openrouter_primary():
    """OpenRouter (gpt-4o-mini) is tier 0 in production and succeeds first."""
    calls = {"openrouter": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert "openrouter.ai" in str(request.url)
        calls["openrouter"] += 1
        return _openai_ok('{"name": "Anthropic"}')

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            orch = LLMOrchestrator(client)  # default tiers: OpenRouter first
            assert orch.tiers[0].name == "openrouter", [t.name for t in orch.tiers]
            result, report = await orch.extract("Anthropic is an AI company.",
                                                'Extract as JSON {"name": ...}.')
            assert result == {"name": "Anthropic"}, result
            assert report["tier"] == "openrouter", report
    asyncio.run(run())
    print("D OpenRouter primary tier: PASS")


if __name__ == "__main__":
    test_helpers()
    test_429_fallback()
    test_chunking()
    test_413_shrink()
    test_openrouter_primary()
    print("ALL LLM ORCHESTRATOR TESTS PASSED")
