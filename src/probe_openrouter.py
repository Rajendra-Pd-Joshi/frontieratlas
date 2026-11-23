"""Live probe of the OpenRouter model chain.

Minimal call per model (max_tokens 10) to prove which slugs answer on
the chat-completions endpoint — especially the ":batch" variants, which
OpenRouter documents for its Batch API and may reject here. Writes
data/openrouter_probe.json (no key is ever written to disk).

Usage: OPENROUTER_API_KEY=... .venv/bin/python -m src.probe_openrouter
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from .common import ROOT, make_client
from .llm_gateway import OPENROUTER_URL, USER_MODEL_SLUGS


async def probe(model: str) -> dict:
    t0 = time.time()
    async with make_client() as client:
        try:
            r = await client.post(
                f"{OPENROUTER_URL}/chat/completions",
                json={"model": model,
                      "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                      # Reasoning models (gpt-oss) spend tokens thinking
                      # first; max_tokens=10 returns empty content.
                      "max_tokens": 300 if "gpt-oss" in model else 10,
                      "temperature": 0},
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/Rajendra-Pd-Joshi",
                         "X-Title": "FrontierAtlas"},
                timeout=90)
        except Exception as e:  # noqa: BLE001
            return {"model": model, "ok": False, "http_status": None,
                    "error": str(e)[:200], "ms": round((time.time() - t0) * 1000)}
    ms = round((time.time() - t0) * 1000)
    if r.status_code != 200:
        return {"model": model, "ok": False, "http_status": r.status_code,
                "error": r.text[:200], "ms": ms}
    try:
        content = r.json()["choices"][0]["message"].get("content") or ""
    except Exception:  # noqa: BLE001
        content = ""
    if not content.strip():
        return {"model": model, "ok": False, "http_status": 200,
                "error": "empty content (reasoning-only or no completion)",
                "ms": ms}
    return {"model": model, "ok": True, "http_status": 200,
            "reply": content.strip()[:80], "ms": ms}


async def main_async() -> int:
    results = [await probe(model) for model in USER_MODEL_SLUGS]
    out = ROOT / "data" / "openrouter_probe.json"
    out.write_text(json.dumps({"probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "models": results}, indent=2))
    for r in results:
        print(f"{r['model']}: {'OK' if r['ok'] else 'FAIL'} "
              f"({r.get('http_status')}, {r['ms']}ms)"
              + (f" reply={r.get('reply')!r}" if r.get('reply') else "")
              + (f" err={r.get('error','')[:100]}" if not r['ok'] else ""))
    return 0 if all(r["ok"] for r in results) else 1


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set")
        return 2
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
