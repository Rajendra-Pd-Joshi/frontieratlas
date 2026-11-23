"""LiteLLM gateway — one entry point for all LLM calls in this project.

Why a gateway (LiteLLM Router) instead of hand-rolled per-provider
calls: the Router owns model routing, fallbacks, retries and cooldowns
in one place, so the extraction engine only says "complete this" and
the gateway decides which model answers. Portkey's gateway was
considered and not used: it needs a separate Portkey API key / hosted
account, while LiteLLM Router runs locally with the single OpenRouter
key this project uses.

Model chain (fallback order, cost first):
  gpt-4o-mini -> gpt-oss-120b -> gpt-oss-20b -> gpt-6-luna
All four are OpenRouter models on the same chat-completions endpoint.
Probe results (data/openrouter_probe.json, 23 Nov 2026): the user-
specified ":batch" variants (gpt-oss-120b:batch, gpt-oss-20b:batch,
gpt-6-luna:batch) are Batch-API-only — OpenRouter rejects them on
chat/completions with 404 — so the live chain uses the same models
without the ":batch" suffix. The ":batch" slugs are kept in
BATCH_ONLY_SLUGS for a future bulk Batch-API path. Note the gpt-oss
models are reasoning models: they need a real max_tokens budget
(max_tokens=10 returns empty content).
"""
from __future__ import annotations

import os

import litellm
from litellm import Router

OPENROUTER_URL = "https://openrouter.ai/api/v1"

# (model_name, litellm model id, max input tokens)
OPENROUTER_MODELS: list[tuple[str, str, int]] = [
    ("gpt-4o-mini", "openrouter/openai/gpt-4o-mini", 8000),
    ("gpt-oss-120b", "openrouter/openai/gpt-oss-120b", 16000),
    ("gpt-oss-20b", "openrouter/openai/gpt-oss-20b", 8000),
    ("gpt-6-luna", "openrouter/openai/gpt-6-luna", 16000),
]
PRIMARY = "gpt-4o-mini"
BUDGETS = {name: budget for name, _, budget in OPENROUTER_MODELS}
MODEL_IDS = {name: mid for name, mid, _ in OPENROUTER_MODELS}
# Router model_name lookup by the exact user slug, e.g.
# "openai/gpt-oss-120b:batch" -> "gpt-oss-120b-batch".
MODEL_NAME_BY_SLUG = {mid.removeprefix("openrouter/"): name
                       for name, mid, _ in OPENROUTER_MODELS}
# The exact slugs the user specified (for docs / probe parity).
USER_MODEL_SLUGS = ["openai/gpt-4o-mini", "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b", "openai/gpt-6-luna"]
# Batch-API-only variants the user specified; NOT usable on
# chat/completions (verified 23 Nov 2026). Kept for a future bulk path.
BATCH_ONLY_SLUGS = ["openai/gpt-oss-120b:batch", "openai/gpt-oss-20b:batch",
                    "openai/gpt-6-luna:batch"]


def build_router(api_key: str | None = None) -> Router:
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    model_list = [
        {"model_name": name,
         "litellm_params": {"model": mid, "api_key": key,
                            "api_base": OPENROUTER_URL}}
        for name, mid, _ in OPENROUTER_MODELS
    ]
    fallbacks = [{PRIMARY: [name for name, _, _ in OPENROUTER_MODELS[1:]]}]
    return Router(model_list=model_list, fallbacks=fallbacks,
                  num_retries=2, cooldown_time=30, allowed_fails=3)


class OpenRouterGateway:
    """Thin wrapper over the LiteLLM Router.

    complete() returns (text, model_actually_used). The Router applies
    its fallback chain internally; the caller only sees the final text
    and which model produced it.
    """

    def __init__(self, router: Router | None = None, api_key: str | None = None):
        self.router = router or build_router(api_key)
        self.calls: list[dict] = []

    async def complete(self, messages: list[dict], model: str = PRIMARY,
                       json_mode: bool = True) -> tuple[str, str]:
        kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self.router.acompletion(**kwargs)
        text = (resp.choices[0].message.content or "")
        used = getattr(resp, "model", None) or model
        self.calls.append({"model": model, "used": used, "chars": len(text)})
        return text, used

    def complete_sync(self, messages: list[dict], model: str = PRIMARY,
                      json_mode: bool = True) -> tuple[str, str]:
        kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.router.completion(**kwargs)
        text = (resp.choices[0].message.content or "")
        used = getattr(resp, "model", None) or model
        self.calls.append({"model": model, "used": used, "chars": len(text)})
        return text, used


# Silence LiteLLM's cost-map fetch warning in offline/proxied runs.
litellm.suppress_debug_info = True
