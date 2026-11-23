"""Shared async HTTP + logging + raw-store helpers.

Design decisions (evaluated in the architecture doc):
- httpx AsyncClient with a bounded connection pool (concurrency is a config
  value, not unbounded) and a per-host polite delay.
- This VM reaches the internet through an egress proxy with TLS interception:
  the proxy URL and CA bundle come from the environment (HTTPS_PROXY /
  SSL_CERT_FILE) — never hardcoded, never committed. httpx is created with
  trust_env=False because the stock NO_PROXY value breaks httpx's parser.
- Retries: exponential backoff with full jitter. 429 / 5xx are retried,
  413 is NOT retried (payload must shrink — see llm_orchestrator).
- Raw fidelity: every fetched payload is appended to data/raw/<name>.jsonl
  with fetched_at + source URL before any normalisation runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("frontieratlas")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CANON = ROOT / "data" / "canonical"


def make_client(headers: dict | None = None) -> httpx.AsyncClient:
    """Async client honouring the environment's egress proxy + CA bundle."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        trust_env=False,
        proxy=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
        verify=os.environ.get("SSL_CERT_FILE") or True,
        headers=headers or {},
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        follow_redirects=True,
    )


class RateLimited(Exception):
    pass


async def fetch_json(client: httpx.AsyncClient, url: str, *,
                     headers: dict | None = None, sem: asyncio.Semaphore,
                     retries: int = 4, timeout: float = 30.0) -> dict | list:
    """GET JSON with backoff+jitter on 429/5xx. Returns parsed JSON."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        async with sem:
            try:
                r = await client.get(url, headers=headers, timeout=timeout)
                if r.status_code == 429:
                    raise RateLimited(f"429 {url}")
                if r.status_code >= 500:
                    raise httpx.HTTPError(f"{r.status_code} {url}")
                if 400 <= r.status_code < 500:
                    # Other 4xx (e.g. 404 dead repo) will never succeed on retry.
                    r.raise_for_status()
                r.raise_for_status()
                return r.json()
            except RateLimited as e:
                last = e
                if attempt == retries:
                    raise
                delay = (2 ** attempt) + random.uniform(0, 1)
                log.warning("retry %s attempt=%d delay=%.1fs err=%s", url, attempt, delay, e)
                await asyncio.sleep(delay)
            except httpx.HTTPStatusError as e:
                # 4xx: fail fast, no retry. 5xx already raised as HTTPError above.
                raise
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                last = e
                if attempt == retries:
                    raise
                delay = (2 ** attempt) + random.uniform(0, 1)
                log.warning("retry %s attempt=%d delay=%.1fs err=%s", url, attempt, delay, e)
                await asyncio.sleep(delay)
    raise last  # noqa: B904


async def fetch_text(client: httpx.AsyncClient, url: str, *,
                     headers: dict | None = None, sem: asyncio.Semaphore,
                     retries: int = 4, timeout: float = 30.0) -> str:
    last: Exception | None = None
    for attempt in range(retries + 1):
        async with sem:
            try:
                r = await client.get(url, headers=headers, timeout=timeout)
                if r.status_code == 429:
                    raise RateLimited(f"429 {url}")
                if r.status_code >= 500:
                    raise httpx.HTTPError(f"{r.status_code} {url}")
                r.raise_for_status()
                return r.text
            except RateLimited as e:
                last = e
                if attempt == retries:
                    raise
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
            except httpx.HTTPStatusError:
                raise
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                last = e
                if attempt == retries:
                    raise
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
    raise last  # noqa: B904


def append_raw(name: str, record: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_canonical(name: str, records: list[dict]) -> Path:
    CANON.mkdir(parents=True, exist_ok=True)
    path = CANON / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
