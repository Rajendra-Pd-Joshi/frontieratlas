# FrontierAtlas — Technical Architecture

**Scope.** Ingestion pipeline for the Intelligence Graph: startups,
products, research papers, AI jobs and news. Demo run (23 Nov 2026):
1,000 startups, 1,000 products, 1,000 papers (990 with live GitHub
stars), 80 news + 10 jobs in the last 24h. Every record carries its
source URL; no record is generated without a fetched source.

## 1. Pipeline

A LangGraph state machine (`src/graph.py`): fetch_startups →
fetch_papers → fetch_products → resolve_entities → validate. Each node
writes raw JSONL first, then canonical JSONL, so a crash resumes at the
failed node instead of restarting. Validation enforces the assignment
minimums (1,000 per type, ≥95% of papers with live stars) and fails
the run otherwise. A FastAPI service (`src/api.py`) exposes
`/run/phase1`, `/runs/{id}` and `/review` (the human-in-the-loop queue
for low-confidence merges). All HTTP goes through httpx with a bounded
pool (20 connections), per-host semaphores, and retries (see §3).

Raw fidelity is the first rule: the LLM only structures text that was
actually fetched. Schema validation (Pydantic) rejects anything else
to a dead-letter file. Hallucinated data is a disqualification, so the
pipeline is built to fail loudly rather than invent.

## 2. Scale strategy — 500,000 records, no code changes

The demo already paginates every source (YC Algolia 100/page, PwC
rows API 100/page, HF Spaces cursor pagination, arXiv id-list joins).
Scaling to 500k is infrastructure, not new code:

- **Partition by source + shard key.** Workers are sharded by source
  (e.g. YC pages 0–9, 10–19…) and by entity key hash, each with its
  own checkpoint (last page / cursor) in Postgres. A crashed worker
  resumes from its checkpoint; a new worker takes an unclaimed shard.
- **Backpressure.** Bounded queues (asyncio semaphore + httpx pool);
  producers pause when the queue is full. No unbounded in-memory lists
  anywhere in the pipeline.
- **Idempotent writes.** Canonical key = source + canonical URL (or
  arXiv ID). Re-running a shard upserts, never duplicates.
- **Throughput math.** Demo: 3,000 records in ~2 min single-node
  (~25/s, dominated by GitHub star lookups at concurrency 5). At
  concurrency 50, 500k records is ~6–8 hours on one node, linear
  with workers.

## 3. Handling 413s & 429s

LLM gateway (LiteLLM Router, src/llm_gateway.py) — one entry point
for all LLM calls, with built-in fallbacks, retries and cooldowns.
OpenRouter only; no other provider key is used anywhere in this
project. Model chain (fallback order, cost first): gpt-4o-mini (8k
input tokens) → gpt-oss-120b (16k) → gpt-oss-20b (8k) →
gpt-6-luna (16k). The user-specified ":batch" variants are Batch-API-
only: OpenRouter rejects them on chat/completions with 404 (verified
by the live probe, data/openrouter_probe.json), so the live chain uses
the same models without the suffix; the ":batch" slugs are kept for a
future bulk Batch-API path. The gpt-oss models are reasoning models
and need a real max_tokens budget (max_tokens=10 returns empty
content). Portkey's gateway was considered and not used: it needs a
separate Portkey account/key, while LiteLLM Router runs locally on
the single OpenRouter key.

- **429:** exponential backoff with full jitter, max 3 attempts per
  tier, then fall to the next tier. Three consecutive failures open a
  per-tier circuit breaker for the rest of the run.
- **413:** never retry the same payload. First, split the text into
  paragraph-packed chunks under half the tier budget (title kept as
  context in every chunk), extract per chunk and merge the JSON. If a
  tier still returns 413, shrink head+tail (keep the densest 60% head
  and 40% tail, drop the middle) and retry once, then fall through.
- **Validation:** every LLM output must parse as JSON and pass the
  Pydantic schema; one repair attempt per tier (re-prompt with the
  parse error), then the next tier. All attempts are logged with tier,
  tokens, latency and outcome.

Mock-tested (`tests/test_llm_orchestrator.py`): 429 fallback, chunked
extraction (9 chunks merged), 413 shrink-and-retry — all pass.

## 4. Freshness & exactly-once

- **Dates:** feed/API timestamp first, then article meta tags, then
  relative dates ("2 hours ago" → UTC), then a labelled heuristic
  (first_seen_at, only for daily feeds, flagged
  `date_source="heuristic:first_seen"`). All dates are normalised to
  UTC ISO-8601. An item counts only if published within 24h of the run.
- **Dedupe across nodes:** `seen` table in SQLite (Postgres at scale):
  key = sha256(source + canonical URL), inserted first; a duplicate
  insert means "already processed". Validation runs use a separate
  seen DB so they never eat the submission run's window.
- **Dynamic metrics:** GitHub stars are a time-series snapshot
  (stars + collectedAt per run), never an overwrite — otherwise the
  trend is destroyed.

## 5. Storage

- **Primary: Postgres.** Canonical records, checkpoints, the `seen`
  table and the entity-mapping log. JSONB for flexible content,
  indexes on canonical keys and URLs.
- **Vector: pgvector** (same Postgres). Embeddings of titles /
  descriptions for semantic dedupe and near-duplicate detection in
  entity resolution — the deterministic resolver runs first, pgvector
  only proposes candidates above a similarity threshold.
- **Graph: Neo4j.** Startup–product, paper–repo, founder–company
  edges, and lineage (every edge keeps its source URL). The graph is
  derived from Postgres, never the system of record.
- **Raw lake: JSONL** (S3/GCS at scale). Raw payloads are never edited;
  re-normalisation always replays from raw.

## 6. Anti-bot & high-value sources

Hierarchy, in order: (1) official API / dataset, (2) RSS / sitemap,
(3) browser automation (Playwright async) only for JS-rendered pages.
That order is why the demo works: arXiv API, YC Algolia (public
search-only key, refetched per run — it expires in ~24h), PapersWithCode
data via Hugging Face datasets, Greenhouse/Ashby job-board APIs.

For Cloudflare / Datadome-protected domains: real browser sessions
(Playwright, persistent context, realistic viewport + headers),
human-like pacing, session/cookie reuse, and residential-proxy
rotation only as a last resort. No captcha solving is claimed: a
CAPTCHA is a stop signal — the item is queued for a later run or a
manual pass. TechCrunch (403) and VentureBeat (429) are documented
as browser-only sources for exactly this reason. Scraping always
respects robots.txt and the site's terms.

## 7. Entity resolution

Deterministic first: normalise (lowercase, strip legal suffixes and
punctuation — but not "AI"/"Labs", which are part of AI company names),
alias table, then a 50-startup seed list (domain / GitHub org match).
Fuzzy (difflib ≥0.90) auto-merges; 0.80–0.90 goes to the human-review
queue and is never auto-merged; no match creates a new canonical
entity instead of forcing a seed match. Full run: 3 seed, 1 alias,
7 review, 989 new. Every decision is in the Entity Mapping Log.

## 8. Evaluation & limits (honest notes)

- DeepEval suite (`src/deepeval_suite.py`, official DeepEval 4.3.4
  metrics, judge = gpt-4o-mini via OpenRouter): 15 golden cases,
  live run 23 Nov 2026 — GEval schema-compliance 1.00, GEval
  groundedness 1.00, Faithfulness 0.93, Answer Relevancy 1.00,
  Hallucination 1.00, Contextual Recall 1.00, custom Extraction
  Field F1 1.00. Scores are 0–1; a case passes iff score >= threshold
  (0.7, F1 0.8). Faithfulness was 1.00 on the first run — the judge
  is non-deterministic (±0.07 run-to-run).
- Guardrails (`src/guardrails.py`): LangChain JSON parser guard,
  grounded-values guard (a value not in the source text is dropped,
  never kept), PII guard, input-injection flagging. Live extraction
  run: 1.0 valid-JSON, 1.0 mean field F1 across 15 cases.
- Products are all priced FREE: they come from Hugging Face Spaces
  (public spaces). Pricing diversity needs the Product Hunt GraphQL
  tier (developer token not yet available).
- PapersWithCode's own API is dead (redirects to Hugging Face Papers);
  the demo uses their archived dataset on Hugging Face instead.
- News volume depends on the hour: the validation run found 80 fresh
  items, 68 of them via Google News, whose article redirects 429 under
  bulk fetch — mitigated with concurrency 2 + retries; direct feeds
  (Verge, Wired, NYT, IEEE) give reliable full text.
- 10 of 1,000 paper repo links are dead (404) in the source dataset;
  they are recorded with no stars, not dropped or invented.
