# FrontierAtlas — GraphOne / FrontierAtlas AI Engineer Demo Task

Production-grade, fault-tolerant ingestion pipeline for the global
Intelligence Graph (startups, founders, products, AI jobs, research
papers, real-time news).

Public Google Sheet (6 tabs):
https://docs.google.com/spreadsheets/d/1Ill1fRX481TuNFUl1Tqw9o7CBlYgPOq5B8zY6QldRug/edit?usp=sharing

## Results (23 Nov 2026)

**Phase 0 — source validation: 8/8 OK**
(`python -m src.validate_sources`, report in
`data/validation_report.json`).

**Phase I — full run: 1,000 startups, 1,000 products, 1,000 papers,
990 with live GitHub stars** (10 repo links are genuinely dead, 404,
and are recorded as-is). Startups: 979/1,000 with employeeCount,
977/1,000 with website. Papers: 994/1,000 with authors and published
dates. Products are sourced from Hugging Face Spaces (public,
free-to-use spaces), so pricingModel is FREE across this set.

**Phase II — news + jobs crawler** (`python -m src.phase2`): 8 news
feeds, 8 job sources (RemoteOK, 4 Greenhouse boards, 3 Ashby boards).
Run of 23 Nov 2026 found 80 news / 10 jobs in the last 24h.
TechCrunch (403) and VentureBeat (429) are browser-only sources,
documented in the code. The Jobs/News tabs hold the 24h window as of
the run date; the pipeline re-runs them with `--full` for a fresh
window. Validation runs use a separate seen DB so they never eat the
dedupe window.

**Phase III — LLM orchestrator** (`python -m tests.test_llm_orchestrator`:
429 fallback, chunking, 413 shrink all PASS). OpenRouter-only chain:
gpt-4o-mini -> gpt-oss-120b -> gpt-oss-20b -> gpt-6-luna, with
per-tier circuit breakers. The ":batch" variants are Batch-API-only
(verified by live probe) and are NOT in the chat chain. Runs live on
the OpenRouter key only — no other provider key is used.

**Phase IV — entity resolution** (`src/entity_resolver.py`, 50-startup
seed list). Full run on 1,000 startups: 3 seed, 1 alias, 7 review,
989 new — mapping log in
`data/canonical/entity_mapping_log.jsonl`.

**Stack:** LangGraph pipeline (`src/graph.py`, sample run status ok),
OpenRouter as primary LLM tier (gpt-4o-mini, configurable via
OPENROUTER_MODEL), FastAPI service (`src/api.py`, /health and /review
verified), Dockerfile + docker-compose (Postgres + pgvector).

| Source | Check | Result |
|---|---|---|
| arXiv API | `export.arxiv.org/api/query` | 200, Atom feed |
| PapersWithCode API / paper pages | `paperswithcode.com/api/v1/papers/` | **Dead** — redirects to Hugging Face Papers, returns HTML |
| PapersWithCode data (HF datasets) | `pwc-archive/links-between-paper-and-code` | 200, parquet + rows API, fields `paper_url, paper_title, paper_arxiv_id, repo_url, is_official` |
| PapersWithCode data (HF datasets) | `pwc-archive/papers-with-abstracts` | 200, 4 parquet shards |
| Hugging Face Papers API | `huggingface.co/api/papers/search` | 200, arXiv id / authors / publishedAt |
| Semantic Scholar API | `api.semanticscholar.org/graph/v1/paper/search` | 200 |
| Y Combinator directory (Algolia) | public search-only key embedded in `ycombinator.com/companies` (`window.AlgoliaOpts`, app `45BWZJ1SGC`) | 200, returns name / website / `team_size` / industry / batch / tags |
| GitHub API | `api.github.com/repos/...` | 200; star lookups ran with an authenticated token (5,000 req/hr vs 60 unauthenticated) |
| RemoteOK jobs API | `remoteok.com/api` | 200 |
| Product Hunt Atom feed | `producthunt.com/feed` | 200, 50 entries (recent only — not a bulk source) |
| Hugging Face Spaces API | `huggingface.co/api/spaces` | 200, bulk product source |
| News RSS | The Verge AI, MIT Tech Review AI | 200, 10 items each. TechCrunch AI 403 and VentureBeat AI 429 via plain HTTP — browser-only sources |

## Evaluation

- **LLM orchestrator mock tests** — `.venv/bin/python -m tests.test_llm_orchestrator`
  (429 fallback, chunking, 413 shrink, OpenRouter primary tier).
- **Extraction eval** — `.venv/bin/python -m src.eval_extraction --mock`
  (golden set built from the real canonical records). Live run
  (`--live`, 15 cases): 1.0 valid-JSON, 1.0 mean field F1, all via
  gpt-4o-mini.
- **DeepEval suite** — `.venv/bin/python -m src.deepeval_suite`
  (official DeepEval metrics, judge = gpt-4o-mini via OpenRouter;
  report in `data/deepeval_report.json`). Live run 23 Nov 2026,
  15 cases: GEval schema-compliance 1.00, GEval groundedness 1.00,
  Faithfulness 0.93, Answer Relevancy 1.00, Hallucination 1.00,
  Contextual Recall 1.00, custom Extraction Field F1 1.00.
  (Faithfulness was 1.00 on the first run — LLM-judge variance.)
  Pytest wrapper: `.venv/bin/python -m pytest tests/test_eval_deepeval.py`.
- **LangGraph pipeline** — `.venv/bin/python -m src.graph --sample`.

## Submission artefacts

- Public Google Sheet (6 tabs) — link at the top of this README.
- `architecture.pdf` — 3 pages, built from `docs/architecture.md` with
  `.venv/bin/python docs/build_architecture_pdf.py`.
- `data/sheets/*.csv` — the 6 tabs, built with
  `.venv/bin/python -m src.export_sheets` (add `--upload` with
  `GOOGLE_SHEETS_CREDENTIALS` set to push to a public Google Sheet).
- `SUBMISSION.md` — deliverables, results and reproduction steps.

- **LangGraph** — `src/graph.py`: the pipeline is a checkpointed state
  machine (fetch -> resolve -> validate), same pattern as ALBA ONE.
- **LLM gateway** — `src/llm_gateway.py`: LiteLLM Router over the four
  OpenRouter models (gpt-4o-mini -> gpt-oss-120b -> gpt-oss-20b ->
  gpt-6-luna) with built-in fallbacks, retries and cooldowns.
- **Guardrails** — `src/guardrails.py`: LangChain JSON parser guard,
  grounded-values guard (every extracted value must appear in the
  source text), PII guard, and input-injection flagging.
- **LLM tiers** — `src/llm_orchestrator.py`: extraction engine on top of
  the gateway, with chunking, 429/413 handling and circuit breakers.
  No other provider key is used.
- **FastAPI** — `src/api.py`: `/health`, `/run/phase1`, `/runs/{id}`,
  `/review` (human-in-the-loop queue).
- **Postgres + pgvector** — scale-out store (docker-compose.yml);
  demo runs on SQLite + JSONL raw store.
- **Evals** — extraction quality is validated against Pydantic schemas
  on every LLM call, plus the golden-set harness
  (`src/eval_extraction.py`) and the official DeepEval suite
  (`src/deepeval_suite.py`).

```
sources (APIs / RSS / Algolia / HF datasets)
  -> raw store (JSONL, one file per source+run, source URL + fetched_at kept)
  -> normalise (Pydantic schemas, UTC ISO-8601, dedupe by canonical key)
  -> LLM extraction (OpenRouter chain, chunking, 429/413 handling)
  -> entity resolution (deterministic first, seed list of 50, confidence-scored)
  -> canonical store (SQLite for the demo; Postgres + pgvector/Neo4j at 500k scale)
  -> Google Sheets export (6 tabs)
```

Raw fidelity rule: **no hallucinated data.** Every canonical record keeps
its `source.url`. The LLM only structures text that was actually fetched;
schema validation rejects anything else to a dead-letter file.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENROUTER_API_KEY (the only key used)
python -m src.validate_sources   # Phase 0: prove every source works before the bulk run
```

## Layout

- `src/validate_sources.py` — Phase 0 source checks (live, no keys needed)
- `src/` — crawler, LLM orchestrator, entity resolver
- `data/raw/` — raw fetched payloads (never edited)
- `data/canonical/` — validated canonical JSONL per entity type
- `docs/` — plan + architecture (3-page `architecture.pdf`)
