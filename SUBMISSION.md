# Submission — GraphOne / FrontierAtlas AI Engineer Demo Task

**Candidate:** Rajendra Prasad Joshi
**Date:** 23 November 2026

## Deliverables

| Deliverable | Link / location |
|---|---|
| Public Google Sheet — 6 tabs | https://docs.google.com/spreadsheets/d/1Ill1fRX481TuNFUl1Tqw9o7CBlYgPOq5B8zY6QldRug/edit?usp=sharing |
| GitHub repository | https://github.com/Rajendra-Pd-Joshi/frontieratlas |
| Architecture document (3 pages) | `architecture.pdf` (also `docs/architecture.md`) |

### Google Sheet contents

| Tab | Rows | Notes |
|---|---|---|
| Startups | 1,000 | Y Combinator directory; 979 with employeeCount, 977 with website |
| Products | 1,000 | Hugging Face Spaces (public, free-to-use — pricingModel FREE across this set) |
| Research Papers | 1,000 | arXiv / PapersWithCode data; 990 with live GitHub stars |
| Jobs | 10 | All AI jobs found in the 24h window (Greenhouse / Ashby / RemoteOK) |
| News | 80 | All AI news found in the 24h window (8 RSS feeds) |
| Entity Mapping Log | 1,000 | Seed-list resolution: 3 seed, 1 alias, 7 review, 989 new |

## What was built

A production-grade ingestion pipeline for the Intelligence Graph:
LangGraph state machine (fetch → resolve → validate) with a
checkpointed FastAPI service, an OpenRouter-only LLM tier behind a
LiteLLM Router gateway (gpt-4o-mini → gpt-oss-120b → gpt-oss-20b →
gpt-6-luna, with fallbacks, retries and circuit breakers), layered
guardrails (LangChain JSON parser, grounded-values, PII, and
input-injection flagging), and deterministic-first entity resolution
against a 50-startup seed list.

## Verification (all run live, 23 Nov 2026)

- **Sources:** 8/8 validated before the bulk run
  (`python -m src.validate_sources`, `data/validation_report.json`).
  PapersWithCode's API and paper pages are dead — the pipeline uses
  its Hugging Face datasets mirror instead, documented in the code.
- **Extraction eval:** 15 golden cases built from the real canonical
  records — 1.0 valid-JSON, 1.0 mean field F1, all via gpt-4o-mini.
- **Official DeepEval 4.3.4 suite** (judge: gpt-4o-mini via OpenRouter),
  15 cases: GEval schema-compliance 1.00, GEval groundedness 1.00,
  Faithfulness 0.93, Answer Relevancy 1.00, Hallucination 1.00,
  Contextual Recall 1.00, custom Extraction Field F1 1.00.
- **Orchestrator tests:** 429 fallback, chunking, 413 shrink-and-retry,
  guardrails — all pass (`python -m tests.test_llm_orchestrator`).

## Data integrity

No hallucinated data: every canonical record keeps its `source.url`,
and the LLM only structures text that was actually fetched. Gaps are
recorded as-is, never filled in: 10 paper repo links are genuinely
dead (404) and are recorded with no stars; 23 startups have no
website in the source data; news full-text is 0 where the fetch was
blocked (429/403). Jobs and News hold the 24h window as of the run
date; the pipeline re-runs them with `python -m src.phase2 --full`
for a fresh window.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENROUTER_API_KEY (the only key used)
python -m src.validate_sources   # Phase 0: prove every source works
python -m src.phase1             # Phase I: 1,000 / 1,000 / 1,000
python -m src.phase2 --full      # Phase II: fresh 24h news + jobs
python -m src.export_sheets      # 6-tab CSV export
python -m src.deepeval_suite     # official DeepEval metrics
python docs/build_architecture_pdf.py   # rebuild architecture.pdf
```
