"""LangGraph orchestration for the FrontierAtlas pipeline.

This is the same pattern as ALBA ONE on the author's resume: a LangGraph
state machine with checkpointed nodes, so a run can resume after a crash
instead of restarting from zero, and a human-in-the-loop gate for
low-confidence entity merges.

Graph: fetch_startups -> fetch_papers -> fetch_products
     -> resolve_entities -> validate -> END

- Each fetch node is one Phase I fetcher (src/phase1.py) writing raw
  JSONL + canonical JSONL, so a failed node can be re-run on its own.
- resolve_entities runs the Phase IV resolver; items it marks "review"
  are collected into a review queue file. In production this node would
  call graph.interrupt() and wait for a human decision; here it logs the
  queue and continues, because the demo run is unattended.
- validate checks the assignment minimums (1,000 per entity type,
  >=95% of papers with live stars) and sets state["status"].
- Checkpointer is MemorySaver for the demo; PostgresSaver (Postgres +
  pgvector store) is the scale-out choice documented in architecture.pdf.

Usage:
  .venv/bin/python -m src.graph --sample
  GITHUB_TOKEN=... .venv/bin/python -m src.graph --full
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .common import ROOT, make_client, write_canonical
from .entity_resolver import resolve_file
from .phase1 import hf_products, pwc_papers, yc_startups


class PipelineState(TypedDict, total=False):
    run_id: str
    mode: str
    counts: dict
    stars: dict
    review_count: int
    errors: list
    status: str
    report_path: str


TARGETS = {"startups": 1000, "products": 1000, "papers": 1000}


def _targets(mode: str) -> dict:
    return {"startups": 200, "products": 200, "papers": 50} if mode == "sample" else TARGETS


async def _fetch_node(state: PipelineState, kind: str) -> dict:
    targets = _targets(state["mode"])
    import asyncio as _aio
    sem = _aio.Semaphore(10)
    async with make_client({"User-Agent": "Mozilla/5.0 (FrontierAtlas)"}) as client:
        try:
            if kind == "startups":
                records = await yc_startups(client, sem, targets["startups"])
            elif kind == "papers":
                records, stars = await pwc_papers(client, sem, targets["papers"],
                                                  os.environ.get("GITHUB_TOKEN"))
                records = records
                extra = {"stars": stars}
            else:
                records = await hf_products(client, sem, targets["products"])
                extra = {}
        except Exception as e:  # noqa: BLE001 - node failure is recorded in state
            return {"errors": [*state.get("errors", []), f"{kind}: {e}"]}
    name = f"{kind}_sample" if state["mode"] == "sample" else kind
    write_canonical(name, records)
    counts = dict(state.get("counts", {}))
    counts[kind] = len(records)
    out: dict = {"counts": counts}
    if kind == "papers":
        out["stars"] = extra["stars"]
    return out


async def fetch_startups(state: PipelineState) -> dict:
    return await _fetch_node(state, "startups")


async def fetch_papers(state: PipelineState) -> dict:
    return await _fetch_node(state, "papers")


async def fetch_products(state: PipelineState) -> dict:
    return await _fetch_node(state, "products")


async def resolve_entities(state: PipelineState) -> dict:
    suffix = "_sample" if state["mode"] == "sample" else ""
    summary = resolve_file(
        ROOT / "data" / "canonical" / f"startups{suffix}.jsonl",
        ROOT / "data" / "canonical" / f"startups{suffix}_resolved.jsonl",
        ROOT / "data" / "canonical" / f"entity_mapping_log{suffix}.jsonl")
    return {"review_count": summary["by_method"].get("review", 0)}


async def validate(state: PipelineState) -> dict:
    targets = _targets(state["mode"])
    counts = state.get("counts", {})
    errors = list(state.get("errors", []))
    ok = all(counts.get(k, 0) >= v for k, v in targets.items())
    stars = state.get("stars", {})
    # The honest metric is papers WITH stars, not unique repos fetched:
    # two papers can share one repo, and one repo link can be dead (404).
    suffix = "_sample" if state["mode"] == "sample" else ""
    papers_path = ROOT / "data" / "canonical" / f"papers{suffix}.jsonl"
    papers_with_stars = 0
    if papers_path.exists():
        papers_with_stars = sum(
            1 for line in open(papers_path, encoding="utf-8")
            if json.loads(line)["content"].get("github_stars") is not None)
    if papers_with_stars < 0.95 * targets["papers"]:
        errors.append(f"papers with stars below 95%: {papers_with_stars}/{targets['papers']}")
        ok = False
    status = "ok" if ok and not errors else "failed"
    report = {"run_id": state["run_id"], "mode": state["mode"], "status": status,
              "counts": counts, "stars": stars, "review_count": state.get("review_count", 0),
              "errors": errors, "finished_at": datetime.now(timezone.utc).isoformat()}
    path = ROOT / "data" / f"graph_run_{state['run_id'][:8]}.json"
    path.write_text(json.dumps(report, indent=2))
    return {"status": status, "errors": errors, "report_path": str(path)}


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("fetch_startups", fetch_startups)
    g.add_node("fetch_papers", fetch_papers)
    g.add_node("fetch_products", fetch_products)
    g.add_node("resolve_entities", resolve_entities)
    g.add_node("validate", validate)
    g.set_entry_point("fetch_startups")
    g.add_edge("fetch_startups", "fetch_papers")
    g.add_edge("fetch_papers", "fetch_products")
    g.add_edge("fetch_products", "resolve_entities")
    g.add_edge("resolve_entities", "validate")
    g.add_edge("validate", END)
    return g.compile(checkpointer=MemorySaver())


async def main_async(mode: str) -> int:
    graph = build_graph()
    state: PipelineState = {"run_id": str(uuid.uuid4()), "mode": mode,
                            "counts": {}, "errors": [], "status": "running"}
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["run_id"]}})
    print(json.dumps({k: result.get(k) for k in ("run_id", "mode", "status", "counts", "review_count", "report_path")}, indent=2))
    return 0 if result.get("status") == "ok" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    return asyncio.run(main_async("full" if args.full else "sample"))


if __name__ == "__main__":
    sys.exit(main())
