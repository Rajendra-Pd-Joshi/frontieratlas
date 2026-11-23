"""FastAPI service wrapping the FrontierAtlas pipeline.

Endpoints:
  GET  /health          liveness
  POST /run/phase1      start a Phase I bulk run in the background
                        (body: {"mode": "sample" | "full"}); returns a run_id
  GET  /runs/{run_id}   read the run report written by the LangGraph run
  GET  /review          low-confidence entity merges awaiting human review

Long runs execute as background tasks — the HTTP call returns immediately
with a run_id; the LangGraph checkpointer is the source of truth for
progress. API keys come from the environment, never from request bodies.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI

from .common import ROOT

app = FastAPI(title="FrontierAtlas Ingestion API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run/phase1")
def run_phase1(mode: str = "sample", background: BackgroundTasks = None) -> dict:
    import subprocess
    import sys
    run_id = f"api-{mode}"
    log = ROOT / "data" / f"api_run_{mode}.log"
    with open(log, "w") as f:
        subprocess.Popen([sys.executable, "-m", "src.phase1",
                          "--full" if mode == "full" else "--sample"],
                         cwd=ROOT, stdout=f, stderr=f)
    return {"run_id": run_id, "mode": mode, "status": "started", "log": str(log)}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    path = ROOT / "data" / f"graph_run_{run_id[:8]}.json"
    if not path.exists():
        return {"run_id": run_id, "status": "not_found"}
    return json.loads(path.read_text())


@app.get("/review")
def review() -> dict:
    log_path = ROOT / "data" / "canonical" / "entity_mapping_log.jsonl"
    if not log_path.exists():
        return {"review_items": []}
    items = [json.loads(l) for l in open(log_path, encoding="utf-8")
             if json.loads(l).get("method") == "review"]
    return {"review_items": items, "count": len(items)}
