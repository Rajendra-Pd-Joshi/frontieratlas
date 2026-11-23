"""Export the 6 submission tabs as CSVs (Google Sheets upload when creds exist).

Tabs: Startups, Products, Research Papers, Jobs, News, Entity Mapping Log.
Source of truth is the canonical JSONL files; Jobs/News come from the
latest Phase II run (validation files until the submission-day --full run).

Usage:
  .venv/bin/python -m src.export_sheets
  # With a Google service-account JSON in GOOGLE_SHEETS_CREDENTIALS and
  # gspread installed, also uploads to a new spreadsheet and prints its URL:
  GOOGLE_SHEETS_CREDENTIALS=/path/to/sa.json .venv/bin/python -m src.export_sheets --upload
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from .common import ROOT

SHEETS_DIR = ROOT / "data" / "sheets"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _write_csv(name: str, fields: list[str], rows: list[dict]) -> Path:
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    out = SHEETS_DIR / f"{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def build() -> dict[str, Path]:
    out: dict[str, Path] = {}
    startups = _read_jsonl(ROOT / "data" / "canonical" / "startups.jsonl")
    out["Startups"] = _write_csv("startups",
        ["entityName", "employeeCount", "website", "source_url", "collectedAt"],
        [{"entityName": r["content"]["entityName"],
          "employeeCount": r["content"].get("employeeCount"),
          "website": r["content"].get("website", ""),
          "source_url": r["source"]["url"], "collectedAt": r["collectedAt"]} for r in startups])
    products = _read_jsonl(ROOT / "data" / "canonical" / "products.jsonl")
    out["Products"] = _write_csv("products",
        ["startupName", "pricingModel", "source_url", "collectedAt"],
        [{"startupName": r["content"]["startupName"], "pricingModel": r["content"]["pricingModel"],
          "source_url": r["source"]["url"], "collectedAt": r["collectedAt"]} for r in products])
    papers = _read_jsonl(ROOT / "data" / "canonical" / "papers.jsonl")
    out["Research Papers"] = _write_csv("papers",
        ["title", "authors", "paper_url", "github_url", "github_stars", "published_date", "collectedAt"],
        [{"title": r["content"]["title"], "authors": "; ".join(r["content"].get("authors", [])),
          "paper_url": r["content"].get("paper_url", ""), "github_url": r["content"].get("github_url", ""),
          "github_stars": r["content"].get("github_stars"),
          "published_date": r["content"].get("published_date"),
          "collectedAt": r["collectedAt"]} for r in papers])
    jobs = _read_jsonl(ROOT / "data" / "canonical" / "jobs.jsonl") or \
        _read_jsonl(ROOT / "data" / "canonical" / "jobs_validation.jsonl")
    out["Jobs"] = _write_csv("jobs",
        ["company", "role", "date", "is_remote", "role_family", "source_url", "collectedAt"],
        [{"company": r["content"]["company"], "role": r["content"].get("title", ""),
          "date": r["content"].get("date", ""), "is_remote": r["content"].get("is_remote"),
          "role_family": r["content"].get("role_family", ""),
          "source_url": r["source"]["url"], "collectedAt": r["collectedAt"]} for r in jobs])
    news = _read_jsonl(ROOT / "data" / "canonical" / "news.jsonl") or \
        _read_jsonl(ROOT / "data" / "canonical" / "news_validation.jsonl")
    out["News"] = _write_csv("news",
        ["title", "source", "date", "date_source", "url", "full_text_chars", "collectedAt"],
        [{"title": r["content"]["title"], "source": r["source"]["name"],
          "date": r["content"].get("published_date", ""), "date_source": r["content"].get("date_source", ""),
          "url": r["source"]["url"],
          "full_text_chars": r["content"].get("full_text_chars", 0),
          "collectedAt": r["collectedAt"]} for r in news])
    mapping = _read_jsonl(ROOT / "data" / "canonical" / "entity_mapping_log.jsonl")
    out["Entity Mapping Log"] = _write_csv("entity_mapping_log",
        ["raw", "canonical", "method", "confidence"],
        [{"raw": r["raw"], "canonical": r["canonical"], "method": r["method"],
          "confidence": r.get("confidence")} for r in mapping])
    return out


def upload(files: dict[str, Path]) -> str:
    import gspread
    creds = os.environ["GOOGLE_SHEETS_CREDENTIALS"]
    gc = gspread.service_account(filename=creds)
    sh = gc.create("FrontierAtlas — Intelligence Graph (Demo)")
    for tab, path in files.items():
        ws = sh.add_worksheet(title=tab, rows=100, cols=20) if tab != "Startups" else sh.sheet1
        ws.update_title(tab)
        with open(path, newline="", encoding="utf-8") as f:
            ws.update([next(csv.reader(f))] + list(csv.reader(f)))
    sh.share(None, perm_type="anyone", role="reader")
    return sh.url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    files = build()
    for tab, path in files.items():
        print(f"{tab}: {path} ({sum(1 for _ in open(path)) - 1} rows)")
    if args.upload:
        print("sheet url:", upload(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
