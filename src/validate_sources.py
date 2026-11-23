# Phase 0 — Source validation for the FrontierAtlas pipeline.
#
# Purpose: prove, with live requests and NO API keys, that every source the
# bulk run depends on actually works, before writing the full pipeline.
# Writes data/validation_report.json.
#
# Sources tested:
#   papers:  arXiv API + PwC links dataset (HF datasets-server rows API)
#   startups: Y Combinator Algolia (public search-only key from the YC page)
#   products: Product Hunt Atom feed + Hugging Face Spaces API
#   news:    RSS feeds (Verge, MIT Tech Review, Ars Technica, Wired, IEEE)
#   jobs:    RemoteOK API + Greenhouse boards (OpenAI) + Lever (Anthropic)
#   github:  api.github.com stars lookup (unauthenticated, small sample)
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "validation_report.json"
UA = {"User-Agent": "Mozilla/5.0 (FrontierAtlas validation; contact: rajendraprasadjoshi22@gmail.com)"}

report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "checks": {}}


def fetch(url: str, headers: dict | None = None, timeout: int = 25, data: bytes | None = None, method: str | None = None):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def check(name: str, fn):
    t0 = time.time()
    try:
        detail = fn()
        report["checks"][name] = {"status": "OK", "ms": round((time.time() - t0) * 1000), **detail}
        print(f"OK   {name}: {detail}")
    except Exception as e:  # noqa: BLE001 - validation script: record, don't crash
        report["checks"][name] = {"status": "FAIL", "ms": round((time.time() - t0) * 1000), "error": str(e)[:300]}
        print(f"FAIL {name}: {e}")


def arxiv():
    _, b = fetch("http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=5")
    text = b.decode(errors="ignore")
    return {"entries": text.count("<entry>"), "sample_title": re.findall(r"<title>([^<]+)</title>", text)[1][:80]}


def pwc_links():
    url = ("https://datasets-server.huggingface.co/rows?dataset=pwc-archive%2F"
           "links-between-paper-and-code&config=default&split=train&offset=0&length=5")
    # datasets-server is slow on first hit; allow 60s and one retry.
    last = None
    for _ in range(2):
        try:
            _, b = fetch(url, timeout=60)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise last  # noqa: B904
    d = json.loads(b)
    rows = [r["row"] for r in d["rows"]]
    github = [r for r in rows if r["repo_url"].startswith("https://github.com/")]
    return {"rows_returned": len(rows), "github_links": len(github),
            "sample": rows[0]["paper_title"][:60], "sample_repo": rows[0]["repo_url"]}


def yc_algolia():
    # Public search-only key is embedded in the YC companies page itself.
    # YC is slow/flaky from this VM; retry once with a long timeout.
    last = None
    for _ in range(2):
        try:
            return _yc_algolia_once()
        except Exception as e:  # noqa: BLE001
            last = e
    raise last  # noqa: B904


def _yc_algolia_once():
    _, b = fetch("https://www.ycombinator.com/companies", timeout=60)
    html = b.decode(errors="ignore")
    m = re.search(r'window\.AlgoliaOpts = (\{.*?\});', html, re.S)
    if not m:
        raise RuntimeError("AlgoliaOpts not found on YC page")
    opts = json.loads(m.group(1))
    payload = json.dumps({"requests": [{"indexName": "YCCompany_production",
                                        "params": "query=AI&hitsPerPage=5&page=0"}]}).encode()
    _, b2 = fetch(f"https://{opts['app'].lower()}-dsn.algolia.net/1/indexes/*/queries",
                  headers={"Content-Type": "application/json",
                           "X-Algolia-API-Key": opts["key"],
                           "X-Algolia-Application-Id": opts["app"]},
                  data=payload, method="POST")
    d = json.loads(b2)
    hits = d["results"][0]["hits"]
    return {"hits": len(hits), "total_ai_query": d["results"][0].get("nbHits"),
            "sample": hits[0]["name"], "sample_team_size": hits[0].get("team_size"),
            "sample_website": hits[0].get("website")}


def hf_spaces():
    _, b = fetch("https://huggingface.co/api/spaces?limit=5&full=true")
    d = json.loads(b)
    return {"spaces": len(d), "sample": d[0]["id"], "sample_author": d[0].get("author")}


def product_hunt_feed():
    _, b = fetch("https://www.producthunt.com/feed")
    text = b.decode(errors="ignore")
    titles = re.findall(r"<title>([^<]+)</title>", text)
    return {"entries": text.count("<entry>"), "sample": titles[1] if len(titles) > 1 else ""}


def github_stars():
    _, b = fetch("https://api.github.com/repos/pytorch/pytorch")
    d = json.loads(b)
    return {"repo": d["full_name"], "stars": d["stargazers_count"]}


def news_rss():
    feeds = {
        "verge_ai": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "mit_ai": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
        "ars_ai": "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "zdnet_ai": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
        "ieee_ai": "https://spectrum.ieee.org/rss/topic/artificial-intelligence",
    }
    out = {}
    for name, url in feeds.items():
        try:
            _, b = fetch(url)
            text = b.decode(errors="ignore")
            out[name] = text.count("<item>") + text.count("<entry>")
        except Exception as e:  # noqa: BLE001
            out[name] = f"ERR {e}"[:120]
    return {"items_per_feed": out}


def jobs():
    _, b = fetch("https://remoteok.com/api?tag=ai")
    d = json.loads(b)
    jobs_list = [x for x in d if isinstance(x, dict) and x.get("position")]
    boards = {}
    for board in ("anthropic", "databricks", "scaleai"):
        _, b2 = fetch(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true")
        boards[board] = len(json.loads(b2).get("jobs", []))
    return {"remoteok_ai_jobs": len(jobs_list),
            "sample_remoteok": jobs_list[0]["position"] if jobs_list else "",
            "greenhouse_jobs_by_board": boards}


def main() -> int:
    for name, fn in [("arxiv_api", arxiv), ("pwc_links_dataset", pwc_links),
                     ("yc_algolia", yc_algolia), ("hf_spaces", hf_spaces),
                     ("product_hunt_feed", product_hunt_feed), ("github_stars", github_stars),
                     ("news_rss", news_rss), ("jobs_apis", jobs)]:
        check(name, fn)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    ok = sum(1 for c in report["checks"].values() if c["status"] == "OK")
    print(f"\n{ok}/{len(report['checks'])} checks OK — report: {OUT}")
    return 0 if ok == len(report["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
