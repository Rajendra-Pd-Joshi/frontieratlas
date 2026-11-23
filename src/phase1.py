"""Phase I — massive one-time bulk extraction (sample + full modes).

Sources (all validated live in src/validate_sources.py, 8/8 OK):
- Startups: Y Combinator directory via its public Algolia search-only key
  (fresh key fetched from the YC companies page on every run — the embedded
  key expires ~24h after issue). Query "AI" returns 3,382 hits.
- Papers: PapersWithCode links dataset (HF datasets-server rows API,
  pwc-archive/links-between-paper-and-code) joined to the arXiv API for
  authors + published_date, then GitHub API for live stars.
- Products: Hugging Face Spaces API (bulk, no key) + Product Hunt Atom
  feed (recent 50 only; GraphQL bulk needs a PH developer token — Tier B).

Usage:
  .venv/bin/python -m src.phase1 --sample        # 200 startups, 50 papers, 200 products
  .venv/bin/python -m src.phase1 --full          # 1000 / 1000 / 1000 (needs GITHUB_TOKEN)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timezone

import httpx

from .common import RAW, append_raw, fetch_json, fetch_text, log, make_client, write_canonical
from .schemas import (PaperContent, PricingModel, Product, ProductContent,
                      ResearchPaper, Source, Startup, StartupContent)

UA = {"User-Agent": "Mozilla/5.0 (FrontierAtlas; contact: rajendraprasadjoshi22@gmail.com)"}
YC_APP = "45BWZJ1SGC"


async def yc_startups(session, sem, target: int) -> list[dict]:
    """Paginate YC Algolia (100 hits/page) until `target` unique companies."""
    html = await fetch_text(session, "https://www.ycombinator.com/companies", sem=sem, timeout=60)
    m = re.search(r'window\.AlgoliaOpts = (\{.*?\});', html, re.S)
    if not m:
        raise RuntimeError("AlgoliaOpts not found on YC page")
    opts = json.loads(m.group(1))
    # The embedded key is base64(<real-key> + query params); Algolia accepts
    # the raw embedded value as the API key (validated live).
    headers = {"Content-Type": "application/json",
               "X-Algolia-API-Key": opts["key"], "X-Algolia-Application-Id": opts["app"]}
    seen: dict[str, dict] = {}
    page = 0
    while len(seen) < target:
        params = urllib.parse.urlencode({"query": "AI", "hitsPerPage": 100, "page": page})
        payload = {"requests": [{"indexName": "YCCompany_production", "params": params}]}
        async with sem:
            r = await session.post(
                f"https://{opts['app'].lower()}-dsn.algolia.net/1/indexes/*/queries",
                json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        hits = data["results"][0]["hits"]
        if not hits:
            break
        for h in hits:
            slug = h.get("slug") or str(h.get("objectID"))
            if slug in seen:
                continue
            seen[slug] = h
            append_raw("yc_startups", {"fetched_at": datetime.now(timezone.utc).isoformat(),
                                       "source_url": f"https://www.ycombinator.com/companies/{slug}", "hit": h})
        log.info("yc page=%d unique=%d total_hits=%s", page, len(seen), data["results"][0].get("nbHits"))
        page += 1
        if page > 40:
            break
    out = []
    for slug, h in list(seen.items())[:target]:
        out.append(Startup(source=Source(name="Y Combinator",
                                         url=f"https://www.ycombinator.com/companies/{slug}"),
                           content=StartupContent(entityName=h["name"],
                                                  employeeCount=h.get("team_size"))).model_dump(mode="json"))
    return out


async def pwc_papers(session, sem, target: int, gh_token: str | None) -> tuple[list[dict], dict]:
    """PwC links -> arXiv join -> GitHub stars. Returns (records, stars_report)."""
    seen: dict[str, dict] = {}
    offset = 0
    while len(seen) < target:
        url = ("https://datasets-server.huggingface.co/rows?dataset=pwc-archive%2F"
               f"links-between-paper-and-code&config=default&split=train&offset={offset}&length=100")
        try:
            data = await fetch_json(session, url, sem=sem, timeout=60)
        except Exception as e:  # noqa: BLE001 - datasets-server is slow; log and stop paging
            log.warning("pwc rows offset=%d failed: %s", offset, e)
            break
        rows = [r["row"] for r in data.get("rows", [])]
        if not rows:
            break
        for r in rows:
            arxiv_id = (r.get("paper_arxiv_id") or "").strip()
            repo = (r.get("repo_url") or "").strip()
            if not arxiv_id or not repo.startswith("https://github.com/"):
                continue
            # Prefer official repos; keep first repo per paper.
            if arxiv_id not in seen or r.get("is_official"):
                seen[arxiv_id] = r
        log.info("pwc offset=%d unique_papers=%d", offset, len(seen))
        offset += len(rows)
    papers = list(seen.values())[:target]
    # arXiv join (authors + published) in chunks of 50 ids.
    meta: dict[str, dict] = {}
    for i in range(0, len(papers), 50):
        ids = ",".join(p["paper_arxiv_id"] for p in papers[i:i + 50])
        xml = await fetch_text(
            session, f"http://export.arxiv.org/api/query?id_list={ids}&max_results=50",
            sem=sem, timeout=60)
        for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
            arxiv_id = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+)", entry)
            title = re.search(r"<title>(.*?)</title>", entry, re.S)
            published = re.search(r"<published>(.*?)</published>", entry)
            authors = re.findall(r"<name>([^<]+)</name>", entry)
            if arxiv_id:
                meta[arxiv_id.group(1)] = {
                    "title": (title.group(1).strip() if title else ""),
                    "published": published.group(1) if published else None,
                    "authors": authors}
    # GitHub stars — the rate-limit bottleneck. Token: 5,000/hr. No token: 60/hr.
    gh_headers = {"Accept": "application/vnd.github+json", **UA}
    if gh_token:
        gh_headers["Authorization"] = f"Bearer {gh_token}"
    stars: dict[str, int] = {}
    rate_limited = False
    gh_sem = asyncio.Semaphore(5)

    async def one_star(repo_url: str):
        nonlocal rate_limited
        if rate_limited:
            return
        # PwC repo URLs sometimes point at a subtree, e.g.
        # github.com/<owner>/<repo>/tree/main/<path> — the API needs owner/repo.
        parts = repo_url.replace("https://github.com/", "").strip("/").split("/")
        if len(parts) < 2:
            return
        owner_repo = parts[:2]
        try:
            d = await fetch_json(session, f"https://api.github.com/repos/{owner_repo[0]}/{owner_repo[1]}",
                                 headers=gh_headers, sem=gh_sem, retries=1, timeout=20)
            stars[repo_url] = d.get("stargazers_count")
        except Exception as e:  # noqa: BLE001
            if "403" in str(e) or "rate" in str(e).lower():
                rate_limited = True
                log.warning("github rate limit hit — stars incomplete (need GITHUB_TOKEN)")
            else:
                log.warning("github %s failed: %s", repo_url, e)

    await asyncio.gather(*(one_star(p["repo_url"]) for p in papers))
    out = []
    for p in papers:
        m = meta.get(p["paper_arxiv_id"], {})
        title = m.get("title") or p.get("paper_title") or ""
        out.append(ResearchPaper(
            source=Source(name="Papers with Code",
                          url=p.get("paper_url") or f"https://arxiv.org/abs/{p['paper_arxiv_id']}"),
            content=PaperContent(title=title, authors=m.get("authors", []),
                                 paper_url=f"https://arxiv.org/abs/{p['paper_arxiv_id']}",
                                 github_url=p["repo_url"],
                                 github_stars=stars.get(p["repo_url"]),
                                 published_date=m.get("published"))).model_dump(mode="json"))
        append_raw("pwc_papers", {"fetched_at": datetime.now(timezone.utc).isoformat(),
                                  "paper_arxiv_id": p["paper_arxiv_id"], "row": p,
                                  "arxiv_meta": m, "github_stars": stars.get(p["repo_url"])})
    return out, {"stars_fetched": len(stars), "stars_requested": len(papers),
                 "rate_limited": rate_limited, "token_present": bool(gh_token)}


async def hf_products(session, sem, target: int) -> list[dict]:
    """Hugging Face Spaces API, paginated via cursor (Link header)."""
    seen: dict[str, dict] = {}
    cursor: str | None = None
    while len(seen) < target:
        url = "https://huggingface.co/api/spaces?limit=100&full=false"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        async with sem:
            r = await session.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            data = r.json()
            link = r.headers.get("Link", "")
        if not data:
            break
        for s in data:
            seen.setdefault(s["id"], s)
        log.info("hf spaces unique=%d", len(seen))
        m = re.search(r'cursor=([^&>]+)>;\s*rel="next"', link)
        cursor = urllib.parse.unquote(m.group(1)) if m else None
        if not cursor:
            break
    out = []
    for sid, s in list(seen.items())[:target]:
        author = s.get("author") or sid.split("/")[0]
        out.append(Product(source=Source(name="Hugging Face Spaces",
                                        url=f"https://huggingface.co/spaces/{sid}"),
                           content=ProductContent(startupName=author,
                                                  pricingModel=PricingModel.FREE)).model_dump(mode="json"))
    return out


async def main_async(sample: bool) -> int:
    target_startups = 200 if sample else 1000
    target_papers = 50 if sample else 1000
    target_products = 200 if sample else 1000
    gh_token = os.environ.get("GITHUB_TOKEN")
    sem = asyncio.Semaphore(10)
    async with make_client(UA) as session:
        startups, papers, products = await asyncio.gather(
            yc_startups(session, sem, target_startups),
            pwc_papers(session, sem, target_papers, gh_token),
            hf_products(session, sem, target_products))
    papers_list, stars_report = papers
    paths = {
        "startups": str(write_canonical("startups_sample" if sample else "startups",
                                        startups)),
        "papers": str(write_canonical("papers_sample" if sample else "papers", papers_list)),
        "products": str(write_canonical("products_sample" if sample else "products", products)),
    }
    report = {"mode": "sample" if sample else "full",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "counts": {"startups": len(startups), "papers": len(papers_list),
                         "products": len(products),
                         "papers_with_stars": sum(1 for p in papers_list
                                                  if p["content"]["github_stars"] is not None)},
              "stars": stars_report, "files": paths}
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW.parent / ("phase1_sample_report.json" if sample else "phase1_report.json")).write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    return asyncio.run(main_async(sample=not args.full))


if __name__ == "__main__":
    sys.exit(main())
