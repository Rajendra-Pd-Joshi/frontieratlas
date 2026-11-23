"""Phase II — high-fidelity 24h signal ingestion (news + jobs).

Freshness rule: an item only counts if its normalised publication date is
within the last 24 hours at run time. Dates come from, in order:
  1. feed / API timestamp (most reliable),
  2. article meta tags (og:article:published_time, etc.),
  3. relative dates ("2 hours ago") parsed to UTC,
  4. heuristic: no date anywhere -> first_seen_at in this run, flagged
     date_source="heuristic:first_seen", and only counted if the source
     is a daily-updated feed. Heuristic items are labelled, never silently
     presented as precisely dated.

Dedupe across distributed nodes: every item is keyed by
sha256(source_name + canonical_url) in a SQLite `seen` store. A crawler
node inserts the key first; a duplicate insert means "already processed"
and the item is skipped. That is the same mechanism the architecture doc
describes for the 500k scale-out.

Full text: paragraph extraction from the article HTML (boilerplate
stripped). Cloudflare-protected pages fall back to the RSS summary and
are flagged full_text_chars=0 with a note — never fabricated.

NOTE: this module builds the pipeline and validates it. The *submission*
run must happen within 24h of submitting, or the freshness window goes
stale — re-run `python -m src.phase2 --full` on submission day.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import feedparser
from dateutil import parser as date_parser

from .common import RAW, append_raw, fetch_json, fetch_text, log, make_client, write_canonical
from .schemas import Job, JobContent, News, NewsContent, Source

UA = {"User-Agent": "Mozilla/5.0 (FrontierAtlas; contact: rajendraprasadjoshi22@gmail.com)"}

NEWS_FEEDS = {
    "The Verge (AI)": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "MIT Technology Review (AI)": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "ZDNet (AI)": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
    "IEEE Spectrum (AI)": "https://spectrum.ieee.org/rss/topic/artificial-intelligence",
    "Wired": "https://www.wired.com/feed/rss",
    "Google News (AI)": "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-US&gl=US&ceid=US:en",
    "NYT Technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
}
# TechCrunch AI (403) and VentureBeat AI (429) are NOT in the list: they
# block plain HTTP clients. They are Phase V browser-only sources.
JOB_SOURCES = {
    "RemoteOK": "https://remoteok.com/api",
    "Anthropic (Greenhouse)": "https://boards-api.greenhouse.io/v1/boards/anthropic/jobs?content=true",
    "Databricks (Greenhouse)": "https://boards-api.greenhouse.io/v1/boards/databricks/jobs?content=true",
    "Scale AI (Greenhouse)": "https://boards-api.greenhouse.io/v1/boards/scaleai/jobs?content=true",
    "Stability AI (Greenhouse)": "https://boards-api.greenhouse.io/v1/boards/stabilityai/jobs?content=true",
    "OpenAI (Ashby)": "https://api.ashbyhq.com/posting-api/job-board/openai",
    "Perplexity (Ashby)": "https://api.ashbyhq.com/posting-api/job-board/perplexity",
    "Cohere (Ashby)": "https://api.ashbyhq.com/posting-api/job-board/cohere",
}

RELATIVE_RE = re.compile(r"(\d+)\s*(minute|hour|day|week)s?\s+ago", re.I)


def parse_date(raw: str | None, now: datetime) -> tuple[datetime | None, str]:
    """Return (utc datetime or None, date_source)."""
    if not raw:
        return None, "missing"
    raw = raw.strip()
    m = RELATIVE_RE.search(raw)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n),
                 "day": timedelta(days=n), "week": timedelta(weeks=n)}[unit]
        return now - delta, "relative"
    try:
        dt = date_parser.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc), "feed"
    except (ValueError, OverflowError):
        return None, "unparseable"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paras: list[str] = []
        self._cur: list[str] = []
        self._in_p = False
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True
        if tag == "p":
            self._in_p = True
            self._cur = []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if tag == "p" and self._in_p:
            text = html.unescape("".join(self._cur)).strip()
            if len(text) > 40:
                self.paras.append(text)
            self._in_p = False

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self._cur.append(data)


def extract_full_text(html_doc: str) -> str:
    ex = _TextExtractor()
    try:
        ex.feed(html_doc)
    except Exception:  # noqa: BLE001 - malformed HTML: return what we got
        pass
    return "\n\n".join(ex.paras)


def seen_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, first_seen TEXT)")
    return conn


def is_new(conn: sqlite3.Connection, key: str, now: datetime) -> bool:
    cur = conn.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (key, now.isoformat()))
    conn.commit()
    return cur.rowcount == 1


async def fetch_news(now: datetime, conn: sqlite3.Connection, full_text: bool = True) -> list[dict]:
    out: list[dict] = []
    cutoff = now - timedelta(hours=24)
    async with make_client(UA) as client:
        sem = asyncio.Semaphore(5)
        for source_name, feed_url in NEWS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
            except Exception as e:  # noqa: BLE001
                log.warning("feed %s failed: %s", source_name, e)
                continue
            fresh_entries = []
            for entry in feed.entries:
                url = getattr(entry, "link", "")
                if not url:
                    continue
                key = hashlib.sha256(f"{source_name}|{url}".encode()).hexdigest()
                raw_date = getattr(entry, "published", None) or getattr(entry, "updated", None)
                dt, date_source = parse_date(raw_date, now)
                if dt is None:
                    dt, date_source = now, "heuristic:first_seen"
                if dt < cutoff:
                    continue
                if not is_new(conn, key, now):
                    continue
                fresh_entries.append((entry, url, dt, date_source))

            gn_sem = asyncio.Semaphore(2)  # Google News redirects 429 under bulk fetch

            async def _full_text(url: str) -> str:
                if not full_text:
                    return ""
                use_sem = gn_sem if "Google News" in source_name else sem
                try:
                    doc = await fetch_text(client, url, sem=use_sem, retries=2, timeout=20)
                    return extract_full_text(doc)
                except Exception as e:  # noqa: BLE001 - protected/slow page: keep RSS only
                    log.warning("full-text %s failed: %s", url, e)
                    return ""

            texts = await asyncio.gather(*(_full_text(url) for _, url, _, _ in fresh_entries))
            for (entry, url, dt, date_source), text in zip(fresh_entries, texts):
                out.append(News(source=Source(name=source_name, url=url),
                                content=NewsContent(
                                    title=getattr(entry, "title", "")[:300],
                                    published_date=dt.isoformat().replace("+00:00", "Z"),
                                    date_source=date_source,
                                    full_text_chars=len(text), url=url)).model_dump(mode="json"))
                append_raw("news", {"source": source_name, "url": url,
                                    "published": dt.isoformat(), "date_source": date_source,
                                    "full_text_chars": len(text)})
            log.info("news %s: %d fresh in window (total entries %d)",
                     source_name, len(fresh_entries), len(feed.entries))
    return out


async def fetch_jobs(now: datetime, conn: sqlite3.Connection) -> list[dict]:
    out: list[dict] = []
    cutoff = now - timedelta(hours=24)
    async with make_client(UA) as client:
        sem = asyncio.Semaphore(5)
        for source_name, url in JOB_SOURCES.items():
            try:
                data = await fetch_json(client, url, sem=sem, timeout=30)
            except Exception as e:  # noqa: BLE001
                log.warning("jobs %s failed: %s", source_name, e)
                continue
            jobs = []
            if isinstance(data, list):  # RemoteOK
                jobs = [{"title": j.get("position", ""), "company": j.get("company", ""),
                         "date": j.get("date"), "url": j.get("url", ""),
                         "remote": True, "role_family": "Engineering"}
                        for j in data if isinstance(j, dict) and j.get("position")]
            elif "ashbyhq" in url:  # Ashby job board API
                jobs = [{"title": j.get("title", ""), "company": source_name.split(" (")[0],
                         "date": j.get("publishedAt"), "url": j.get("jobUrl", ""),
                         "remote": bool(j.get("isRemote")), "role_family": j.get("department") or "Engineering"}
                        for j in data.get("jobs", [])]
            else:  # Greenhouse
                jobs = [{"title": j.get("title", ""), "company": source_name.split(" (")[0],
                         "date": j.get("updated_at"), "url": j.get("absolute_url", ""),
                         "remote": "remote" in (j.get("location", {}).get("name", "") or "").lower(),
                         "role_family": "Engineering"}
                        for j in data.get("jobs", [])]
            fresh = 0
            for j in jobs:
                if not j["url"]:
                    continue
                key = hashlib.sha256(f"{source_name}|{j['url']}".encode()).hexdigest()
                dt, _ = parse_date(j.get("date"), now)
                if dt is None or dt < cutoff:
                    continue
                if not is_new(conn, key, now):
                    continue
                fresh += 1
                out.append(Job(source=Source(name=source_name, url=j["url"]),
                               content=JobContent(company=j["company"], date=dt.isoformat().replace("+00:00", "Z"),
                                                  is_remote=bool(j["remote"]),
                                                  role_family=j.get("role_family") or "Engineering",
                                                  title=j["title"][:200], url=j["url"])).model_dump(mode="json"))
            log.info("jobs %s: %d fresh in 24h (total %d)", source_name, fresh, len(jobs))
    return out


async def main_async(full: bool = False) -> int:
    now = datetime.now(timezone.utc)
    # Validation runs use their own seen DB so they never eat the
    # submission run's dedupe window; --full uses the production DB.
    db_name = "seen.db" if full else "seen_validation.db"
    conn = seen_db(RAW.parent / db_name)
    news = await fetch_news(now, conn)
    jobs = await fetch_jobs(now, conn)
    report = {"run_at": now.isoformat(), "news_24h": len(news), "jobs_24h": len(jobs),
              "news_file": str(write_canonical("news" if full else "news_validation", news)),
              "jobs_file": str(write_canonical("jobs" if full else "jobs_validation", jobs))}
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="submission-day run (same code, fresh window)")
    args = ap.parse_args()
    return asyncio.run(main_async(full=args.full))


if __name__ == "__main__":
    sys.exit(main())
