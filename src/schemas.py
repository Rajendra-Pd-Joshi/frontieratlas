"""Canonical schemas for FrontierAtlas entities.

Every record carries its source URL — no hallucinated data: a record only
exists if it was fetched from that URL. `collectedAt` is always UTC ISO-8601.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Source(BaseModel):
    name: str
    url: HttpUrl


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"


class Startup(BaseModel):
    schemaVersion: str = "1.0"
    recordType: str = "STARTUP"
    source: Source
    content: "StartupContent"
    collectedAt: str = Field(default_factory=utcnow)


class StartupContent(BaseModel):
    entityName: str
    employeeCount: Optional[int] = None
    website: Optional[str] = None


class Product(BaseModel):
    schemaVersion: str = "1.0"
    recordType: str = "PRODUCT"
    source: Source
    content: "ProductContent"
    collectedAt: str = Field(default_factory=utcnow)


class ProductContent(BaseModel):
    startupName: str
    pricingModel: PricingModel


class ResearchPaper(BaseModel):
    schemaVersion: str = "1.0"
    recordType: str = "RESEARCH_PAPER"
    source: Source
    content: "PaperContent"
    collectedAt: str = Field(default_factory=utcnow)


class PaperContent(BaseModel):
    title: str
    authors: list[str] = []
    paper_url: HttpUrl
    github_url: Optional[HttpUrl] = None
    github_stars: Optional[int] = None
    published_date: Optional[str] = None


class Job(BaseModel):
    schemaVersion: str = "1.0"
    recordType: str = "JOB"
    source: Source
    content: "JobContent"
    collectedAt: str = Field(default_factory=utcnow)


class JobContent(BaseModel):
    company: str
    date: Optional[str] = None
    is_remote: bool = False
    role_family: str = "Engineering"
    title: str = ""
    url: Optional[HttpUrl] = None


class News(BaseModel):
    schemaVersion: str = "1.0"
    recordType: str = "NEWS"
    source: Source
    content: "NewsContent"
    collectedAt: str = Field(default_factory=utcnow)


class NewsContent(BaseModel):
    title: str
    published_date: Optional[str] = None
    date_source: str = "feed"  # feed | meta | relative | heuristic:first_seen
    full_text_chars: int = 0
    url: HttpUrl
