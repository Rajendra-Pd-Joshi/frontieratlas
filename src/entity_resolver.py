"""Phase IV — deterministic entity resolution.

Order of operations (deterministic first, fuzzy/LLM last):
1. Normalise: lowercase, strip legal suffixes, strip punctuation, collapse
   whitespace. "OpenAI, Inc." and "Open AI" both normalise to "openai".
2. Alias table: hand-curated raw -> canonical for known variants.
3. Seed list: 50 known AI startups/companies with their canonical form.
   A normalised name matching a seed entry (or its domain) resolves with
   confidence 1.0 and method "seed".
4. Fuzzy: difflib ratio >= 0.90 against seed names -> "fuzzy" with the
   ratio as confidence; 0.80-0.90 -> "review" (never auto-merge).
5. No match: the entity is new. Canonical form is the cleaned original
   (title-cased), confidence 0.5, method "new" — it is NOT forced onto a
   seed entry. Forcing a match is how hallucinated merges happen.

Every decision is written to the Entity Mapping Log (raw, canonical,
method, confidence) — that log is a deliverable tab in the Sheet.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

LEGAL_SUFFIXES = {
    "inc", "llc", "ltd", "corp", "corporation", "company", "co", "plc",
    "gmbh", "pty", "pbc",
}
# Deliberately NOT stripped: "ai", "labs", "technologies" — they are part
# of AI company names ("Mistral AI", "AI21 Labs"). Stripping them would
# collapse "Open AI" to "open" and break alias/seed matching.

SEED_CANONICAL = [
    "OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Microsoft",
    "NVIDIA", "Hugging Face", "Stability AI", "Midjourney", "Cohere",
    "AI21 Labs", "Mistral AI", "Databricks", "Scale AI", "Runway",
    "Perplexity", "xAI", "Inflection", "Character.AI", "Replit",
    "Jasper", "Copy.ai", "Notion", "Canva", "Adobe", "Salesforce",
    "Oracle", "IBM", "Intel", "AMD", "Palantir", "Snowflake",
    "MongoDB", "Elastic", "Datadog", "GitHub", "GitLab", "Figma",
    "Zoom", "Slack", "Shopify", "Stripe", "Airbnb", "Uber",
    "ElevenLabs", "Synthesia", "Writer", "Glean", "Hebbia", "DeepL",
]  # exactly 50 entries

ALIASES = {
    "open ai": "OpenAI",
    "anthropic pbc": "Anthropic",
    "huggingface": "Hugging Face",
    "hugging face inc": "Hugging Face",
    "google deepmind technologies": "Google DeepMind",
    "deepmind": "Google DeepMind",
    "stability": "Stability AI",
    "mistral": "Mistral AI",
    "character ai": "Character.AI",
    "characterai": "Character.AI",
    "copy ai": "Copy.ai",
    "eleven labs": "ElevenLabs",
    "scale": "Scale AI",
    "x ai": "xAI",
    "jasperai": "Jasper",
    "meta": "Meta AI",
    "facebook ai": "Meta AI",
}


def normalize(name: str) -> str:
    """Lowercase, drop punctuation, drop trailing legal/industry suffixes."""
    s = name.lower().strip()
    s = re.sub(r"[.&'’]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    words = s.split()
    while words and words[-1] in LEGAL_SUFFIXES and len(words) > 1:
        words.pop()
    return " ".join(words)


def _seed_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for canonical in SEED_CANONICAL:
        idx[normalize(canonical)] = canonical
        idx[canonical.lower().replace(" ", "")] = canonical
    return idx


SEED_INDEX = _seed_index()


def resolve(name: str) -> dict:
    """Resolve one raw name -> {raw, canonical, method, confidence}."""
    raw = (name or "").strip()
    if not raw:
        return {"raw": raw, "canonical": "", "method": "empty", "confidence": 0.0}
    norm = normalize(raw)
    if norm in ALIASES:
        return {"raw": raw, "canonical": ALIASES[norm], "method": "alias", "confidence": 1.0}
    if norm in SEED_INDEX:
        return {"raw": raw, "canonical": SEED_INDEX[norm], "method": "seed", "confidence": 1.0}
    squashed = norm.replace(" ", "")
    if squashed in SEED_INDEX:
        return {"raw": raw, "canonical": SEED_INDEX[squashed], "method": "seed", "confidence": 1.0}
    # Fuzzy against seed canonical names (normalised form).
    best, best_ratio = None, 0.0
    for seed_norm, canonical in SEED_INDEX.items():
        ratio = difflib.SequenceMatcher(None, norm, seed_norm).ratio()
        if ratio > best_ratio:
            best, best_ratio = canonical, ratio
    if best_ratio >= 0.90:
        return {"raw": raw, "canonical": best, "method": "fuzzy", "confidence": round(best_ratio, 3)}
    if best_ratio >= 0.80:
        return {"raw": raw, "canonical": best, "method": "review",
                "confidence": round(best_ratio, 3),
                "note": "below auto-merge threshold — human review required"}
    # New entity: keep the name as given (cleaned only of surrounding
    # punctuation) — do not force it onto a seed entry, and do not strip
    # parts of its real name.
    return {"raw": raw, "canonical": raw, "method": "new", "confidence": 0.5}


def resolve_file(input_path: Path, output_path: Path, log_path: Path) -> dict:
    records = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]
    resolved, mapping_log = [], []
    for rec in records:
        name = rec.get("content", {}).get("entityName") or rec.get("content", {}).get("startupName", "")
        r = resolve(name)
        mapping_log.append(r)
        rec = dict(rec)
        rec["content"] = dict(rec["content"])
        rec["content"]["entityName"] = r["canonical"]
        rec["content"]["_resolution"] = {"method": r["method"], "confidence": r["confidence"]}
        resolved.append(rec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in resolved:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(log_path, "w", encoding="utf-8") as f:
        for row in mapping_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"records": len(records),
               "by_method": {m: sum(1 for r in mapping_log if r["method"] == m)
                             for m in sorted({r["method"] for r in mapping_log})}}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    print(json.dumps(resolve_file(Path(args.input), Path(args.output), Path(args.log)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
