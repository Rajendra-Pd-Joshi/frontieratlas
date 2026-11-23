"""Guardrails for LLM extraction — LangChain parsers + grounded validators.

This is the guardrail layer the assignment's "no hallucinated data"
rule rests on. It is deliberately layered, in the order output flows:

1. Input guard — the system prompt tells the model the fetched text is
   DATA, never instructions. Fetched text is scanned for instruction-
   like patterns ("ignore previous instructions", "you are now", ...);
   hits are flagged in the report, never silently executed.
2. LangChain output guard — `JsonOutputParser` (langchain-core) parses
   the raw text; anything that is not valid JSON fails here and goes to
   the repair path. This is LangChain's default structured-output guard.
3. Grounded-values guard — every scalar value in the parsed JSON must
   be grounded in the source text (case-insensitive substring after
   normalisation; numbers must appear as text). A value that is not in
   the source is an invention: it is dropped and recorded as a
   violation. If a required field is dropped, the record fails.
4. PII guard — emails / phone numbers in the output that do not appear
   in the source text are dropped (they were invented or leaked).

The validators follow the Guardrails AI pattern (a validator takes a
value, returns (fixed_value, violation_or_None)), so a Guardrails AI
`Guard` can be dropped in front of `apply_guardrails` without changing
callers; NVIDIA NeMo Guardrails would sit at the same layer for a
deployed service (input/output rails in a config), which is out of
scope for this offline demo.
"""
from __future__ import annotations

import re

from langchain_core.output_parsers import JsonOutputParser

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
_INJECTION_RES = [
    re.compile(r"ignore (all |any )?(previous|prior|above) instructions?", re.I),
    re.compile(r"you are now (a|an|the) ", re.I),
    re.compile(r"^\s*system\s*:", re.I | re.M),
    re.compile(r"disregard (the )?(system|your) (prompt|instructions?)", re.I),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def scan_input(text: str) -> list[str]:
    """Flag instruction-like patterns in fetched (data) text."""
    return [f"input-flag: possible prompt injection pattern {rx.pattern[:40]!r}"
            for rx in _INJECTION_RES if rx.search(text)]


def _is_grounded(value, source_norm: str) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return str(value) in source_norm or _norm(str(value)) in source_norm
    if isinstance(value, str):
        v = _norm(value)
        return bool(v) and v in source_norm
    if isinstance(value, list):
        return all(_is_grounded(v, source_norm) for v in value)
    return False


def apply_guardrails(parsed, source_text: str, required_fields: list[str] | None = None) -> tuple[dict | list | None, list[str]]:
    """Validate + clean one parsed LLM output. Returns (cleaned, violations).

    `cleaned` is None when the output cannot be trusted (bad JSON shape
    or a required field was invented and dropped).
    """
    violations: list[str] = []
    if not isinstance(parsed, (dict, list)):
        return None, ["output-guard: parsed output is not a JSON object/array"]
    source_norm = _norm(source_text)
    required_fields = required_fields or ([] if isinstance(parsed, list) else list(parsed.keys()))

    def clean_value(value, path: str):
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                cv = clean_value(v, f"{path}.{k}")
                if cv is _DROPPED:
                    violations.append(f"grounded-values: dropped invented value at {path}.{k}: {str(v)[:60]!r}")
                else:
                    out[k] = cv
            return out
        if isinstance(value, list):
            out = []
            for i, v in enumerate(value):
                cv = clean_value(v, f"{path}[{i}]")
                if cv is _DROPPED:
                    violations.append(f"grounded-values: dropped invented list item at {path}[{i}]: {str(v)[:60]!r}")
                else:
                    out.append(cv)
            return out
        if isinstance(value, str):
            for rx, name in ((_EMAIL_RE, "email"), (_PHONE_RE, "phone")):
                if rx.search(value) and not rx.search(source_text):
                    violations.append(f"pii-guard: dropped {name} not present in source at {path}")
                    return _DROPPED
        if not _is_grounded(value, source_norm):
            return _DROPPED
        return value

    cleaned = clean_value(parsed, "$")
    if isinstance(cleaned, dict):
        missing = [f for f in required_fields if f not in cleaned]
        if missing:
            violations.append(f"output-guard: required field(s) missing after grounding: {missing}")
            return None, violations
    return cleaned, violations


class _Dropped:
    pass


_DROPPED = _Dropped()


def parse_json_with_guardrail(raw_text: str):
    """LangChain JsonOutputParser guard: returns parsed JSON or raises."""
    return JsonOutputParser().parse(raw_text)
