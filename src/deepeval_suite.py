"""DeepEval test suite for the FrontierAtlas LLM application.

Built from the official DeepEval docs (docs.confident-ai.com /
deepeval.com): test cases are `LLMTestCase`s, metrics are DeepEval's
own classes, and the judge LLM is a custom model — our OpenRouter
gateway (gpt-4o-mini) wrapped as a `DeepEvalBaseLLM`, exactly as the
"Using Custom LLMs" guide specifies (implement `get_model_name`,
`load_model`, `generate` and `a_generate`; `generate` is schema-aware,
because metrics like Faithfulness require structured output).

Metrics (all DeepEval scores are 0–1; a case passes iff score >= threshold):
  - GEval "schema-compliance" (threshold 0.7): the output must be valid
    JSON with exactly the expected fields, correct types, no invented fields.
  - GEval "groundedness" (threshold 0.7): every fact in the output must
    be grounded in the retrieval context.
  - FaithfulnessMetric (0.7): output factually aligns with retrieval context.
  - AnswerRelevancyMetric (0.7): output is relevant to the input.
  - HallucinationMetric (0.7): output contradicts the provided context.
    (Reads `context`, not `retrieval_context`, per the docs. In
    DeepEval 4.x, 1 is a pass and `threshold` is the MINIMUM passing
    score — the old "maximum violations" reading is deprecated.)
  - ContextualRecallMetric (0.7): retrieval context suffices for the
    expected output.
  - ExtractionFieldF1Metric (custom, deterministic, 0.8): exact-match
    field F1 between actual and expected JSON — the schema-level truth.

Test cases come from the golden set (src/eval_extraction.build_golden):
15 real cases (5 startups, 5 papers, 5 products) built from the
canonical records. `actual_output` is produced live by the
LLMOrchestrator through the LiteLLM gateway; `expected_output` is the
canonical JSON; `retrieval_context`/`context` is the source text the
LLM saw.

This is the same measurement `deepeval.evaluate(test_cases, metrics)`
performs; the runner measures each metric per case directly so the
suite runs offline without a Confident AI login, and writes the full
report to data/deepeval_report.json.

Usage:
  OPENROUTER_API_KEY=... .venv/bin/python -m src.deepeval_suite
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

from .common import ROOT
from .eval_extraction import build_golden, score
from .llm_gateway import OpenRouterGateway
from .llm_orchestrator import LLMOrchestrator


class OpenRouterJudge(DeepEvalBaseLLM):
    """DeepEval judge LLM backed by the project's LiteLLM gateway."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self._model = model
        self._gateway: OpenRouterGateway | None = None

    def get_model_name(self) -> str:
        return f"{self._model} (OpenRouter)"

    def load_model(self):
        if self._gateway is None:
            self._gateway = OpenRouterGateway()
        return self._gateway

    def _call(self, prompt: str, json_mode: bool) -> str:
        gateway = self.load_model()
        text, _ = gateway.complete_sync(
            [{"role": "user", "content": prompt}], model="gpt-4o-mini",
            json_mode=json_mode)
        return text

    async def _acall(self, prompt: str, json_mode: bool) -> str:
        gateway = self.load_model()
        text, _ = await gateway.complete(
            [{"role": "user", "content": prompt}], model="gpt-4o-mini",
            json_mode=json_mode)
        return text

    def generate(self, prompt: str, schema=None) -> str:
        text = self._call(prompt, json_mode=schema is not None)
        if schema is None:
            return text
        return _to_schema(text, schema, lambda p: self._call(p, json_mode=True))

    async def a_generate(self, prompt: str, schema=None) -> str:
        text = await self._acall(prompt, json_mode=schema is not None)
        if schema is None:
            return text
        return _to_schema(text, schema, lambda p: self._call(p, json_mode=True))


def _to_schema(text: str, schema, repair_call):
    """Coerce model text into an instance of the DeepEval schema."""
    def try_parse(t: str):
        m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
        data = json.loads(m.group(1)) if m else json.loads(t)
        fields = set(schema.model_fields.keys())
        return schema(**{k: v for k, v in data.items() if k in fields})
    try:
        return try_parse(text)
    except Exception:  # noqa: BLE001 - one repair attempt, then raise
        fixed = repair_call(
            "Your previous output was not valid JSON for the required schema. "
            f"Return valid JSON only with exactly these fields: {sorted(schema.model_fields.keys())}.\n\n"
            f"Previous output:\n{text[:2000]}")
        return try_parse(fixed)


class ExtractionFieldF1Metric(BaseMetric):
    """Deterministic custom DeepEval metric: exact-match field F1."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.score: float | None = None
        self.success: bool = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        expected = json.loads(test_case.expected_output)
        actual = json.loads(test_case.actual_output)
        result = score(expected, actual)
        self.score = result["field_f1"]
        self.success = self.score >= self.threshold
        self.reason = (f"fields correct: {result.get('fields_correct')}/"
                       f"{result.get('fields_total')}")
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Extraction Field F1"


def build_test_cases(live_outputs: list[dict]) -> list[LLMTestCase]:
    cases = build_golden()
    out = []
    for case, produced in zip(cases, live_outputs):
        expected = json.dumps(case["expected"], ensure_ascii=False)
        actual = json.dumps(produced, ensure_ascii=False)
        out.append(LLMTestCase(
            input=f"{case['instruction']}\n\n---\n\n{case['text']}",
            actual_output=actual,
            expected_output=expected,
            retrieval_context=[case["text"]],
            context=[case["text"]],
        ))
    return out


def build_metrics(judge: DeepEvalBaseLLM) -> list:
    return [
        GEval(name="schema-compliance",
              criteria="The actual output must be valid JSON containing exactly "
                       "the fields of the expected output, with the correct types "
                       "and no extra or invented fields.",
              evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT,
                                 SingleTurnParams.EXPECTED_OUTPUT],
              model=judge, threshold=0.7),
        GEval(name="groundedness",
              criteria="Every fact (name, number, date) in the actual output must "
                       "appear in the retrieval context. Penalise any invented or "
                       "unsupported fact.",
              evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT,
                                 SingleTurnParams.RETRIEVAL_CONTEXT],
              model=judge, threshold=0.7),
        FaithfulnessMetric(threshold=0.7, model=judge),
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        HallucinationMetric(threshold=0.7, model=judge),
        ContextualRecallMetric(threshold=0.7, model=judge),
        ExtractionFieldF1Metric(threshold=0.8),
    ]


async def main_async() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set")
        return 2
    t0 = time.time()
    # 1. Produce live outputs with the real extraction engine.
    orch = LLMOrchestrator()
    cases = build_golden()
    live_outputs = []
    for c in cases:
        result, _ = await orch.extract(c["text"], c["instruction"])
        live_outputs.append(result if isinstance(result, dict) else {"items": result})
    test_cases = build_test_cases(live_outputs)
    # 2. Measure every official metric on every case.
    judge = OpenRouterJudge()
    metrics = build_metrics(judge)
    sem = asyncio.Semaphore(5)

    async def run_one(metric, tc):
        async with sem:
            try:
                s = await metric.a_measure(tc)
                return {"metric": metric.__name__, "score": s,
                        "success": metric.is_successful()}
            except Exception as e:  # noqa: BLE001 - record, don't crash the suite
                return {"metric": metric.__name__, "score": None,
                        "success": False, "error": str(e)[:200]}

    results = []
    for tc in test_cases:
        per_case = await asyncio.gather(*[run_one(m, tc) for m in metrics])
        results.append(per_case)
    # 3. Aggregate per metric (DeepEval semantics: score 0–1, pass >= threshold).
    summary: dict = {}
    for i, metric in enumerate(metrics):
        scores = [r[i]["score"] for r in results if r[i]["score"] is not None]
        summary[metric.__name__] = {
            "threshold": metric.threshold,
            "cases_measured": len(scores),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "pass_rate": round(sum(1 for r in results if r[i]["success"]) / len(results), 3),
            "errors": [r[i].get("error") for r in results if r[i].get("error")],
        }
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "judge_model": judge.get_model_name(),
              "cases": len(test_cases),
              "seconds": round(time.time() - t0),
              "metrics": summary}
    out = ROOT / "data" / "deepeval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
