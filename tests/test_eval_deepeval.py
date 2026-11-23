"""Pytest wrapper for the DeepEval suite (src/deepeval_suite.py).

- Deterministic checks run offline (custom field-F1 metric, report shape).
- The full live suite (real gateway + DeepEval judge) only runs when
  RUN_LIVE_EVAL=1 and OPENROUTER_API_KEY is set; it takes ~2 minutes.

Run: .venv/bin/python -m pytest tests/test_eval_deepeval.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepeval.test_case import LLMTestCase  # noqa: E402

from src.deepeval_suite import ExtractionFieldF1Metric, build_test_cases  # noqa: E402
from src.eval_extraction import build_golden  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_custom_field_f1_metric():
    metric = ExtractionFieldF1Metric(threshold=0.8)
    cases = build_golden()
    # Perfect output scores 1.0.
    tc = LLMTestCase(input="x", actual_output=json.dumps(cases[0]["expected"]),
                     expected_output=json.dumps(cases[0]["expected"]),
                     retrieval_context=[cases[0]["text"]])
    assert metric.measure(tc) == 1.0
    assert metric.is_successful()
    # Corrupted output scores below threshold.
    wrong = dict(cases[0]["expected"])
    first_key = next(iter(wrong))
    wrong[first_key] = "definitely not in the source text 12345"
    tc2 = LLMTestCase(input="x", actual_output=json.dumps(wrong),
                      expected_output=json.dumps(cases[0]["expected"]),
                      retrieval_context=[cases[0]["text"]])
    assert metric.measure(tc2) < 1.0
    print("custom field-F1 metric: PASS")


def test_deepeval_report_shape():
    report_path = ROOT / "data" / "deepeval_report.json"
    assert report_path.exists(), "run: OPENROUTER_API_KEY=... .venv/bin/python -m src.deepeval_suite"
    report = json.loads(report_path.read_text())
    assert report["cases"] == 15
    assert report["judge_model"].startswith("openai/gpt-4o-mini")
    expected_metrics = {"schema-compliance [GEval]", "groundedness [GEval]",
                        "Faithfulness", "Answer Relevancy", "Hallucination",
                        "Contextual Recall", "Extraction Field F1"}
    assert expected_metrics <= set(report["metrics"]), set(report["metrics"])
    for name, m in report["metrics"].items():
        assert m["cases_measured"] == 15, name
        assert m["mean_score"] is not None, name
        assert not m["errors"], (name, m["errors"])
    assert report["metrics"]["Extraction Field F1"]["mean_score"] >= 0.9
    print("deepeval report shape: PASS")


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("RUN_LIVE_EVAL") != "1",
                    reason="live suite needs OPENROUTER_API_KEY and RUN_LIVE_EVAL=1")
def test_live_deepeval_suite():
    import asyncio
    from src.deepeval_suite import main_async
    assert asyncio.run(main_async()) == 0
    report = json.loads((ROOT / "data" / "deepeval_report.json").read_text())
    assert report["metrics"]["Extraction Field F1"]["mean_score"] >= 0.9
    print("live deepeval suite: PASS")
