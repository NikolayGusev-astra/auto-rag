#!/usr/bin/env python3
"""Two-layer golden-set evaluation for gateway retrieval.

Layer 1 is deterministic and has no model dependency. Layer 2 is an optional
Qwen 2.5 evidence judge served through LM Studio's OpenAI-compatible API.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.coordinator import RetrievalCoordinator


DEFAULT_GOLDEN_PATH = Path(os.path.expanduser("~/wiki/eval/golden_set.jsonl"))
DEFAULT_REPORT_PATH = Path("golden_eval_report.json")
DEFAULT_K = 5
EVALUATION_SCHEMA_VERSION = "1.0"
JUDGE_PROMPT_VERSION = "qwen25-evidence-v1"


def compute_precision_at_k(expected: Sequence[str], returned: Sequence[str], k: int) -> float:
    """Return precision@k using unique relevant document identifiers."""
    if k <= 0:
        return 0.0
    expected_ids = set(expected)
    return sum(document_id in expected_ids for document_id in returned[:k]) / k


def compute_recall_at_k(expected: Sequence[str], returned: Sequence[str], k: int) -> float:
    """Return recall@k. An empty expected set has perfect recall by convention."""
    expected_ids = set(expected)
    if not expected_ids:
        return 1.0
    return len(expected_ids.intersection(returned[:k])) / len(expected_ids)


def compute_mrr(expected: Sequence[str], returned: Sequence[str], k: int) -> float:
    """Return the reciprocal rank of the first expected document in top k."""
    expected_ids = set(expected)
    for rank, document_id in enumerate(returned[:k], start=1):
        if document_id in expected_ids:
            return 1.0 / rank
    return 0.0


def _document_id(result: Mapping[str, Any] | Any) -> str:
    if isinstance(result, Mapping):
        return str(result.get("document_id", result.get("id", "")))
    return str(getattr(result, "document_id", getattr(result, "id", "")))


def compute_ndcg(expected: Sequence[str], scored_results: Sequence[Mapping[str, Any] | Any], k: int | None = None) -> float:
    """Return binary-relevance normalized discounted cumulative gain."""
    limit = len(scored_results) if k is None else max(k, 0)
    expected_ids = set(expected)
    if not expected_ids or limit == 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, result in enumerate(scored_results[:limit], start=1)
        if _document_id(result) in expected_ids
    )
    ideal_count = min(len(expected_ids), limit)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def citation_correctness(expected_ids: Sequence[str], returned_ids: Sequence[str]) -> float:
    """Return the fraction of returned citations that cite expected documents."""
    if not returned_ids:
        return 1.0 if not expected_ids else 0.0
    expected = set(expected_ids)
    return sum(document_id in expected for document_id in returned_ids) / len(returned_ids)


def source_coverage(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return the fraction of queries with evidence from each returned source."""
    if not results:
        return {}
    counts: dict[str, int] = {}
    for result in results:
        sources = result.get("returned_sources", result.get("sources", [])) or []
        for source in set(map(str, sources)):
            counts[source] = counts.get(source, 0) + 1
    return {source: count / len(results) for source, count in counts.items()}


def latency_metrics(latencies: Sequence[float]) -> dict[str, float]:
    """Return interpolated p50, p95, and p99 latency values in seconds."""
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(float(latency) for latency in latencies)

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percent
        lower, upper = math.floor(position), math.ceil(position)
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        return round(value, 6)

    return {"p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99)}


def evaluate_retrieval(golden: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], k: int = DEFAULT_K) -> dict[str, Any]:
    """Evaluate a retrieval run deterministically without invoking an LLM."""
    if len(golden) != len(results):
        raise ValueError("golden and results must have the same number of entries")
    per_query: list[dict[str, Any]] = []
    for item, result in zip(golden, results):
        expected_ids = list(item.get("expected_document_ids", []))
        returned_ids = list(result.get("returned_document_ids", []))
        per_query.append({
            "query": item.get("query", ""),
            "precision_at_k": compute_precision_at_k(expected_ids, returned_ids, k),
            "recall_at_k": compute_recall_at_k(expected_ids, returned_ids, k),
            "mrr": compute_mrr(expected_ids, returned_ids, k),
            "ndcg": compute_ndcg(expected_ids, result.get("scored_results", returned_ids), k),
            "citation_correctness": citation_correctness(expected_ids, returned_ids),
            "empty_result": not returned_ids,
        })

    count = len(per_query)
    average = lambda name: sum(row[name] for row in per_query) / count if count else 0.0
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "queries": count,
        "precision_at_k": average("precision_at_k"),
        "recall_at_k": average("recall_at_k"),
        "mrr": average("mrr"),
        "ndcg": average("ndcg"),
        "citation_correctness": average("citation_correctness"),
        "source_coverage": source_coverage(results),
        "latency_s": latency_metrics([result.get("latency_s", 0.0) for result in results]),
        "empty_query_rate": sum(not str(item.get("query", "")).strip() for item in golden) / count if count else 0.0,
        "empty_result_rate": sum(row["empty_result"] for row in per_query) / count if count else 0.0,
        "per_query": per_query,
    }


class Qwen25Judge:
    """Optional, reproducible Qwen 2.5 judge via an OpenAI-compatible endpoint."""

    def __init__(self, base_url: str | None = None, model_id: str = "qwen2.5-7b-instruct", judge_revision: str = "unknown", prompt_version: str = JUDGE_PROMPT_VERSION, evaluation_schema_version: str = EVALUATION_SCHEMA_VERSION, temperature: float = 0) -> None:
        self.base_url = (base_url or os.getenv("RAG_LM_STUDIO_URL", "http://localhost:1234/v1")).rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url.removesuffix("/chat/completions")
        self.judge_model_id = model_id
        self.judge_revision = judge_revision
        self.judge_prompt_version = prompt_version
        self.evaluation_schema_version = evaluation_schema_version
        self.temperature = temperature

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def evaluate_evidence(self, query: str, golden_answer: str | None, evidence_texts: Sequence[str]) -> dict[str, Any]:
        if not self.is_available():
            return {"llm_judge_status": "skipped"}
        prompt = self._prompt(query, golden_answer, evidence_texts)
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json={
                "model": self.judge_model_id,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            }, timeout=30)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            verdict = self._parse_verdict(content)
        except (requests.RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return {"llm_judge_status": "error", "error": str(error)}
        return {"llm_judge_status": "completed", **self.metadata(), **verdict}

    def metadata(self) -> dict[str, Any]:
        return {
            "judge_model_id": self.judge_model_id,
            "judge_revision": self.judge_revision,
            "judge_prompt_version": self.judge_prompt_version,
            "judge_prompt_revision": self.judge_prompt_version,
            "evaluation_schema_version": self.evaluation_schema_version,
            "temperature": self.temperature,
        }

    def _prompt(self, query: str, golden_answer: str | None, evidence_texts: Sequence[str]) -> str:
        evidence = "\n\n".join(evidence_texts)
        return f'''Evaluate whether the retrieved evidence supports the reference answer for this query. Return JSON only.
Schema: {{"relevance": number 0..1, "coverage": number 0..1, "groundedness": number 0..1, "conflicts": [string], "missing_aspects": [string], "pass": boolean}}
Query: {query}
Reference answer: {golden_answer or "(not provided)"}
Evidence:\n{evidence}'''

    @staticmethod
    def _parse_verdict(content: str) -> dict[str, Any]:
        value = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
        required = {"relevance", "coverage", "groundedness", "conflicts", "missing_aspects", "pass"}
        if not required.issubset(value):
            raise ValueError("judge response does not match evaluation schema")
        for field in ("relevance", "coverage", "groundedness"):
            value[field] = float(value[field])
            if not 0 <= value[field] <= 1:
                raise ValueError(f"{field} must be between 0 and 1")
        if not isinstance(value["conflicts"], list) or not isinstance(value["missing_aspects"], list) or not isinstance(value["pass"], bool):
            raise ValueError("judge response has invalid field types")
        return {field: value[field] for field in required}


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Load JSONL golden records, or the legacy ``questions`` JSON envelope."""
    with path.open(encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return []
    if raw.startswith("{"):
        data = json.loads(raw)
        return list(data.get("questions", [data]))
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


async def run_retrieval(golden: Sequence[Mapping[str, Any]], coordinator: RetrievalCoordinator, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Run every golden query through ``RetrievalCoordinator`` and normalize evidence."""
    results: list[dict[str, Any]] = []
    for item in golden:
        started = time.perf_counter()
        evidence = await coordinator.search(SearchRequest(query=str(item.get("query", "")), topk=k))
        results.append({
            "returned_document_ids": [entry.document_id for entry in evidence],
            "returned_sources": [entry.source for entry in evidence],
            "scored_results": [{"document_id": entry.document_id, "score": entry.final_score} for entry in evidence],
            "evidence_texts": [entry.text for entry in evidence],
            "latency_s": time.perf_counter() - started,
        })
    return results


def regression_check(current: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    """Gate a release when deterministic metrics or completed judge scores regress."""
    if baseline is None:
        return {"passed": True, "reason": "no baseline supplied"}
    deterministic = ("precision_at_k", "recall_at_k", "mrr", "ndcg", "citation_correctness")
    regressions = [name for name in deterministic if current.get(name, 0.0) < baseline.get(name, 0.0)]
    current_judge = current.get("llm_judge", {})
    baseline_judge = baseline.get("llm_judge", {})
    if current_judge.get("llm_judge_status") == baseline_judge.get("llm_judge_status") == "completed":
        for name in ("relevance", "coverage", "groundedness"):
            if current_judge.get(name, 0.0) < baseline_judge.get(name, 0.0):
                regressions.append(f"llm_judge.{name}")
    return {"passed": not regressions, "regressions": regressions}


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    golden = load_golden_set(Path(args.golden).expanduser())
    results = await run_retrieval(golden, RetrievalCoordinator(), args.k)
    metrics = evaluate_retrieval(golden, results, args.k)
    if args.judge:
        judge = Qwen25Judge(model_id=args.judge_model_id, judge_revision=args.judge_revision)
        verdicts = [judge.evaluate_evidence(item.get("query", ""), item.get("golden_answer"), result["evidence_texts"]) for item, result in zip(golden, results)]
        completed = [verdict for verdict in verdicts if verdict.get("llm_judge_status") == "completed"]
        metrics["llm_judge"] = ({"llm_judge_status": "skipped"} if not completed else {
            "llm_judge_status": "completed",
            **judge.metadata(),
            **{field: statistics.mean(verdict[field] for verdict in completed) for field in ("relevance", "coverage", "groundedness")},
            "pass": all(verdict["pass"] for verdict in completed),
        })
        metrics["per_query_judge"] = verdicts
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    report = {"summary": metrics, "results": results, "regression_check": regression_check(metrics, baseline)}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": metrics, "regression_check": report["regression_check"]}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate golden retrieval records")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_PATH), help="golden JSONL path")
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="report JSON path")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--judge", action="store_true", help="enable optional Qwen 2.5 judge")
    parser.add_argument("--judge-model-id", default="qwen2.5-7b-instruct")
    parser.add_argument("--judge-revision", default="unknown")
    parser.add_argument("--baseline", help="previous report used as the release-gate baseline")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
