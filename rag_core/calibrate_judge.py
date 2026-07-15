#!/usr/bin/env python3
"""
Calibrate LLM-as-Judge against human-verified golden set.

Usage:
  python3 calibrate_judge.py                        # full calibration
  python3 calibrate_judge.py --quick                 # 10 questions sanity check
  python3 calibrate_judge.py --judge-model qwen2.5-7b-instruct

Output:
  - judge_calibration_report.json — confusion matrix + per-question scores
  - Recommends score thresholds for verdicts (correct/partial/incorrect)
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent.parent / "rag_core"
CALIBRATION_PATH = HERE / "judge_calibration_set.json"
REPORT_PATH = HERE / "judge_calibration_report.json"
LLM_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_JUDGE = "qwen2.5-7b-instruct"

# Default calibration set with human-verified scores (0.0–1.0)
DEFAULT_SET = {
    "meta": {
        "description": "LLM Judge calibration set — human-verified scores",
        "version": "1.0",
        "n_questions": 12
    },
    "questions": [
        {
            "id": "rag-intro-1",
            "query": "What is RAG and how does retrieval-augmented generation work?",
            "context": "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval from a knowledge base with text generation. It works by first retrieving relevant documents from a vector database, then feeding them as context to an LLM for answer generation. This grounds the model's output in factual data rather than relying solely on parametric knowledge.",
            "key_facts": ["retrieval from knowledge base", "vector database", "LLM context", "grounding"],
            "human_score": 0.95,
            "verdict": "correct"
        },
        {
            "id": "rag-intro-2",
            "query": "What is RAG and how does retrieval-augmented generation work?",
            "context": "RAG is a popular framework for building AI applications. Many companies use it for customer support and internal knowledge management.",
            "key_facts": ["retrieval from knowledge base", "vector database"],
            "human_score": 0.3,
            "verdict": "weak"
        },
        {
            "id": "chunking-1",
            "query": "What chunking strategies are used in RAG pipelines?",
            "context": "Multiple chunking strategies have been proposed for RAG pipelines. Fixed-size chunking splits documents into chunks of a predefined size. Recursive chunking uses a hierarchy of delimiters to split at paragraph and sentence boundaries. Semantic chunking uses LLM-detected topic boundaries.",
            "key_facts": ["fixed-size chunking", "recursive chunking", "semantic chunking", "paragraph boundaries"],
            "human_score": 0.95,
            "verdict": "correct"
        },
        {
            "id": "chunking-2",
            "query": "What chunking strategies are used in RAG pipelines?",
            "context": "Document processing pipelines often split text into smaller pieces for processing. Different approaches exist for different document types.",
            "key_facts": ["fixed-size chunking", "recursive chunking"],
            "human_score": 0.2,
            "verdict": "incorrect"
        },
        {
            "id": "dcd-1",
            "query": "How does DCD routing improve RAG accuracy?",
            "context": "DCD (Domain-Collection-Document) routing classifies queries into predefined domains before retrieval. This constraint reduces search space and improves precision, especially for small models that struggle with broad corpus retrieval.",
            "key_facts": ["domain classification", "reduced search space", "improved precision"],
            "human_score": 0.9,
            "verdict": "correct"
        },
        {
            "id": "dcd-2",
            "query": "How does DCD routing improve RAG accuracy?",
            "context": "Search engines use query classification to route requests to appropriate backends. This technique is common in information retrieval systems.",
            "key_facts": ["domain classification", "reduced search space"],
            "human_score": 0.4,
            "verdict": "partial"
        },
        {
            "id": "canary-1",
            "query": "What is canary deployment in ML systems?",
            "context": "Canary release is a technique to reduce the risk of introducing a new software version in production by slowly rolling out the change to a small subset of users before full deployment. The candidate model's performance is measured against the existing model's metrics. If key metrics degrade significantly, the canary is aborted.",
            "key_facts": ["slow rollout", "subset of users", "performance measurement", "abort on degradation"],
            "human_score": 0.95,
            "verdict": "correct"
        },
        {
            "id": "eval-1",
            "query": "How should RAG systems be evaluated in production?",
            "context": "A mature RAG evaluation framework must enable two fundamental tasks: measurement and tuning. Measurement is the systematic monitoring of application quality. LLM-as-a-Judge must be calibrated against a human-verified golden test set to avoid measuring self-reflection instead of accuracy.",
            "key_facts": ["measurement", "tuning", "LLM-as-judge", "golden test set", "calibration"],
            "human_score": 0.9,
            "verdict": "correct"
        },
        {
            "id": "eval-2",
            "query": "How should RAG systems be evaluated in production?",
            "context": "Testing is important for any software system. RAG applications should be tested before deployment.",
            "key_facts": ["measurement", "golden test set"],
            "human_score": 0.15,
            "verdict": "incorrect"
        },
        {
            "id": "hnsw-1",
            "query": "What is HNSW indexing and how does it work for vector search?",
            "context": "A Reference Production Architecture — addressing issues like high latency, security, and vendor chaos requires a deliberate architecture. Milvus supports various index types including HNSW and IVF for vector search. In-process HNSW has lower latency for single-node setups.",
            "key_facts": ["HNSW", "vector search", "in-process", "low latency"],
            "human_score": 0.85,
            "verdict": "correct"
        },
        {
            "id": "mcp-1",
            "query": "How does MCP integration work for multi-source RAG?",
            "context": "MCP servers provide structured access to external tools. Each tool has a name, description, and input schema that lets the LLM decide when to use it. Common MCP sources include Confluence, Jira, web search, and code repositories.",
            "key_facts": ["MCP servers", "structured access", "external tools", "Confluence", "Jira"],
            "human_score": 0.9,
            "verdict": "correct"
        },
        {
            "id": "empty-1",
            "query": "What is the capital of France?",
            "context": "",
            "key_facts": ["Paris"],
            "human_score": 0.0,
            "verdict": "incorrect"
        }
    ]
}


def load_or_create_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            return json.load(f)
    # Create default set
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SET, f, ensure_ascii=False, indent=2)
    print(f"  Created default calibration set: {CALIBRATION_PATH}")
    return DEFAULT_SET


def llm_judge_score(query: str, context: str, key_facts: list[str], model: str) -> float:
    """Get LLM judge score (0.0–1.0) for a query+context pair."""
    if not context.strip():
        return 0.0
    prompt = (
        f"Rate how well the context answers the question.\n\n"
        f"Question: {query[:200]}\n\n"
        f"Context:\n{context[:2000]}\n\n"
        f"Expected key facts: {', '.join(key_facts)}\n\n"
        f"Reply ONLY with a single number 0.0 to 1.0:\n"
        f"1.0 = complete answer with all key facts\n"
        f"0.7 = good answer, missing some details\n"
        f"0.5 = partial, some facts present\n"
        f"0.3 = weak, only tangentially related\n"
        f"0.0 = does not answer"
    )
    try:
        r = requests.post(LLM_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 10,
        }, timeout=15)
        if r.status_code != 200:
            return -1.0
        answer = r.json()["choices"][0]["message"]["content"].strip()
        nums = re.findall(r'0\.\d+|1\.0', answer)
        return float(nums[0]) if nums else 0.0
    except Exception:
        return -1.0


def compute_thresholds(results: list[dict]) -> dict:
    """Find optimal score thresholds for verdicts based on human scores."""
    # Find thresholds that maximize agreement
    cuts = [t / 20 for t in range(1, 20)]  # 0.05, 0.10, ..., 0.95
    best_correct = 0.7
    best_partial = 0.4
    best_agreement = 0

    for c in cuts:
        for p in cuts:
            if p >= c:
                continue
            agree = 0
            for r in results:
                llm = r["llm_score"]
                human_verdict = r["human_verdict"]
                if llm >= c:
                    llm_v = "correct"
                elif llm >= p:
                    llm_v = "partial"
                else:
                    llm_v = "incorrect"
                if llm_v == human_verdict:
                    agree += 1
            agreement = agree / max(len(results), 1)
            if agreement > best_agreement:
                best_agreement = agreement
                best_correct = c
                best_partial = p

    return {
        "recommended_correct_threshold": round(best_correct, 2),
        "recommended_partial_threshold": round(best_partial, 2),
        "agreement_rate": round(best_agreement, 3),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate LLM-as-Judge")
    parser.add_argument("--quick", action="store_true", help="Run 5 questions only")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE, help="LLM model for judge")
    parser.add_argument("--llm-url", default=LLM_URL, help="LLM API endpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Judge Calibration")
    print(f"  Model: {args.judge_model}")
    print(f"  URL: {args.llm_url}")
    print("=" * 60)

    cal = load_or_create_calibration()
    questions = cal["questions"]
    if args.quick:
        questions = questions[:5]
    print(f"  Questions: {len(questions)}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']} ... ", end="", flush=True)
        t0 = time.time()
        score = llm_judge_score(q["query"], q["context"], q["key_facts"], args.judge_model)
        dt = time.time() - t0
        if score < 0:
            print(f"ERROR (LLM down?)")
            continue
        verdict = "correct" if score >= 0.7 else ("partial" if score >= 0.4 else "incorrect")
        human = q["human_score"]
        match = "✓" if abs(score - human) < 0.2 else ("~" if abs(score - human) < 0.4 else "✗")
        print(f"LLM={score:.2f}  human={human:.2f}  {match} ({dt:.1f}s)")

        results.append({
            "id": q["id"],
            "query": q["query"][:80],
            "llm_score": round(score, 2),
            "human_score": human,
            "llm_verdict": verdict,
            "human_verdict": q.get("verdict", verdict),
            "error": abs(score - human),
        })

    if not results:
        print("No results — LLM endpoint may be down")
        return

    # Compute metrics
    errors = [r["error"] for r in results if r["llm_score"] >= 0]
    mae = sum(errors) / max(len(errors), 1)
    print(f"\n  Mean Absolute Error: {mae:.3f}")

    # Confusion matrix
    print(f"\n── Confusion Matrix ──")
    print(f"{'':20s} {'correct':>10s} {'partial':>10s} {'incorrect':>10s}")
    for hv in ["correct", "partial", "incorrect"]:
        row = f"{hv:20s}"
        for lv in ["correct", "partial", "incorrect"]:
            n = sum(1 for r in results if r["human_verdict"] == hv and r["llm_verdict"] == lv)
            row += f" {n:>10d}"
        print(row)

    # Optimal thresholds
    thresholds = compute_thresholds(results)
    print(f"\n── Optimal Thresholds ──")
    print(f"  Correct ≥ {thresholds['recommended_correct_threshold']:.2f}")
    print(f"  Partial ≥ {thresholds['recommended_partial_threshold']:.2f}")
    print(f"  Agreement: {thresholds['agreement_rate']*100:.1f}%")

    # Recommend update
    current_thresholds = {"correct": 0.7, "partial": 0.5}
    rec = thresholds["recommended_correct_threshold"]
    rec_p = thresholds["recommended_partial_threshold"]
    print(f"\n── Recommendation ──")
    print(f"  Current: correct≥0.70  partial≥0.50")
    print(f"  Optimal: correct≥{rec:.2f}  partial≥{rec_p:.2f}")
    if abs(rec - 0.7) > 0.05 or abs(rec_p - 0.5) > 0.05:
        print(f"  → Update thresholds in eval_golden.py llm_judge()")

    # Save report
    report = {
        "meta": {"model": args.judge_model, "n_questions": len(results)},
        "mae": round(mae, 3),
        "optimal_thresholds": thresholds,
        "results": results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Report: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()