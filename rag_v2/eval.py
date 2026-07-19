"""
RAG v2 Golden Set Evaluator — сравнивает v1 (rag_async) vs v2 (rag_v2).

Запуск:
  python -m rag_v2.eval                  # v2 only, quick
  python -m rag_v2.eval --compare        # v1 vs v2
  python -m rag_v2.eval --query "..."    # single query
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_v2.engine import rag_v2_search

# ── LLM Judge (copy from eval_golden.py) ──
LLM_JUDGE_PROMPT = """Ты — строгий судья RAG-системы.

Запрос: {query}

Ответ системы: {answer}

Ключевые факты, которые должны быть в ответе: {key_facts}

Оцени ответ по шкале 0.0-1.0:
- 0.0 = ответ неверный или пустой
- 0.3 = частично затронута тема, но ключевые факты отсутствуют
- 0.7 = большинство фактов есть, ответ полезный
- 1.0 = все факты есть, ответ полный и точный

Верни ТОЛЬКО число от 0.0 до 1.0"""


def _llm_judge(query: str, answer: str, key_facts: list[str]) -> float:
    """LLM judge для оценки качества ответа."""
    import requests as sync_requests
    try:
        from rag_core.rag_config import LM_STUDIO_CHAT_URL
        r = sync_requests.post(LM_STUDIO_CHAT_URL, json={
            "model": "qwen2.5-7b-instruct",
            "messages": [{"role": "user", "content": LLM_JUDGE_PROMPT.format(
                query=query, answer=answer[:1000], key_facts=", ".join(key_facts))}],
            "temperature": 0.0, "max_tokens": 10,
        }, timeout=15)
        text = r.json()["choices"][0]["message"]["content"].strip()
        import re
        nums = re.findall(r'0\.\d+|1\.0', text)
        return float(nums[0]) if nums else 0.0
    except Exception:
        return 0.0


def load_golden() -> list[dict]:
    """Загрузить golden_set.json."""
    path = os.path.join(os.path.dirname(__file__), '..', 'golden_set.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("questions", [])


async def eval_v2(questions: list[dict]) -> list[dict]:
    """Прогнать v2 на всех вопросах."""
    results = []
    for i, q in enumerate(questions):
        t0 = time.time()
        print(f"  [{i+1}/{len(questions)}] {q['id']} ... ", end="", flush=True)
        try:
            result = await rag_v2_search(q["query"])
            answer = result.get("answer", "")
            source = result.get("source", "empty")
            fusion = result.get("fusion_needed", False)
            sources_used = result.get("sources_used", [])
            latency = time.time() - t0
            
            # LLM judge
            key_facts = q.get("key_facts", [])
            score = _llm_judge(q["query"], answer, key_facts) if answer else 0.0
            verdict = "correct" if score >= 0.7 else "partial" if score >= 0.3 else "incorrect"
            
            results.append({
                "id": q["id"],
                "query": q["query"],
                "source": source,
                "fusion": fusion,
                "sources_used": sources_used,
                "answer": answer[:200],
                "llm_score": score,
                "verdict": verdict,
                "latency_s": round(latency, 2),
            })
            print(f"{verdict} ({source}, {latency:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": q["id"], "error": str(e), "verdict": "error", "latency_s": round(time.time() - t0, 2),
            })
    
    return results


def print_summary(results: list[dict]):
    """Печать сводки."""
    total = len(results)
    correct = sum(1 for r in results if r.get("verdict") == "correct")
    partial = sum(1 for r in results if r.get("verdict") == "partial")
    incorrect = sum(1 for r in results if r.get("verdict") == "incorrect")
    errors = sum(1 for r in results if r.get("verdict") == "error")
    empty = sum(1 for r in results if r.get("source") == "empty")
    
    print(f"\n{'='*60}")
    print(f"RAG v2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:       {total}")
    print(f"  Correct:     {correct} ({correct/total*100:.0f}%)")
    print(f"  Partial:     {partial} ({partial/total*100:.0f}%)")
    print(f"  Incorrect:   {incorrect} ({incorrect/total*100:.0f}%)")
    print(f"  Errors:      {errors}")
    print(f"  Empty:       {empty}")
    print(f"  Avg latency: {sum(r['latency_s'] for r in results if 'latency_s' in r)/len(results):.1f}s")
    
    # Source breakdown
    from collections import Counter
    sources = Counter(r.get("source", "empty") for r in results)
    print(f"\n  Source breakdown:")
    for src, count in sources.most_common():
        print(f"    {src:20s}: {count}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare v1 vs v2")
    parser.add_argument("--query", type=str, help="Single query")
    args = parser.parse_args()
    
    if args.query:
        result = asyncio.run(rag_v2_search(args.query))
        print(f"Source: {result.get('source')}")
        print(f"Answer: {result.get('answer', '')[:500]}")
        print(f"Src used: {result.get('sources_used', [])}")
        return
    
    questions = load_golden()
    print(f"RAG v2 Eval — {len(questions)} questions")
    print(f"{'='*60}")
    
    results = asyncio.run(eval_v2(questions))
    print_summary(results)
    
    # Save
    report = {
        "version": "v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "summary": {
            "total": len(results),
            "correct": sum(1 for r in results if r.get("verdict") == "correct"),
            "partial": sum(1 for r in results if r.get("verdict") == "partial"),
            "incorrect": sum(1 for r in results if r.get("verdict") == "incorrect"),
            "breakdown": dict(Counter(r.get("source", "empty") for r in results)),
        }
    }
    from collections import Counter
    
    path = os.path.join(os.path.dirname(__file__), '..', 'golden_eval_v2_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()