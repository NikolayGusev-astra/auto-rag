#!/usr/bin/env python3
"""Run golden set and report results."""
import json, os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_search import CragSearch
from dcd_router import classify

with open(os.path.join(os.path.dirname(__file__), 'rag_golden.json')) as f:
    golden = json.load(f)

s = CragSearch()
results = []

for t in golden["tests"]:
    q = t["query"]
    exp = t["expected"]
    
    t0 = time.time()
    r = s.search(q, k=5)
    elapsed = time.time() - t0
    
    chunks = r.get("chunks", [])
    top_score = max((c.get("score", 0) for c in chunks), default=0)
    sources = [c.get("source", "") for c in chunks]
    sources_str = " ".join(sources)
    
    # Check pass/fail
    must_be_empty = exp.get("must_be_empty", False)
    if must_be_empty:
        passed = len(chunks) == 0
    else:
        source_ok = any(x in sources_str for x in exp.get("source_contains", []))
        score_ok = top_score >= exp.get("min_score", 0)
        passed = source_ok or score_ok
    
    results.append({
        "id": t["id"],
        "domain": t["domain"],
        "query": q[:50],
        "passed": passed,
        "top_score": round(top_score, 4),
        "sources": sources[:3],
        "time": round(elapsed, 2),
        "em_pass": r.get("entity_match_pass", None),
    })
    
    status = "✅" if passed else "❌"
    src = sources[0][:40] if sources else "(empty)"
    print(f"{status} {t['id']:12s} score={top_score:.3f} time={elapsed:.1f}s src={src}")

passed = sum(1 for r in results if r["passed"])
total = len(results)
recall = passed / total * 100
print(f"\n{'='*50}")
print(f"Recall@{5}: {passed}/{total} = {recall:.0f}%")
print(f"Avg time: {sum(r['time'] for r in results)/total:.2f}s")
print(f"Entity Match pass rate: {sum(1 for r in results if r['em_pass']==True)}/{total}")

# Save detailed results
with open('/root/rag-deploy-v2/rag_benchmark_result.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nDetailed results saved to rag_benchmark_result.json")
