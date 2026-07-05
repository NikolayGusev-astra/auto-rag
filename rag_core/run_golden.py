#!/usr/bin/env python3
"""Run golden set and report results."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcd_router import classify

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_set.json')
REPORT_PATH = os.path.join(os.path.dirname(__file__), 'golden_eval_report.json')

def load_golden():
    with open(GOLDEN_PATH, encoding='utf-8') as f:
        return json.load(f)

def main():
    golden = load_golden()
    questions = golden.get('questions', [])
    results = []

    for t in questions:
        q = t['query']
        exp_domain = t.get('expected_domain', '')
        exp_source = t.get('expected_source', '')

        t0 = time.time()
        dcd = classify(q)
        elapsed = time.time() - t0

        domain = dcd.get('domain', '')
        collection = dcd.get('collection', '')
        confidence = dcd.get('confidence', 0)

        domain_ok = domain == exp_domain or not exp_domain
        passed = domain_ok

        results.append({
            'id': t['id'],
            'query': q[:80],
            'passed': passed,
            'expected_domain': exp_domain,
            'actual_domain': domain,
            'expected_source': exp_source,
            'confidence': round(confidence, 3),
            'time': round(elapsed, 3),
        })

        status = '✅' if passed else '❌'
        print(f"{status} {t['id']:20s} domain={domain:15s} expected={exp_domain:15s} conf={confidence:.2f} time={elapsed:.2f}s")

    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    accuracy = passed / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"DCD accuracy: {passed}/{total} = {accuracy:.0f}%")
    print(f"Avg time: {sum(r['time'] for r in results)/total:.2f}s" if total else "")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'type': 'dcd_eval'}, 'results': results}, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {REPORT_PATH}")

if __name__ == '__main__':
    main()
