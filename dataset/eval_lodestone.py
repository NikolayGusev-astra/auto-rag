#!/usr/bin/env python3
"""
Lodestone Golden Set Evaluator — проверяет качество поиска Lodestone.

Использование:
  python eval_lodestone.py                                    # полный прогон
  python eval_lodestone.py --query "ALD Pro настройка DNS"    # одиночный запрос
  python eval_lodestone.py --list                             # список вопросов

Результат: lodestone_eval_report.json
"""

import json, os, sys, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# LLM Judge (qwen2.5-7b, локальный)
LLM_JUDGE_URL = "http://localhost:1234/v1/chat/completions"
LLM_JUDGE_PROMPT = """Ты — строгий судья RAG-системы.

Запрос: {query}

Ответ RAG: {answer}

Ключевые факты, которые должны быть в ответе: {key_facts}

Оцени ответ по шкале 0.0-1.0:
- 0.0 = ответ пустой или полностью неверный
- 0.3 = частично затронута тема, ключевые факты отсутствуют
- 0.7 = большинство фактов есть, ответ полезный
- 1.0 = все факты есть, ответ полный и точный

Верни ТОЛЬКО число от 0.0 до 1.0"""


def llm_judge(query, answer, key_facts):
    import requests
    try:
        r = requests.post(LLM_JUDGE_URL, json={
            "model": "qwen2.5-7b-instruct",
            "messages": [{"role": "user", "content": LLM_JUDGE_PROMPT.format(
                query=query, answer=answer[:1000], key_facts=", ".join(key_facts))}],
            "temperature": 0.0, "max_tokens": 10,
        }, timeout=15)
        text = r.json()["choices"][0]["message"]["content"].strip()
        nums = re.findall(r'0\.\d+|1\.0', text)
        return float(nums[0]) if nums else 0.0
    except:
        return 0.0


def load_golden(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'golden_set_lodestone.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("questions", []), data.get("meta", {})


def query_lodestone(query, max_results=3):
    """Запросить Lodestone через MCP с токеном из Hermes config."""
    import yaml
    import requests as sync_requests
    
    # Читаем токен из config.yaml
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    token = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        ls_cfg = config.get('lodestone', {})
        token = ls_cfg.get('headers', {}).get('Authorization', '')
    
    if not token:
        # Fallback: env var
        token = os.getenv("LODESTONE_TOKEN", "")
    
    url = "<LODESTONE_MCP_URL>"
    headers = {
        "Authorization": token,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    
    try:
        # Init
        r = sync_requests.post(url, json={
            "jsonrpc": "2.0", "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
            "id": "1"
        }, headers=headers, timeout=10)
        data = r.json()
        sid = data.get("sessionId", "")
        
        # Search
        h2 = dict(headers)
        if sid:
            h2["mcp-session-id"] = sid
        
        # Notify
        sync_requests.post(url, json={
            "jsonrpc": "2.0", "method": "notifications/initialized", "params": {}
        }, headers=h2, timeout=5)
        
        # Call search
        r = sync_requests.post(url, json={
            "jsonrpc": "2.0", "id": "2", "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": query, "limit": max_results}}
        }, headers=h2, timeout=20)
        data = r.json()
        
        # Parse result
        result = data.get("result", data.get("content", []))
        chunks = []
        if isinstance(result, list):
            for item in result[:max_results]:
                if isinstance(item, dict):
                    text = item.get("text", "") or item.get("resource", {}).get("text", "")
                    if text:
                        chunks.append({"text": text[:800], "source": "lodestone"})
        return chunks
    except Exception as e:
        print(f"  [Lodestone error: {e}]")
        return []


def eval_one(query, key_facts):
    t0 = time.time()
    results = query_lodestone(query)
    lat = time.time() - t0
    
    answer = "\n".join(r.get("text", "")[:500] for r in results[:3]) if results else ""
    has_content = len(results) > 0
    
    score = llm_judge(query, answer, key_facts) if answer else 0.0
    verdict = "correct" if score >= 0.7 else "partial" if score >= 0.3 else "incorrect"
    empty = "empty" if not has_content else ""
    
    return {
        "chunks": len(results),
        "has_content": has_content,
        "latency_s": round(lat, 2),
        "answer": answer[:300],
        "llm_score": score,
        "verdict": verdict if not empty else "empty",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query")
    parser.add_argument("--list", action="store_true", help="List questions")
    parser.add_argument("--limit", type=int, default=0, help="Max questions")
    args = parser.parse_args()
    
    questions, meta = load_golden()
    
    if args.list:
        print(f"Lodestone Golden Set — {len(questions)} вопросов")
        print("="*60)
        for i, q in enumerate(questions, 1):
            print(f"{i:2d}. {q['query'][:70]}")
            print(f"    source: {q['source_type']}  facts: {len(q['key_facts'])}")
        return
    
    if args.query:
        print(f"Query: {args.query}")
        results = query_lodestone(args.query)
        print(f"Chunks: {len(results)}")
        for r in results:
            print(f"  {(r.get('text','') or r.get('content',''))[:300]}")
        return
    
    n = args.limit or len(questions)
    print(f"Lodestone Eval — {n}/{len(questions)} questions")
    print("="*60)
    
    all_results = []
    ok, partial, incorrect, empty = 0, 0, 0, 0
    
    for i, q in enumerate(questions[:n]):
        print(f"  [{i+1}/{n}] {q['id'][:30]} ... ", end="", flush=True)
        r = eval_one(q["query"], q["key_facts"])
        all_results.append({**{"id": q["id"], "query": q["query"]}, **r})
        
        if r["verdict"] == "correct": ok += 1
        elif r["verdict"] == "partial": partial += 1
        elif r["verdict"] == "empty": empty += 1
        else: incorrect += 1
        
        print(f"{r['verdict']:10s} {r.get('latency_s',0):.1f}s  {r['chunks']}ch")
    
    total = n
    print("="*60)
    print(f"Correct:   {ok}/{total} ({ok/total*100:.0f}%)")
    print(f"Partial:   {partial}/{total} ({partial/total*100:.0f}%)")
    print(f"Incorrect: {incorrect}/{total} ({incorrect/total*100:.0f}%)")
    print(f"Empty:     {empty}/{total} ({empty/total*100:.0f}%)")
    print(f"Avg chunks: {sum(r['chunks'] for r in all_results)/len(all_results):.1f}")
    
    report = {
        "meta": meta,
        "summary": {
            "total": total, "correct": ok, "partial": partial,
            "incorrect": incorrect, "empty": empty,
        },
        "results": all_results,
    }
    path = os.path.join(os.path.dirname(__file__), 'lodestone_eval_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()