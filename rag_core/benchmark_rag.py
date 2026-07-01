#!/usr/bin/env python3
"""Benchmark: ZVec speed + bge-reranker vs qwen3-4b eval. 10 queries."""
import json, os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import *
from dcd_router import classify

# ── Helpers ───────────────────────────────────────────────────────
def embed(text):
    p = subprocess.run(["curl","-s","--max-time","10",EMBEDDING_URL,
        "-d",json.dumps({"model":EMBEDDING_MODEL,"input":[text]}),
        "-H","Content-Type: application/json"],
        capture_output=True,text=True,timeout=15)
    return json.loads(p.stdout)["data"][0]["embedding"] if p.returncode==0 and p.stdout else None

def zvec_search(emb, topk=5):
    import zvec; zvec.init()
    from zvec import Query
    lock = os.path.join(ZVEC_PATH, ZVEC_WIKI_COLLECTION, "LOCK")
    if not os.path.exists(lock):
        fd = os.open(lock, os.O_CREAT|os.O_WRONLY, 0o644); os.close(fd)
    coll = zvec.open(os.path.join(ZVEC_PATH, ZVEC_WIKI_COLLECTION))
    t0 = time.time()
    res = coll.query(queries=[Query(field_name="embedding", vector=emb)], topk=topk,
                     output_fields=["source","content"])
    return time.time()-t0, res

def rerank(query, chunks):
    """bge-reranker via LM Studio — embed query+passage pairs and compare."""
    if not chunks or not RERANK_ENABLED: return chunks
    import zvec
    from zvec import Query
    # Embed query
    q_emb = embed(query)
    if not q_emb: return chunks
    # Score each chunk by cosine similarity to query
    for c in chunks:
        c_content = (c.fields or {}).get("content", "")
        if not c_content: continue
        c_emb = embed(c_content[:500])
        if c_emb:
            dot = sum(a*b for a,b in zip(q_emb, c_emb))
            nq = sum(a*a for a in q_emb)**0.5
            nc = sum(b*b for b in c_emb)**0.5
            c.score = dot / (nq * nc) if nq*nc else 0
    chunks.sort(key=lambda x: x.score, reverse=True)
    return chunks[:RERANK_FINAL_K]

def qwen_eval(query, chunks):
    p = subprocess.run(["curl","-s","--max-time","30",LM_STUDIO_CHAT_URL,
        "-d",json.dumps({"model":LLM_REWRITE_MODEL,"messages":[
            {"role":"user","content":f"Rate relevance 0.0-1.0. Reply ONLY a number.\nQuery: {query[:200]}\nDocs:\n"+'\n'.join([f'[{i}] {c["content"][:200]}' for i,c in enumerate(chunks[:3])])
        }],"temperature":0.0,"max_tokens":200}),
        "-H","Content-Type: application/json"],
        capture_output=True,text=True,timeout=35)
    if p.returncode==0 and p.stdout:
        data = json.loads(p.stdout)
        content = data["choices"][0]["message"].get("content","") or ""
        # qwen3-4b thinking model puts answer in reasoning_content
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content","") or ""
        import re
        nums = re.findall(r"0\.\d+|1\.0", content)
        if nums: return float(nums[0])
    return 0.0

# ── Test queries ──────────────────────────────────────────────────
queries = [
    # ford-club
    "Ford Explorer II 1998 шрус полуось ремонт",
    "VIN 1FMZU34E6WUD14705 замена масла двигатель",
    "carpc установка в Ford Explorer магнитола",
    "зимние шины 235/75 R15 для Ford Explorer",
    "партномер F5TZ-3A427A пыльник шруса",
    # devops
    "настройка nginx reverse proxy docker",
    "postgresql debian установка настройка бэкап",
    "openvpn server setup ubuntu конфиг",
    "xray reality vless config tls",
    "systemd service автозапуск скрипта",
]

print(f"{'Query':<35} {'Domain':<18} {'Emb':>6} {'ZVec':>6} {'Qwen':>6} {'TopScore':>8} {'QwenScore':>9}")
print("-"*95)

for q in queries:
    dcd = classify(q)
    dom = dcd["domain"]

    t0 = time.time()
    emb = embed(q)
    t_emb = time.time()-t0
    if not emb: continue

    t_vec, results = zvec_search(emb, topk=5)
    top_score = results[0].score if results else 0

    # Qwen eval
    qwen_score = 0.0
    t_qwen = 0.0
    if results:
        chunks = [{"content":(c.fields or {}).get("content","")} for c in results[:3]]
        t0 = time.time()
        qwen_score = qwen_eval(q, chunks)
        t_qwen = time.time()-t0

    print(f"{q[:34]:<35} {dom:<18} {t_emb:>5.2f}s {t_vec:>5.2f}s {t_qwen:>5.2f}s {top_score:>7.4f} {qwen_score:>8.4f}")

print("\nDone.")
