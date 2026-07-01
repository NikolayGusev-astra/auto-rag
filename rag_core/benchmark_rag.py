#!/usr/bin/env python3
"""Benchmark: сравнение скорости RAG pipeline v0 vs v2."""
import asyncio
import json
import sys
import time
import os
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcd_router import classify as dcd
from rag_config import *

TEST_CASES = [
    {'id': 'ald-pro-ip', 'query': 'ALD Pro 3.2.1 смена IP адресов серверной группировки после развертывания'},
    {'id': 'sssd-krb5', 'query': 'SSSD krb5_auth_timeout ldap_deref_threshold ALD Pro доверительные отношения MSAD'},
    {'id': 'postgresql', 'query': 'настройка postgresql streaming replication debian 12'},
    {'id': 'terraform', 'query': 'terraform aws backend s3 state locking dynamodb'},
    {'id': 'алабуга', 'query': 'ООО АЛАБУГА МАШИНЕРИ внедрение ALD Pro VMmanager Termidesk RuPost WorksPad'},
]

def run_v0(query):
    """v0 — последовательная, без кеша."""
    import zvec
    from zvec import Query as ZQ
    import requests as rq
    
    traces = []
    t0 = time.time()
    
    t1 = time.time()
    r = dcd(query)
    traces.append('DCD({:.2f}s)'.format(time.time()-t1))
    
    zpath = r'C:\Users\n.gusev\.cache\zvec\wiki'
    with open(zpath + r'\LOCK', 'w') as f: f.write('')
    
    t1 = time.time()
    emb = rq.post(EMBEDDING_URL, json={'model': EMBEDDING_MODEL, 'input': [query[:2000]]}, timeout=30)
    emb = emb.json()['data'][0]['embedding']
    doclist = zvec.open(zpath).query(queries=[ZQ(field_name='embedding', vector=emb)], topk=5)
    zt = time.time()-t1
    traces.append('ZVec({}ch {:.2f}s)'.format(len(doclist), zt))
    
    t1 = time.time()
    web_r = rq.get('http://localhost:8888/search?q={}&format=json'.format(quote(query)),
                    headers={'User-Agent':'HermesRAG/1.0'}, timeout=10)
    wd = web_r.json() if web_r.status_code == 200 else {'results':[]}
    wt = time.time()-t1
    traces.append('Web({}res {:.2f}s)'.format(len(wd.get('results',[])), wt))
    
    total = time.time()-t0
    return total, traces

async def run_v2(query):
    """v2 — asyncio.gather + cache."""
    from rag_async import async_rag_search
    t0 = time.time()
    r = dcd(query)
    result = await async_rag_search(query, r)
    total = time.time()-t0
    result['total'] = total
    result['domain'] = r['domain']
    return result

print('=' * 70)
print('BENCHMARK: sync(v0) vs async(v2)')
print('=' * 70)

results = []
for tc in TEST_CASES:
    q = tc['query']
    qid = tc['id']
    print('\n--- {} ---'.format(qid))
    print('Query: {}...'.format(q[:60]))
    
    v0_total, v0_traces = run_v0(q)
    
    v2 = asyncio.run(run_v2(q))
    v2_total = v2['total']
    
    speedup = v0_total / max(v2_total, 0.01)
    
    print('  v0(sync) : {:.1f}s  {}'.format(v0_total, ' -> '.join(v0_traces)))
    print('  v2(async): {:.1f}s  source={} trace={}'.format(
        v2_total, v2.get('source','?'), v2.get('trace','?')))
    print('  speedup  : {:.1f}x'.format(speedup))
    
    results.append({
        'id': qid,
        'v0': round(v0_total, 1),
        'v2': round(v2_total, 1),
        'speedup': round(speedup, 1),
        'v0_trace': ' -> '.join(v0_traces),
        'v2_trace': v2.get('trace', '?'),
    })

print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
header = '{:25s} {:>8s} {:>8s} {:>8s}  {}'.format('Test', 'v0(s)', 'v2(s)', 'SpdUp', 'v2 source')
print(header)
print('-' * 70)
for r in results:
    line = '{:25s} {:7.1f}s {:7.1f}s {:6.1f}x  {}'.format(
        r['id'], r['v0'], r['v2'], r['speedup'], r['v2_trace'][:40])
    print(line)

avg = sum(r['speedup'] for r in results) / len(results)
print('-' * 70)
print('Average speedup: {:.1f}x'.format(avg))