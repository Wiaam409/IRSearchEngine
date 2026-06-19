# -*- coding: utf-8 -*-
"""Quick API integration test — run from project root after indexes are built."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests

BASE = 'http://localhost:8000'

def test(name, path, payload):
    r = requests.post(BASE + path, json=payload, timeout=30)
    try:
        d = r.json()
    except Exception:
        print(f'[{name}] HTTP={r.status_code} BODY={r.text[:200]}')
        return
    status = d.get('status', 'N/A')
    data   = d.get('data', {})
    results = data.get('results', [])
    print(f'[{name}] HTTP={r.status_code} API={status} results={len(results)}')
    if r.status_code != 200:
        print(f'   ERROR detail: {d.get("detail", r.text[:200])}')
    for res in results[:3]:
        print(f'   doc_id={res["doc_id"]}  score={res["score"]:.4f}')

def test_refine(query):
    r = requests.post(BASE + '/refine', json={'query': query, 'spell_check': True, 'expand_synonyms': True}, timeout=10)
    d = r.json()
    print(f'[Refine] HTTP={r.status_code} original={d.get("data",{}).get("original_query","?")} refined={d.get("data",{}).get("refined_query","?")}')

print('=== API Integration Test ===')
test('TF-IDF',         '/search/tfidf',           {'query': 'سيارة', 'top_k': 5})
test('BM25',           '/search/bm25',            {'query': 'سيارة', 'top_k': 5, 'k1': 1.2, 'b': 0.75})
test('Embeddings',     '/search/embeddings',      {'query': 'سيارة', 'top_k': 5})
test('Hybrid-Parallel','/search/hybrid/parallel', {'query': 'سيارة', 'top_k': 5, 'sparse_model': 'bm25', 'fusion': 'rrf', 'sparse_weight': 0.5, 'dense_weight': 0.5})
test('Hybrid-Serial',  '/search/hybrid/serial',   {'query': 'سيارة', 'top_k': 5, 'candidate_multiplier': 10})
test_refine('سيار')
print('=== Done ===')
