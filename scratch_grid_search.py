import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.adapters.beir_adapter import BeirAdapter
from services.evaluation_service.domain.models import RelevanceJudgment
from api.dependencies.containers import get_bm25_service, get_evaluation_service
from services.evaluation_service.domain.interfaces import IRetrievalModelAdapter
from services.ranking_service.domain.models import Bm25Parameters

class ParamEvalAdapter(IRetrievalModelAdapter):
    def __init__(self, use_case, k1, b):
        self.use_case = use_case
        self.k1 = k1
        self.b = b
        
    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int):
        params = Bm25Parameters(k1=self.k1, b=self.b)
        result = self.use_case.rank(query_text, top_k=top_k, params=params)
        return [(getattr(doc, "doc_id", ""), getattr(doc, "score", 0.0)) for doc in result]

def run_grid():
    adapter = BeirAdapter()
    queries = {q.query_id: q.text for q in adapter.load_queries()}
    qrels = [RelevanceJudgment(q.query_id, q.doc_id, q.relevance) for q in adapter.load_qrels() if q.query_id in queries]
    
    bm25 = get_bm25_service()
    evaluator = get_evaluation_service()
    
    k1_vals = [0.9, 1.2, 1.5, 2.0]
    b_vals = [0.4, 0.6, 0.75, 0.9]
    
    best_ndcg = 0
    best_params = None
    
    print("Running BM25 Grid Search...")
    for k1 in k1_vals:
        for b in b_vals:
            model = ParamEvalAdapter(bm25, k1, b)
            report = evaluator.evaluate(model, f"BM25_k1={k1}_b={b}", queries, qrels, [10])
            ndcg = next((m.value for m in report.aggregate_metrics if m.metric_name == "nDCG@10"), 0)
            print(f"k1={k1}, b={b} -> NDCG@10: {ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_params = (k1, b)
                
    print(f"\nBest Params: k1={best_params[0]}, b={best_params[1]} with NDCG@10={best_ndcg:.4f}")

if __name__ == '__main__':
    run_grid()
