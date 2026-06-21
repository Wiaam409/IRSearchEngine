import sys
import os
import json
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUTF8"] = "1"
from datasets.adapters.beir_adapter import BeirAdapter
from services.evaluation_service.domain.models import RelevanceJudgment
from services.evaluation_service.domain.interfaces import IRetrievalModelAdapter
from api.dependencies.containers import (
    get_evaluation_service, get_tfidf_service, get_bm25_service,
    get_dense_service, get_parallel_hybrid_service, get_serial_hybrid_service,
    get_refinement_service
)

class EvalAdapter(IRetrievalModelAdapter):
    def __init__(self, use_case):
        self.use_case = use_case
        
    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int):
        if hasattr(self.use_case, 'retrieve'):
            result = self.use_case.retrieve(query_text, top_k=top_k)
        elif hasattr(self.use_case, 'rank'):
            result = self.use_case.rank(query_text, top_k=top_k)
        else:
            raise AttributeError("Use case must have retrieve or rank method")
            
        docs = result if isinstance(result, list) else getattr(result, "results", [])
        return [(getattr(doc, "doc_id", doc.get("doc_id", "") if isinstance(doc, dict) else ""), 
                 getattr(doc, "score", doc.get("score", 0.0) if isinstance(doc, dict) else 0.0)) for doc in docs]

class RefinedAdapter(IRetrievalModelAdapter):
    def __init__(self, use_case, refinement_service):
        self.use_case = use_case
        self.refine = refinement_service
        
    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int):
        refined = self.refine.refine(query_text)
        new_q = refined.refined_query if hasattr(refined, 'refined_query') else refined
        if hasattr(self.use_case, 'retrieve'):
            result = self.use_case.retrieve(new_q, top_k=top_k)
        else:
            result = self.use_case.rank(new_q, top_k=top_k)
            
        docs = result if isinstance(result, list) else getattr(result, "results", [])
        return [(getattr(doc, "doc_id", doc.get("doc_id", "") if isinstance(doc, dict) else ""), 
                 getattr(doc, "score", doc.get("score", 0.0) if isinstance(doc, dict) else 0.0)) for doc in docs]

class TunedAdapter(IRetrievalModelAdapter):
    def __init__(self, use_case, k1=1.5, b=0.4):
        self.use_case = use_case
        self.k1 = k1
        self.b = b
        
    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int):
        from services.ranking_service.domain.models import Bm25Parameters
        params = Bm25Parameters(k1=self.k1, b=self.b)
        result = self.use_case.rank(query_text, top_k=top_k, params=params)
        return [(getattr(doc, "doc_id", ""), getattr(doc, "score", 0.0)) for doc in result]

def run_evaluation():
    print(f"Running evaluation...")
    adapter = BeirAdapter()
    
    # Load queries
    queries = {}
    print("Loading queries...")
    for q in tqdm(adapter.load_queries(), desc="Loading queries"):
        queries[q.query_id] = q.text
        
    # Load qrels
    qrels = []
    print("Loading qrels...")
    for qrel in tqdm(adapter.load_qrels(), desc="Loading qrels"):
        if qrel.query_id in queries:
            qrels.append(RelevanceJudgment(qrel.query_id, qrel.doc_id, qrel.relevance))
            
    print(f"Total qrels loaded: {len(qrels)}")
    print(f"Total queries used: {len(queries)}")
    print(f"All qrels queries used: {len(set(q.query_id for q in qrels)) == len(queries)}")
            
    eval_service = get_evaluation_service()
    
    print("Loading models into memory...")
    models = {
        "TF-IDF": EvalAdapter(get_tfidf_service()),
        "BM25": EvalAdapter(get_bm25_service()),
        "BM25 (Tuned)": TunedAdapter(get_bm25_service(), k1=1.5, b=0.4),
        "BM25 (Refined)": RefinedAdapter(get_bm25_service(), get_refinement_service()),
        "Embeddings": EvalAdapter(get_dense_service()),
        "Hybrid (RRF)": EvalAdapter(get_parallel_hybrid_service("bm25", "rrf")),
        "Hybrid (Serial)": EvalAdapter(get_serial_hybrid_service())
    }
    
    top_k_list = [10, 100]
    
    results = {
        "Model": [],
        "NDCG@10": [],
        "Recall@100": [],
        "Precision@10": [],
        "MAP": []
    }
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        report = eval_service.evaluate(model, name, queries, qrels, top_k_list)
        
        metrics = {m.metric_name: m.value for m in report.aggregate_metrics}
        
        results["Model"].append(name)
        results["NDCG@10"].append(round(metrics.get("nDCG@10", 0.0), 4))
        results["Recall@100"].append(round(metrics.get("Recall@100", 0.0), 4))
        results["Precision@10"].append(round(metrics.get("P@10", 0.0), 4))
        results["MAP"].append(round(metrics.get("MAP", 0.0), 4))
        
    results["metadata"] = {
        "total_qrels_loaded": len(qrels),
        "total_queries_used": len(queries),
        "all_qrels_queries_used": len(set(q.query_id for q in qrels)) == len(queries)
    }
        
    out_dir = "datasets/cache"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evaluation_metrics.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation complete! Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_evaluation()
