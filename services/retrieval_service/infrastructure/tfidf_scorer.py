import math
from typing import Dict
from services.retrieval_service.domain.interfaces import IScorer, IIndexReader

class TfidfScorer(IScorer):
    def __init__(self, index_reader: IIndexReader):
        self.index_reader = index_reader

    def score(self, query_vector: Dict[str, float], doc_id: str, precomputed_dot_product: float = None) -> float:
        """
        Calculates cosine similarity. If dot_product is passed (via TAAT algorithm),
        it acts as an O(1) finalizer. Otherwise, it falls back to DAAT calculation.
        """
        if precomputed_dot_product is not None:
            dot_product = precomputed_dot_product
        else:
            dot_product = 0.0
            for term, q_weight in query_vector.items():
                postings = self.index_reader.get_term_posting_list(term)
                for pid, tf in postings:
                    if pid == doc_id:
                        idf = self.index_reader.get_idf(term)
                        dot_product += q_weight * (tf * idf)
                        break

        doc_norm = self.index_reader.get_doc_norm(doc_id)
        
        query_norm_sq = sum(w * w for w in query_vector.values())
        query_norm = math.sqrt(query_norm_sq) if query_norm_sq > 0 else 1.0

        if doc_norm == 0.0 or query_norm == 0.0:
            return 0.0

        return dot_product / (query_norm * doc_norm)
