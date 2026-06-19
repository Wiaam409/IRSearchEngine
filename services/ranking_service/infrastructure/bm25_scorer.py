from services.ranking_service.domain.interfaces import IBm25Scorer

class Bm25Scorer(IBm25Scorer):
    def compute_score(self, tf: int, doc_len: int, avgdl: float, idf: float, k1: float, b: float) -> float:
        if avgdl <= 0:
            return 0.0
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
        
        return idf * (numerator / denominator)
