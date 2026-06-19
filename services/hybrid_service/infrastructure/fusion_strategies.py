from typing import List, Dict
from collections import defaultdict
from services.hybrid_service.domain.interfaces import IFusionStrategy
from services.hybrid_service.domain.models import ScoredDocument, FusionConfig

class RrFusionStrategy(IFusionStrategy):
    def fuse(self, ranked_lists: List[List[ScoredDocument]], config: FusionConfig) -> List[ScoredDocument]:
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                score = 1.0 / (config.rrf_k_constant + (rank + 1))
                doc_scores[doc.doc_id] += score
                
        fused = [ScoredDocument(doc_id, score) for doc_id, score in doc_scores.items()]
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

class ScoreFusionStrategy(IFusionStrategy):
    def fuse(self, ranked_lists: List[List[ScoredDocument]], config: FusionConfig) -> List[ScoredDocument]:
        if not ranked_lists or len(ranked_lists) < 2:
            return ranked_lists[0] if ranked_lists else []
            
        sparse_list = ranked_lists[0]
        dense_list = ranked_lists[1]
        
        doc_scores: Dict[str, float] = defaultdict(float)
        
        def normalize(lst: List[ScoredDocument]) -> Dict[str, float]:
            if not lst:
                return {}
            min_s = min(lst, key=lambda x: x.score).score
            max_s = max(lst, key=lambda x: x.score).score
            
            normalized = {}
            for doc in lst:
                if max_s == min_s:
                    normalized[doc.doc_id] = 1.0 if max_s > 0 else 0.0
                else:
                    normalized[doc.doc_id] = (doc.score - min_s) / (max_s - min_s)
            return normalized

        norm_sparse = normalize(sparse_list)
        norm_dense = normalize(dense_list)
        
        all_docs = set(norm_sparse.keys()).union(set(norm_dense.keys()))
        
        for doc_id in all_docs:
            s_score = norm_sparse.get(doc_id, 0.0)
            d_score = norm_dense.get(doc_id, 0.0)
            combined = (config.sparse_weight * s_score) + (config.dense_weight * d_score)
            doc_scores[doc_id] = combined
            
        fused = [ScoredDocument(doc_id, score) for doc_id, score in doc_scores.items()]
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused
