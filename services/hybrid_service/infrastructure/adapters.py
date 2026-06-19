from typing import List, Any
from services.hybrid_service.domain.interfaces import IRetriever
from services.hybrid_service.domain.models import ScoredDocument

class RetrieverAdapter(IRetriever):
    def __init__(self, use_case: Any):
        """
        Wraps TF-IDF, BM25, or Dense retrieval use cases.
        """
        self.use_case = use_case

    def retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        if hasattr(self.use_case, 'retrieve'):
            result = self.use_case.retrieve(query, top_k=top_k)
            if isinstance(result, list):
                return [ScoredDocument(doc.doc_id, doc.score) for doc in result]
            else:
                return [ScoredDocument(doc.doc_id, doc.score) for doc in getattr(result, "results", [])]
        elif hasattr(self.use_case, 'rank'):
            result = self.use_case.rank(query, top_k=top_k)
            return [ScoredDocument(doc.doc_id, doc.score) for doc in result]
        else:
            raise AttributeError("Use case does not have 'retrieve' or 'rank' method.")
