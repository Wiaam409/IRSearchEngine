from typing import List, Tuple
from services.evaluation_service.domain.interfaces import IRetrievalModelAdapter
from services.retrieval_service.domain.interfaces import IRetrievalService
from services.ranking_service.domain.interfaces import IRankingService

class TfidfModelAdapter(IRetrievalModelAdapter):
    def __init__(self, use_case: IRetrievalService):
        self.use_case = use_case

    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        result = self.use_case.retrieve(query_text, top_k=top_k)
        return [(doc.doc_id, doc.score) for doc in result.results]

class Bm25ModelAdapter(IRetrievalModelAdapter):
    def __init__(self, use_case: IRankingService):
        self.use_case = use_case

    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        # Using default BM25 parameters.
        result = self.use_case.rank(query_text, top_k=top_k)
        return [(doc.doc_id, doc.score) for doc in result]
