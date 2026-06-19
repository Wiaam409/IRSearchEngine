from typing import List
from services.hybrid_service.domain.interfaces import IHybridRetrievalService, IRetriever, IFusionStrategy
from services.hybrid_service.domain.models import HybridResult, FusionConfig

class ParallelHybridRetrieveUseCase(IHybridRetrievalService):
    def __init__(self, retrievers: List[IRetriever], fusion_strategy: IFusionStrategy):
        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy

    def retrieve(self, query: str, top_k: int = 10, config: FusionConfig = None) -> HybridResult:
        if config is None:
            config = FusionConfig()
            
        ranked_lists = []
        for retriever in self.retrievers:
            results = retriever.retrieve(query, top_k=top_k)
            ranked_lists.append(results)
            
        fused_results = self.fusion_strategy.fuse(ranked_lists, config)
        
        return HybridResult(
            query=query,
            results=fused_results[:top_k],
            fusion_method_used=self.fusion_strategy.__class__.__name__
        )

class SerialHybridRetrieveUseCase(IHybridRetrievalService):
    def __init__(self, sparse_retriever: IRetriever, dense_retriever: IRetriever):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever

    def retrieve(self, query: str, top_k: int = 10, candidate_multiplier: int = 10) -> HybridResult:
        candidate_count = top_k * candidate_multiplier
        
        sparse_candidates = self.sparse_retriever.retrieve(query, top_k=candidate_count)
        sparse_doc_ids = {doc.doc_id for doc in sparse_candidates}
        
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_count)
        
        filtered_dense = [doc for doc in dense_results if doc.doc_id in sparse_doc_ids]
        filtered_dense.sort(key=lambda x: x.score, reverse=True)
        
        return HybridResult(
            query=query,
            results=filtered_dense[:top_k],
            fusion_method_used="SerialCascade"
        )
