import argparse
from services.hybrid_service.domain.models import FusionConfig
from services.hybrid_service.infrastructure.fusion_strategies import RrFusionStrategy, ScoreFusionStrategy
from services.hybrid_service.infrastructure.adapters import RetrieverAdapter
from services.hybrid_service.application.use_cases import ParallelHybridRetrieveUseCase, SerialHybridRetrieveUseCase

class MockSparseUseCase:
    def retrieve(self, query, top_k):
        from services.hybrid_service.domain.models import ScoredDocument
        return [ScoredDocument("doc1", 0.9), ScoredDocument("doc2", 0.8), ScoredDocument("doc3", 0.7)][:top_k]

class MockDenseUseCase:
    def retrieve(self, query, top_k):
        from services.hybrid_service.domain.models import ScoredDocument
        return [ScoredDocument("doc3", 0.95), ScoredDocument("doc1", 0.85), ScoredDocument("doc4", 0.6)][:top_k]

def main():
    parser = argparse.ArgumentParser(description="Hybrid Retrieval CLI")
    parser.add_argument("--mode", choices=["parallel", "serial"], default="parallel")
    parser.add_argument("--fusion", choices=["rrf", "score"], default="rrf")
    parser.add_argument("--sparse_weight", type=float, default=0.5)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    sparse_retriever = RetrieverAdapter(MockSparseUseCase())
    dense_retriever = RetrieverAdapter(MockDenseUseCase())

    if args.mode == "parallel":
        strategy = RrFusionStrategy() if args.fusion == "rrf" else ScoreFusionStrategy()
        use_case = ParallelHybridRetrieveUseCase(
            retrievers=[sparse_retriever, dense_retriever],
            fusion_strategy=strategy
        )
        config = FusionConfig(sparse_weight=args.sparse_weight, dense_weight=args.dense_weight)
        res = use_case.retrieve("test query", top_k=args.top_k, config=config)
    else:
        use_case = SerialHybridRetrieveUseCase(sparse_retriever, dense_retriever)
        res = use_case.retrieve("test query", top_k=args.top_k)

    print(f"\n--- Hybrid Results (Mode: {args.mode}, Fusion: {res.fusion_method_used}) ---")
    for i, doc in enumerate(res.results, 1):
        print(f"{i}. {doc.doc_id} (Score: {doc.score:.4f})")

if __name__ == "__main__":
    main()
