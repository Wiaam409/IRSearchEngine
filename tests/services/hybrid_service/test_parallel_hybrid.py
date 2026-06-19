import unittest
from services.hybrid_service.domain.models import ScoredDocument, FusionConfig
from services.hybrid_service.domain.interfaces import IRetriever, IFusionStrategy
from services.hybrid_service.application.use_cases import ParallelHybridRetrieveUseCase

class MockRetriever(IRetriever):
    def __init__(self, prefix: str):
        self.prefix = prefix
    def retrieve(self, query: str, top_k: int):
        return [ScoredDocument(f"{self.prefix}_1", 1.0)]

class MockFusion(IFusionStrategy):
    def fuse(self, ranked_lists, config):
        return [ScoredDocument("fused", 100.0)]

class TestParallelHybrid(unittest.TestCase):
    def test_parallel_execution(self):
        r1 = MockRetriever("r1")
        r2 = MockRetriever("r2")
        fusion = MockFusion()
        
        use_case = ParallelHybridRetrieveUseCase([r1, r2], fusion)
        res = use_case.retrieve("q", top_k=1)
        
        self.assertEqual(len(res.results), 1)
        self.assertEqual(res.results[0].doc_id, "fused")
        self.assertEqual(res.fusion_method_used, "MockFusion")

if __name__ == '__main__':
    unittest.main()
