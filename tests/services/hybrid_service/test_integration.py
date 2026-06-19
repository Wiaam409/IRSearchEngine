import os
import json
import unittest
from services.hybrid_service.domain.models import ScoredDocument, FusionConfig
from services.hybrid_service.infrastructure.fusion_strategies import ScoreFusionStrategy
from services.hybrid_service.domain.interfaces import IRetriever
from services.hybrid_service.application.use_cases import ParallelHybridRetrieveUseCase

class FixtureRetriever(IRetriever):
    def __init__(self, data):
        self.data = data
    def retrieve(self, query, top_k):
        return [ScoredDocument(d["doc_id"], d["score"]) for d in self.data][:top_k]

class TestHybridIntegration(unittest.TestCase):
    def test_end_to_end_fusion(self):
        fixture_path = os.path.join("tests", "fixtures", "mock_rankings.json")
        if not os.path.exists(fixture_path):
            self.skipTest("mock_rankings.json not found")
            
        with open(fixture_path, "r") as f:
            data = json.load(f)
            
        sparse = FixtureRetriever(data["sparse"])
        dense = FixtureRetriever(data["dense"])
        
        fusion = ScoreFusionStrategy()
        config = FusionConfig(sparse_weight=0.5, dense_weight=0.5)
        
        use_case = ParallelHybridRetrieveUseCase([sparse, dense], fusion)
        res = use_case.retrieve("q", top_k=3, config=config)
        
        self.assertEqual(len(res.results), 3)

if __name__ == '__main__':
    unittest.main()
