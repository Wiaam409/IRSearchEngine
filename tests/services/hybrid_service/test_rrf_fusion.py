import unittest
from services.hybrid_service.domain.models import ScoredDocument, FusionConfig
from services.hybrid_service.infrastructure.fusion_strategies import RrFusionStrategy

class TestRrfFusion(unittest.TestCase):
    def test_rrf_fusion(self):
        config = FusionConfig(rrf_k_constant=60)
        strategy = RrFusionStrategy()
        
        list1 = [ScoredDocument("doc1", 10.0), ScoredDocument("doc2", 5.0)]
        list2 = [ScoredDocument("doc2", 0.9), ScoredDocument("doc1", 0.8)]
        
        res = strategy.fuse([list1, list2], config)
        self.assertEqual(len(res), 2)
        
        self.assertAlmostEqual(res[0].score, res[1].score)

if __name__ == '__main__':
    unittest.main()
