import unittest
from services.hybrid_service.domain.models import ScoredDocument, FusionConfig
from services.hybrid_service.infrastructure.fusion_strategies import ScoreFusionStrategy

class TestScoreFusion(unittest.TestCase):
    def test_score_fusion_normalization(self):
        config = FusionConfig(sparse_weight=0.5, dense_weight=0.5)
        strategy = ScoreFusionStrategy()
        
        list1 = [ScoredDocument("doc1", 100.0), ScoredDocument("doc2", 50.0), ScoredDocument("doc3", 0.0)]
        list2 = [ScoredDocument("doc3", 1.0), ScoredDocument("doc2", 0.5), ScoredDocument("doc1", 0.0)]
        
        res = strategy.fuse([list1, list2], config)
        self.assertEqual(len(res), 3)
        self.assertAlmostEqual(res[0].score, 0.5)
        self.assertAlmostEqual(res[1].score, 0.5)
        self.assertAlmostEqual(res[2].score, 0.5)

if __name__ == '__main__':
    unittest.main()
