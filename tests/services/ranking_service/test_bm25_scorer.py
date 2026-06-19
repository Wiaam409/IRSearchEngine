import unittest
from services.ranking_service.infrastructure.bm25_scorer import Bm25Scorer

class TestBm25Scorer(unittest.TestCase):
    def setUp(self):
        self.scorer = Bm25Scorer()

    def test_compute_score_standard(self):
        # tf=2, doc_len=100, avgdl=100, idf=1.5, k1=1.2, b=0.75
        score = self.scorer.compute_score(2, 100, 100.0, 1.5, 1.2, 0.75)
        self.assertAlmostEqual(score, 2.0625)

    def test_zero_avgdl(self):
        score = self.scorer.compute_score(2, 100, 0.0, 1.5, 1.2, 0.75)
        self.assertEqual(score, 0.0)

if __name__ == '__main__':
    unittest.main()
