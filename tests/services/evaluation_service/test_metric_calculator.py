import unittest
import math
from services.evaluation_service.infrastructure.metric_calculator import MetricCalculator

class TestMetricCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = MetricCalculator()

    def test_precision_at_k(self):
        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = ["d2", "d4", "d5"]
        self.assertEqual(self.calc.precision_at_k(retrieved, relevant, 2), 0.5)

    def test_recall_at_k(self):
        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = ["d2", "d4", "d5"]
        self.assertAlmostEqual(self.calc.recall_at_k(retrieved, relevant, 4), 2/3)

    def test_average_precision(self):
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = ["d1", "d3", "d5", "d6"]
        expected = (1.0 + 2/3 + 3/5) / 4
        self.assertAlmostEqual(self.calc.average_precision(retrieved, relevant), expected)

    def test_ndcg_at_k(self):
        retrieved = ["d1", "d2", "d3"]
        graded = {"d1": 3, "d2": 2, "d3": 0, "d4": 3}
        dcg = 3 + 2 / math.log2(3)
        idcg = 3 + 3 / math.log2(3) + 1
        expected_ndcg = dcg / idcg
        self.assertAlmostEqual(self.calc.ndcg_at_k(retrieved, graded, 3), expected_ndcg)

if __name__ == '__main__':
    unittest.main()
