import unittest
from typing import List, Tuple, Dict
from services.evaluation_service.domain.interfaces import IRetrievalModelAdapter
from services.evaluation_service.domain.models import RelevanceJudgment
from services.evaluation_service.application.use_cases import EvaluateModelUseCase
from services.evaluation_service.infrastructure.metric_calculator import MetricCalculator

class MockModelAdapter(IRetrievalModelAdapter):
    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        if query_id == "q1":
            return [("d1", 2.0), ("d2", 1.0)]
        elif query_id == "q2":
            return [("d3", 3.0)]
        return []

class MockQrelLoader:
    def load(self, filepath: str):
        return []

class TestEvaluateModelUseCase(unittest.TestCase):
    def test_evaluation_aggregation(self):
        loader = MockQrelLoader()
        calc = MetricCalculator()
        use_case = EvaluateModelUseCase(loader, calc)
        
        queries = {"q1": "test query", "q2": "another query"}
        qrels = [
            RelevanceJudgment("q1", "d1", 1),
            RelevanceJudgment("q2", "d3", 1)
        ]
        
        report = use_case.evaluate(MockModelAdapter(), "MockModel", queries, qrels, [1])
        
        self.assertEqual(report.model_name, "MockModel")
        
        metrics = {m.metric_name: m.value for m in report.aggregate_metrics}
        self.assertEqual(metrics["MAP"], 1.0)
        self.assertEqual(metrics["P@1"], 1.0)

if __name__ == '__main__':
    unittest.main()
