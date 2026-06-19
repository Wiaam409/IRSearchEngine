import os
import json
import unittest
from typing import List, Tuple
from services.evaluation_service.domain.interfaces import IRetrievalModelAdapter
from services.evaluation_service.domain.models import RelevanceJudgment
from services.evaluation_service.application.use_cases import EvaluateModelUseCase
from services.evaluation_service.infrastructure.metric_calculator import MetricCalculator
from services.evaluation_service.infrastructure.qrel_loader import QrelFileLoader

class DummyAdapter(IRetrievalModelAdapter):
    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        return [("1", 5.0), ("2", 4.0), ("3", 1.0)][:top_k]

class TestEvaluationIntegration(unittest.TestCase):
    def setUp(self):
        self.qrel_path = "test_int_qrels.txt"
        with open(self.qrel_path, "w") as f:
            f.write("q1 0 1 1\n")
            f.write("q1 0 2 0\n")
            f.write("q1 0 3 1\n")

    def tearDown(self):
        if os.path.exists(self.qrel_path):
            os.remove(self.qrel_path)

    def test_evaluate_dummy_adapter(self):
        loader = QrelFileLoader()
        calc = MetricCalculator()
        use_case = EvaluateModelUseCase(loader, calc)
        
        qrels = loader.load(self.qrel_path)
        queries = {"q1": "some text"}
        
        report = use_case.evaluate(DummyAdapter(), "Dummy", queries, qrels, [2])
        
        metrics = {m.metric_name: m.value for m in report.aggregate_metrics}
        
        self.assertEqual(metrics["P@2"], 0.5)
        self.assertEqual(metrics["Recall@2"], 0.5)

if __name__ == '__main__':
    unittest.main()
