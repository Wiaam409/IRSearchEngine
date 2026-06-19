import unittest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies.containers import get_evaluation_service
from services.evaluation_service.domain.models import EvaluationReport

class MockEvaluationService:
    def evaluate(self, model, model_name, queries, qrels, top_k_list):
        from services.evaluation_service.domain.models import EvaluationReport, MetricResult
        report = EvaluationReport(model_name=model_name)
        report.aggregate_metrics = [MetricResult("MAP", 0.85), MetricResult("nDCG@10", 0.90)]
        return report

from unittest.mock import patch

class TestEvaluateEndpoint(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_evaluation_service] = lambda: MockEvaluationService()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides = {}

    @patch('api.routers.evaluate.get_bm25_service')
    def test_evaluate(self, mock_get_bm25):
        mock_get_bm25.return_value = None
        payload = {"model": "bm25", "top_k_list": [5]}
        response = self.client.post("/evaluate", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["data"]["metrics"]["MAP"], 0.85)

if __name__ == '__main__':
    unittest.main()
