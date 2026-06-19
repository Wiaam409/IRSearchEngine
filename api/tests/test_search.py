import unittest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies.containers import get_bm25_service

class MockRankResult:
    def __init__(self, doc_id, score):
        self.doc_id = doc_id
        self.score = score

class MockBm25Service:
    def rank(self, query, top_k, params=None):
        from services.ranking_service.domain.models import ScoredDocument
        return [ScoredDocument("doc_3", 3.0), ScoredDocument("doc_4", 2.5)][:top_k]

class TestSearchEndpoints(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_bm25_service] = lambda: MockBm25Service()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides = {}

    def test_bm25_search(self):
        payload = {"query": "test query", "top_k": 2}
        response = self.client.post("/search/bm25", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["data"]["results"]), 2)
        self.assertEqual(data["data"]["results"][0]["doc_id"], "mock_doc_1")
        self.assertEqual(data["data"]["query"], "test query")

if __name__ == '__main__':
    unittest.main()
