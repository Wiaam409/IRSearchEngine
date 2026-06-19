import unittest
from fastapi.testclient import TestClient
from api.main import app

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        from api.dependencies.containers import get_bm25_service
        app.dependency_overrides[get_bm25_service] = lambda: None
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides = {}

    def test_missing_query(self):
        payload = {"top_k": 5}
        response = self.client.post("/search/bm25", json=payload)
        
        self.assertEqual(response.status_code, 422)

    def test_invalid_hybrid_model(self):
        payload = {"query": "test", "sparse_model": "invalid_model"}
        response = self.client.post("/search/hybrid/parallel", json=payload)
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid sparse_model")

if __name__ == '__main__':
    unittest.main()
