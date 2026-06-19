import unittest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies.containers import get_refinement_service
from services.query_refinement_service.domain.models import RefinedQuery

class MockRefineService:
    def refine(self, query: str):
        return RefinedQuery(
            original_text=query,
            refined_text=f"{query} OR synonym",
            suggestions=["synonym"],
            corrections={"bad": "good"}
        )

class TestRefineEndpoint(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_refinement_service] = lambda: MockRefineService()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides = {}

    def test_refine(self):
        payload = {"query": "test query", "spell_check": True}
        response = self.client.post("/refine", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["data"]["original_query"], "test query")
        self.assertEqual(data["data"]["refined_query"], "test query OR synonym")
        self.assertIn("synonym", data["data"]["suggestions"])

if __name__ == '__main__':
    unittest.main()
