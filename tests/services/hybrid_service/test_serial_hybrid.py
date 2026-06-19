import unittest
from services.hybrid_service.domain.models import ScoredDocument
from services.hybrid_service.domain.interfaces import IRetriever
from services.hybrid_service.application.use_cases import SerialHybridRetrieveUseCase

class MockSparse(IRetriever):
    def retrieve(self, query: str, top_k: int):
        return [ScoredDocument("d1", 10.0), ScoredDocument("d2", 5.0), ScoredDocument("d3", 1.0)][:top_k]

class MockDense(IRetriever):
    def retrieve(self, query: str, top_k: int):
        return [ScoredDocument("d3", 0.9), ScoredDocument("d1", 0.5), ScoredDocument("d4", 0.1)][:top_k]

class TestSerialHybrid(unittest.TestCase):
    def test_serial_execution(self):
        sparse = MockSparse()
        dense = MockDense()
        use_case = SerialHybridRetrieveUseCase(sparse, dense)
        
        res = use_case.retrieve("q", top_k=2, candidate_multiplier=2)
        
        self.assertEqual(len(res.results), 2)
        self.assertEqual(res.results[0].doc_id, "d3")
        self.assertEqual(res.results[1].doc_id, "d1")

if __name__ == '__main__':
    unittest.main()
