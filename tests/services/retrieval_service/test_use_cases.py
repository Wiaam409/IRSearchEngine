import unittest
from typing import List, Tuple, Dict
from services.retrieval_service.domain.interfaces import IQueryProcessor, IIndexReader, IScorer
from services.retrieval_service.application.use_cases import TfidfRetrieveUseCase
from services.indexing_service.domain.models import CollectionStatistics

class MockQueryProcessor(IQueryProcessor):
    def process(self, query: str) -> List[str]:
        if not query:
            return []
        return query.split()

class MockIndexReader(IIndexReader):
    def get_term_posting_list(self, term: str) -> List[Tuple[str, int]]:
        if term == "hello":
            return [("doc1", 2), ("doc2", 1)]
        return []

    def get_doc_norm(self, doc_id: str) -> float:
        return 1.0

    def get_idf(self, term: str) -> float:
        return 1.5 if term == "hello" else 0.0

    def get_collection_stats(self) -> CollectionStatistics:
        return CollectionStatistics()

class MockScorer(IScorer):
    def score(self, query_vector: Dict[str, float], doc_id: str, dot_product: float) -> float:
        return dot_product

class TestTfidfRetrieveUseCase(unittest.TestCase):
    def setUp(self):
        self.query_processor = MockQueryProcessor()
        self.index_reader = MockIndexReader()
        self.scorer = MockScorer()
        self.use_case = TfidfRetrieveUseCase(self.query_processor, self.index_reader, self.scorer)

    def test_retrieval_ranking(self):
        # Query "hello" -> idf=1.5, query_weight=1.5
        # doc1 tf=2 -> doc_term_weight=3.0 -> dot_product = 1.5 * 3.0 = 4.5
        # doc2 tf=1 -> doc_term_weight=1.5 -> dot_product = 1.5 * 1.5 = 2.25
        result = self.use_case.retrieve("hello")
        
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0].doc_id, "doc1")
        self.assertEqual(result.results[0].score, 4.5)
        self.assertEqual(result.results[1].doc_id, "doc2")
        self.assertEqual(result.results[1].score, 2.25)

    def test_empty_query(self):
        result = self.use_case.retrieve("")
        self.assertEqual(len(result.results), 0)

if __name__ == '__main__':
    unittest.main()
