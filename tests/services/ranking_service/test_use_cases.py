import unittest
from typing import List, Tuple
from services.ranking_service.domain.interfaces import IQueryProcessor, IIndexReader, IBm25Scorer
from services.ranking_service.application.use_cases import Bm25RankUseCase

class MockQueryProcessor(IQueryProcessor):
    def process(self, query: str) -> List[str]:
        return query.split()

class MockIndexReader(IIndexReader):
    def get_posting_list(self, term: str) -> List[Tuple[str, int]]:
        if term == "hello":
            return [("doc1", 2), ("doc2", 1)]
        return []
    def get_doc_length(self, doc_id: str) -> int:
        return 100
    def get_avgdl(self) -> float:
        return 100.0
    def get_idf(self, term: str) -> float:
        return 1.5 if term == "hello" else 0.0

class MockScorer(IBm25Scorer):
    def compute_score(self, tf: int, doc_len: int, avgdl: float, idf: float, k1: float, b: float) -> float:
        return float(tf) * 2.0

class TestBm25RankUseCase(unittest.TestCase):
    def setUp(self):
        self.use_case = Bm25RankUseCase(MockQueryProcessor(), MockIndexReader(), MockScorer())

    def test_ranking_order(self):
        results = self.use_case.rank("hello")
        self.assertEqual(len(results), 2)
        # doc1 has tf=2 -> score=4.0
        # doc2 has tf=1 -> score=2.0
        self.assertEqual(results[0].doc_id, "doc1")
        self.assertEqual(results[0].score, 4.0)
        self.assertEqual(results[1].doc_id, "doc2")
        self.assertEqual(results[1].score, 2.0)

if __name__ == '__main__':
    unittest.main()
