import unittest
from typing import List
from services.embeddings_service.domain.interfaces import IEmbeddingModel
from services.embeddings_service.domain.models import DenseVector

class MockEmbeddingModel(IEmbeddingModel):
    def encode(self, text: str) -> DenseVector:
        return [0.1, 0.2, 0.3]

    def encode_batch(self, texts: List[str]) -> List[DenseVector]:
        return [[0.1, 0.2, 0.3] for _ in texts]

class TestEmbeddingModel(unittest.TestCase):
    def test_encode(self):
        model = MockEmbeddingModel()
        res = model.encode("test")
        self.assertEqual(res, [0.1, 0.2, 0.3])

    def test_encode_batch(self):
        model = MockEmbeddingModel()
        res = model.encode_batch(["t1", "t2"])
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], [0.1, 0.2, 0.3])

if __name__ == '__main__':
    unittest.main()
