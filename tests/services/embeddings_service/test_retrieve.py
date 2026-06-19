import unittest
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
from services.embeddings_service.application.use_cases import DenseRetrieveUseCase
from tests.services.embeddings_service.test_embedding_model import MockEmbeddingModel

class TestRetrieveUseCase(unittest.TestCase):
    def test_retrieve(self):
        model = MockEmbeddingModel()
        index = InMemoryVectorIndex()
        index.add("d1", [0.1, 0.2, 0.3])
        
        use_case = DenseRetrieveUseCase(model, index)
        res = use_case.retrieve("query", top_k=1)
        
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].doc_id, "d1")

if __name__ == '__main__':
    unittest.main()
