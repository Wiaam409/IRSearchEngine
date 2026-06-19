import os
import shutil
import unittest
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
from services.embeddings_service.application.use_cases import BuildDenseIndexUseCase
from tests.services.embeddings_service.test_embedding_model import MockEmbeddingModel

class TestBuildIndexUseCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_build_emb_index"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_build(self):
        model = MockEmbeddingModel()
        index = InMemoryVectorIndex()
        use_case = BuildDenseIndexUseCase(model, index, batch_size=2)
        
        docs = {"1": "text1", "2": "text2", "3": "text3"}
        use_case.build(docs, save_path=self.test_dir)
        
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "index.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "doc_ids.json")))
        
        res = index.search([0.1, 0.2, 0.3], k=3)
        self.assertEqual(len(res), 3)

if __name__ == '__main__':
    unittest.main()
