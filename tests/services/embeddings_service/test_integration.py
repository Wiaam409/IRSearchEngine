import os
import json
import shutil
import unittest
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
from services.embeddings_service.application.use_cases import BuildDenseIndexUseCase, DenseRetrieveUseCase
from tests.services.embeddings_service.test_embedding_model import MockEmbeddingModel

class TestEmbeddingsIntegration(unittest.TestCase):
    def setUp(self):
        self.fixture_path = os.path.join("tests", "fixtures", "documents.json")
        self.test_dir = "test_int_emb_index"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_end_to_end_mocked(self):
        if not os.path.exists(self.fixture_path):
            self.skipTest("documents.json fixture not found")
            
        with open(self.fixture_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        model = MockEmbeddingModel()
        index = InMemoryVectorIndex()
        
        build_uc = BuildDenseIndexUseCase(model, index, batch_size=5)
        build_uc.build(docs, save_path=self.test_dir)
        
        new_index = InMemoryVectorIndex()
        new_index.load(self.test_dir)
        
        ret_uc = DenseRetrieveUseCase(model, new_index)
        res = ret_uc.retrieve("سيارة", top_k=2)
        
        self.assertEqual(len(res), 2)

if __name__ == '__main__':
    unittest.main()
