import os
import shutil
import unittest
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex

class TestVectorIndex(unittest.TestCase):
    def setUp(self):
        self.index = InMemoryVectorIndex()
        self.test_dir = "test_emb_index"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_and_search(self):
        self.index.add("doc1", [1.0, 0.0])
        self.index.add("doc2", [0.0, 1.0])
        self.index.add("doc3", [0.707, 0.707])
        
        # Searching for something similar to doc1
        res = self.index.search([1.0, 0.0], k=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].doc_id, "doc1")
        self.assertAlmostEqual(res[0].score, 1.0)
        self.assertEqual(res[1].doc_id, "doc3")
        self.assertAlmostEqual(res[1].score, 0.707)

    def test_save_load(self):
        self.index.add("d1", [0.5, 0.5])
        self.index.save(self.test_dir)
        
        new_index = InMemoryVectorIndex()
        new_index.load(self.test_dir)
        
        res = new_index.search([0.5, 0.5], k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].doc_id, "d1")
        self.assertAlmostEqual(res[0].score, 0.5)

if __name__ == '__main__':
    unittest.main()
