import os
import json
import shutil
import math
import unittest
from services.ranking_service.infrastructure.bm25_index_reader import Bm25IndexReader

class TestBm25IndexReader(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_bm25_index_output"
        os.makedirs(self.test_dir, exist_ok=True)
        
        with open(os.path.join(self.test_dir, "collection_statistics.json"), "w") as f:
            json.dump({"total_documents": 2, "total_tokens": 5, "average_document_length": 2.5}, f)
            
        with open(os.path.join(self.test_dir, "inverted_index.json"), "w") as f:
            json.dump({
                "apple": {"df": 1, "cf": 2, "postings": [{"doc_id": "1", "tf": 2}]},
                "banana": {"df": 2, "cf": 2, "postings": [{"doc_id": "1", "tf": 1}, {"doc_id": "2", "tf": 1}]}
            }, f)

        with open(os.path.join(self.test_dir, "document_statistics.json"), "w") as f:
            json.dump({"1": {"length": 3}, "2": {"length": 1}}, f)

        self.reader = Bm25IndexReader(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_idf_computation(self):
        expected_apple_idf = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1)
        self.assertAlmostEqual(self.reader.get_idf("apple"), expected_apple_idf)

    def test_doc_length(self):
        self.assertEqual(self.reader.get_doc_length("1"), 3)
        self.assertEqual(self.reader.get_doc_length("2"), 1)

    def test_avgdl(self):
        self.assertEqual(self.reader.get_avgdl(), 2.5)

if __name__ == '__main__':
    unittest.main()
