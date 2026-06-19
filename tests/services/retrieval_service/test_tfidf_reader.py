import os
import json
import shutil
import math
import unittest
from services.retrieval_service.infrastructure.tfidf_index_reader import TfidfIndexReader

class TestTfidfIndexReader(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_retrieval_index_output"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # collection_statistics
        with open(os.path.join(self.test_dir, "collection_statistics.json"), "w") as f:
            json.dump({"total_documents": 2, "total_tokens": 5, "average_document_length": 2.5}, f)
            
        # inverted_index
        with open(os.path.join(self.test_dir, "inverted_index.json"), "w") as f:
            json.dump({
                "apple": {"df": 1, "cf": 2, "postings": [{"doc_id": "1", "tf": 2}]},
                "banana": {"df": 2, "cf": 2, "postings": [{"doc_id": "1", "tf": 1}, {"doc_id": "2", "tf": 1}]}
            }, f)

        # document_statistics
        with open(os.path.join(self.test_dir, "document_statistics.json"), "w") as f:
            json.dump({"1": {"length": 3}, "2": {"length": 1}}, f)

        self.reader = TfidfIndexReader(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_idf_computation(self):
        # Formula: log((N - df + 0.5) / (df + 0.5) + 1)
        expected_apple_idf = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1)
        self.assertAlmostEqual(self.reader.get_idf("apple"), expected_apple_idf)

    def test_doc_norm_computation(self):
        idf_apple = self.reader.get_idf("apple")
        idf_banana = self.reader.get_idf("banana")
        
        # Doc 1 has 2 apple, 1 banana
        expected_doc1_norm = math.sqrt((2 * idf_apple)**2 + (1 * idf_banana)**2)
        self.assertAlmostEqual(self.reader.get_doc_norm("1"), expected_doc1_norm)

if __name__ == '__main__':
    unittest.main()
