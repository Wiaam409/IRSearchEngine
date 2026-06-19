import os
import shutil
import unittest
from services.indexing_service.infrastructure.json_index import JsonIndexWriter, JsonIndexReader

class TestJsonPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_index_output"
        self.writer = JsonIndexWriter(self.test_dir)
        self.reader = JsonIndexReader(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_write_and_read_index(self):
        # 1. Write
        self.writer.add_document("doc1", ["hello", "world", "hello"])
        self.writer.add_document("doc2", ["hello", "test"])
        self.writer.commit()

        # Verify files exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "inverted_index.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "document_statistics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "collection_statistics.json")))

        # 2. Read
        hello_postings = self.reader.get_postings("hello")
        self.assertIsNotNone(hello_postings)
        self.assertEqual(hello_postings.df, 2)
        self.assertEqual(hello_postings.cf, 3)

        doc_stats = self.reader.get_document_statistics("doc1")
        self.assertEqual(doc_stats.length, 3)

        col_stats = self.reader.get_collection_statistics()
        self.assertEqual(col_stats.total_documents, 2)
        self.assertEqual(col_stats.total_tokens, 5)

if __name__ == '__main__':
    unittest.main()
