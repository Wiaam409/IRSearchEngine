import unittest
from typing import List, Optional
from services.indexing_service.application.use_cases import BuildIndexUseCase, UpdateIndexUseCase
from services.indexing_service.infrastructure.memory_index import InMemoryIndexWriter

class TestUseCases(unittest.TestCase):
    def setUp(self):
        self.writer = InMemoryIndexWriter()
        self.build_uc = BuildIndexUseCase(self.writer)
        self.update_uc = UpdateIndexUseCase(self.writer)

    def test_build_index(self):
        docs = [
            ("1", ["apple", "banana", "apple"]),
            ("2", ["banana", "orange"])
        ]
        self.build_uc.execute(docs)
        
        # Verify stats
        self.assertEqual(self.writer.collection_stats.total_documents, 2)
        self.assertEqual(self.writer.collection_stats.total_tokens, 5)
        self.assertEqual(self.writer.collection_stats.average_document_length, 2.5)

        # Verify lexicon for "apple"
        apple_postings = self.writer.lexicon["apple"]
        self.assertEqual(apple_postings.df, 1)
        self.assertEqual(apple_postings.cf, 2)
        self.assertEqual(apple_postings.postings[0].tf, 2)

        # Verify lexicon for "banana"
        banana_postings = self.writer.lexicon["banana"]
        self.assertEqual(banana_postings.df, 2)
        self.assertEqual(banana_postings.cf, 2)

    def test_update_index(self):
        self.update_uc.execute("3", ["grape"])
        self.assertEqual(self.writer.collection_stats.total_documents, 1)
        self.assertEqual(self.writer.lexicon["grape"].df, 1)

if __name__ == '__main__':
    unittest.main()
