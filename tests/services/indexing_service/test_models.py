import unittest
from services.indexing_service.domain.models import Posting, PostingList, DocumentStatistics, CollectionStatistics

class TestIndexingModels(unittest.TestCase):
    def test_posting_creation(self):
        p = Posting(doc_id="1", tf=5)
        self.assertEqual(p.doc_id, "1")
        self.assertEqual(p.tf, 5)

    def test_posting_list(self):
        pl = PostingList()
        pl.postings.append(Posting("1", 2))
        pl.df = 1
        pl.cf = 2
        self.assertEqual(len(pl.postings), 1)
        self.assertEqual(pl.df, 1)

    def test_collection_statistics(self):
        cs = CollectionStatistics()
        cs.total_documents = 10
        cs.total_tokens = 100
        cs.average_document_length = 10.0
        self.assertEqual(cs.total_documents, 10)

if __name__ == '__main__':
    unittest.main()
