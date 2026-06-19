import os
import json
import shutil
import unittest
from services.preprocessing_service.application.pipeline import PreprocessingPipeline
from services.preprocessing_service.application.arabic_processors import ArabicNormalizer, ArabicTokenizer, ArabicStopwordRemover
from services.ranking_service.infrastructure.bm25_index_reader import Bm25IndexReader
from services.ranking_service.infrastructure.query_processor_adapter import QueryProcessorAdapter
from services.ranking_service.infrastructure.bm25_scorer import Bm25Scorer
from services.ranking_service.application.use_cases import Bm25RankUseCase
from services.ranking_service.domain.models import Bm25Parameters

class TestBm25Integration(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_bm25_integration"
        os.makedirs(self.test_dir, exist_ok=True)
        
        with open(os.path.join(self.test_dir, "collection_statistics.json"), "w", encoding="utf-8") as f:
            json.dump({"total_documents": 2, "total_tokens": 10, "average_document_length": 5.0}, f)
            
        with open(os.path.join(self.test_dir, "inverted_index.json"), "w", encoding="utf-8") as f:
            json.dump({
                "تعليم": {"df": 2, "cf": 3, "postings": [{"doc_id": "1", "tf": 2}, {"doc_id": "2", "tf": 1}]},
                "مدرسة": {"df": 1, "cf": 1, "postings": [{"doc_id": "1", "tf": 1}]}
            }, f)

        with open(os.path.join(self.test_dir, "document_statistics.json"), "w", encoding="utf-8") as f:
            json.dump({"1": {"length": 7}, "2": {"length": 3}}, f)

        pipeline = PreprocessingPipeline(ArabicNormalizer(), ArabicTokenizer(), ArabicStopwordRemover(set()))
        self.ranking_service = Bm25RankUseCase(
            query_processor=QueryProcessorAdapter(pipeline),
            index_reader=Bm25IndexReader(self.test_dir),
            scorer=Bm25Scorer()
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_arabic_query(self):
        results = self.ranking_service.rank("تعليم", params=Bm25Parameters(k1=1.2, b=0.75))
        self.assertEqual(len(results), 2)
        
        self.assertTrue(results[0].score > 0)
        self.assertTrue(results[1].score > 0)
        
        doc_ids = {r.doc_id for r in results}
        self.assertEqual(doc_ids, {"1", "2"})

if __name__ == '__main__':
    unittest.main()
