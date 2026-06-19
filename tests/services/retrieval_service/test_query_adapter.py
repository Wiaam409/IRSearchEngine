import unittest
from typing import List
from services.preprocessing_service.domain.interfaces import IPreprocessingPipeline
from services.retrieval_service.infrastructure.query_processor_adapter import QueryProcessorAdapter

class MockPipeline(IPreprocessingPipeline):
    def process(self, text: str) -> List[str]:
        return ["mocked", "tokens"]

class TestQueryProcessorAdapter(unittest.TestCase):
    def test_delegation(self):
        pipeline = MockPipeline()
        adapter = QueryProcessorAdapter(pipeline)
        
        result = adapter.process("some random query")
        self.assertEqual(result, ["mocked", "tokens"])

if __name__ == '__main__':
    unittest.main()
