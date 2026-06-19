import unittest
from typing import List
from services.preprocessing_service.application.pipeline import PreprocessingPipeline
from services.preprocessing_service.domain.interfaces import INormalizer, ITokenizer, IStopwordRemover

class MockNormalizer(INormalizer):
    def normalize(self, text: str) -> str:
        return text.lower()

class MockTokenizer(ITokenizer):
    def tokenize(self, text: str) -> List[str]:
        return text.split()

class MockStopwordRemover(IStopwordRemover):
    def remove(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t != "stop"]

class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = PreprocessingPipeline(
            normalizer=MockNormalizer(),
            tokenizer=MockTokenizer(),
            stopword_remover=MockStopwordRemover()
        )

    def test_pipeline_execution_order(self):
        text = "Hello STOP World"
        expected = ["hello", "world"]
        result = self.pipeline.process(text)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
