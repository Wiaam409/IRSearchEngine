from typing import List
from services.retrieval_service.domain.interfaces import IQueryProcessor
from services.preprocessing_service.domain.interfaces import IPreprocessingPipeline

class QueryProcessorAdapter(IQueryProcessor):
    """
    Adapter that uses the existing preprocessing pipeline to process the query.
    Ensures that queries undergo the exact same normalization, tokenization, 
    and stopword removal as documents.
    """
    def __init__(self, pipeline: IPreprocessingPipeline):
        self.pipeline = pipeline

    def process(self, query: str) -> List[str]:
        return self.pipeline.process(query)
