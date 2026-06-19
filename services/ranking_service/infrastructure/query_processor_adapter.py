from typing import List
from services.ranking_service.domain.interfaces import IQueryProcessor
from services.preprocessing_service.domain.interfaces import IPreprocessingPipeline

class QueryProcessorAdapter(IQueryProcessor):
    def __init__(self, pipeline: IPreprocessingPipeline):
        self.pipeline = pipeline

    def process(self, query: str) -> List[str]:
        return self.pipeline.process(query)
