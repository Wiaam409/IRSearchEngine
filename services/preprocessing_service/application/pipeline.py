from typing import List

from services.preprocessing_service.domain.interfaces import (
    IPreprocessingPipeline,
    INormalizer,
    ITokenizer,
    IStopwordRemover
)

class PreprocessingPipeline(IPreprocessingPipeline):
    """
    Language-agnostic pipeline that orchestrates the preprocessing steps.
    Follows Dependency Inversion by depending only on the abstract interfaces.
    """
    
    def __init__(
        self,
        normalizer: INormalizer,
        tokenizer: ITokenizer,
        stopword_remover: IStopwordRemover
    ):
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.stopword_remover = stopword_remover

    def process(self, text: str) -> List[str]:
        normalized_text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(normalized_text)
        final_tokens = self.stopword_remover.remove(tokens)
        return final_tokens
