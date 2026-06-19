import re
import unicodedata
from typing import List, Set

from services.preprocessing_service.domain.interfaces import (
    INormalizer,
    ITokenizer,
    IStopwordRemover
)

class BengaliNormalizer(INormalizer):
    def normalize(self, text: str) -> str:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        # Remove common punctuation, keeping Bengali characters and basic whitespace
        text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
        return text.strip()

class BengaliTokenizer(ITokenizer):
    def tokenize(self, text: str) -> List[str]:
        # Split by whitespace
        return [token for token in text.split() if token]

class BengaliStopwordRemover(IStopwordRemover):
    def __init__(self, stopwords: Set[str]):
        self.stopwords = stopwords

    def remove(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords]
