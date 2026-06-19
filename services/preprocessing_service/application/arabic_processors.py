import re
import unicodedata
from typing import List, Set

from services.preprocessing_service.domain.interfaces import (
    INormalizer,
    ITokenizer,
    IStopwordRemover
)

class ArabicNormalizer(INormalizer):
    def normalize(self, text: str) -> str:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        # Remove Arabic specific punctuation
        text = re.sub(r'[،؛؟]', '', text)
        # Remove common punctuation, keeping Arabic characters and basic whitespace
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        return text.strip()

class ArabicTokenizer(ITokenizer):
    def tokenize(self, text: str) -> List[str]:
        # Split by whitespace
        return [token for token in text.split() if token]

class ArabicStopwordRemover(IStopwordRemover):
    def __init__(self, stopwords: Set[str]):
        self.stopwords = stopwords

    def remove(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords]
