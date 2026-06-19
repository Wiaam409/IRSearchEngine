import re
import unicodedata
from typing import List, Set

from services.preprocessing_service.domain.interfaces import (
    INormalizer,
    ITokenizer,
    IStopwordRemover
)

# Common English stopwords
ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "where", "when", "why", "how",
    "not", "no", "nor", "as", "if", "then", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "am", "any", "because",
    "before", "below", "between", "both", "each", "few", "further", "get",
    "got", "here", "into", "more", "most", "much", "must", "myself", "new",
    "now", "only", "other", "out", "over", "own", "same", "so", "some",
    "such", "there", "through", "under", "until", "up", "while", "down",
    "during", "s", "t", "don", "didn", "doesn", "hasn", "haven", "isn",
    "wasn", "weren", "won", "wouldn", "couldn", "shouldn",
}


class EnglishNormalizer(INormalizer):
    def normalize(self, text: str) -> str:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        # Lowercase
        text = text.lower()
        # Remove non-alphanumeric characters (keep letters, digits, whitespace)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class EnglishTokenizer(ITokenizer):
    def tokenize(self, text: str) -> List[str]:
        # Split by whitespace
        return [token for token in text.split() if token]


class EnglishStopwordRemover(IStopwordRemover):
    def __init__(self, stopwords: Set[str] = None):
        self.stopwords = stopwords if stopwords is not None else ENGLISH_STOPWORDS

    def remove(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords]
