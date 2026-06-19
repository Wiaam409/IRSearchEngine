from abc import ABC, abstractmethod
from typing import List, Set

class INormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalizes the input text (e.g., lowercasing, unicode normalization)."""
        pass

class ITokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert raw text into a list of tokens."""
        pass

class IStopwordRemover(ABC):
    @abstractmethod
    def remove(self, tokens: List[str]) -> List[str]:
        """Removes stopwords from a list of tokens."""
        pass

class IPreprocessingPipeline(ABC):
    @abstractmethod
    def process(self, text: str) -> List[str]:
        """Executes the full preprocessing pipeline on the input text."""
        pass
