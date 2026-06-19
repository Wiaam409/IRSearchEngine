from abc import ABC, abstractmethod
from typing import List, Tuple
from services.ranking_service.domain.models import ScoredDocument, Bm25Parameters

class IRankingService(ABC):
    @abstractmethod
    def rank(self, query: str, top_k: int = 10, params: Bm25Parameters = None) -> List[ScoredDocument]:
        """Ranks documents based on a query."""
        pass

class IBm25Scorer(ABC):
    @abstractmethod
    def compute_score(self, tf: int, doc_len: int, avgdl: float, idf: float, k1: float, b: float) -> float:
        """Computes the BM25 contribution for a single term-document pair."""
        pass

class IIndexReader(ABC):
    @abstractmethod
    def get_posting_list(self, term: str) -> List[Tuple[str, int]]:
        """Returns a list of (doc_id, tf) for a term."""
        pass

    @abstractmethod
    def get_doc_length(self, doc_id: str) -> int:
        """Returns the length of a document."""
        pass

    @abstractmethod
    def get_avgdl(self) -> float:
        """Returns the average document length."""
        pass

    @abstractmethod
    def get_idf(self, term: str) -> float:
        """Returns the precomputed IDF for a term."""
        pass

class IQueryProcessor(ABC):
    @abstractmethod
    def process(self, query: str) -> List[str]:
        """Convert a raw query string into a list of normalized tokens."""
        pass
