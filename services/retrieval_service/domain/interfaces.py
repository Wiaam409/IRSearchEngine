from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from services.retrieval_service.domain.models import ScoredDocument, RetrievalResult
from services.indexing_service.domain.models import CollectionStatistics

class IRetrievalService(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        pass

class IQueryProcessor(ABC):
    @abstractmethod
    def process(self, query: str) -> List[str]:
        """Convert a raw query string into a list of normalized tokens."""
        pass

class IIndexReader(ABC):
    @abstractmethod
    def get_term_posting_list(self, term: str) -> List[Tuple[str, int]]:
        """Returns a list of (doc_id, term_frequency)."""
        pass

    @abstractmethod
    def get_doc_norm(self, doc_id: str) -> float:
        """Returns the precomputed TF-IDF L2 norm for the document."""
        pass

    @abstractmethod
    def get_idf(self, term: str) -> float:
        """Returns the precomputed IDF for the term."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> CollectionStatistics:
        pass

class IScorer(ABC):
    @abstractmethod
    def score(self, query_vector: Dict[str, float], doc_id: str) -> float:
        """Computes the final cosine similarity score for a document given a query vector."""
        pass
