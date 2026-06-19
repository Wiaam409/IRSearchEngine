from abc import ABC, abstractmethod
from typing import List, Optional
from .models import PostingList, DocumentStatistics, CollectionStatistics

class IIndexWriter(ABC):
    @abstractmethod
    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        """Add a document and its tokens to the index."""
        pass

    @abstractmethod
    def commit(self) -> None:
        """Commit the changes (e.g., to disk)."""
        pass

class IIndexReader(ABC):
    @abstractmethod
    def get_postings(self, term: str) -> Optional[PostingList]:
        """Retrieve the posting list for a given term."""
        pass

    @abstractmethod
    def get_document_statistics(self, doc_id: str) -> Optional[DocumentStatistics]:
        """Retrieve statistics for a specific document."""
        pass

    @abstractmethod
    def get_collection_statistics(self) -> CollectionStatistics:
        """Retrieve overall collection statistics."""
        pass
