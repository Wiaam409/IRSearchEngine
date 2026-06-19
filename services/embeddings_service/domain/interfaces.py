from abc import ABC, abstractmethod
from typing import List
from services.embeddings_service.domain.models import DenseVector, VectorSearchResult

class IEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, text: str) -> DenseVector:
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[DenseVector]:
        pass

class IVectorIndex(ABC):
    @abstractmethod
    def add(self, doc_id: str, vector: DenseVector):
        pass

    @abstractmethod
    def search(self, query_vector: DenseVector, k: int) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

class IDenseRetrievalService(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[VectorSearchResult]:
        pass
