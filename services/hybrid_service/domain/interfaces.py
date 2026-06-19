from abc import ABC, abstractmethod
from typing import List
from services.hybrid_service.domain.models import ScoredDocument, HybridResult, FusionConfig

class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        pass

class IFusionStrategy(ABC):
    @abstractmethod
    def fuse(self, ranked_lists: List[List[ScoredDocument]], config: FusionConfig) -> List[ScoredDocument]:
        pass

class IHybridRetrievalService(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> HybridResult:
        pass
