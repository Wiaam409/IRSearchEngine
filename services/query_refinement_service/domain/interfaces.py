from abc import ABC, abstractmethod
from typing import List, Dict
from services.query_refinement_service.domain.models import RefinedQuery

class IQueryRefiner(ABC):
    @abstractmethod
    def refine(self, query: str) -> RefinedQuery:
        pass

class ISynonymProvider(ABC):
    @abstractmethod
    def get_synonyms(self, term: str) -> List[str]:
        pass

class ISpellChecker(ABC):
    @abstractmethod
    def correct(self, term: str) -> str:
        pass

class IQueryHistory(ABC):
    @abstractmethod
    def add_query(self, query: str):
        pass

    @abstractmethod
    def get_recent_queries(self, top_k: int) -> List[str]:
        pass
