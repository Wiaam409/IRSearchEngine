from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from services.evaluation_service.domain.models import RelevanceJudgment, EvaluationReport

class IQrelLoader(ABC):
    @abstractmethod
    def load(self, filepath: str) -> List[RelevanceJudgment]:
        """Loads relevance judgments from a file."""
        pass

class IMetricCalculator(ABC):
    @abstractmethod
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        pass

    @abstractmethod
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        pass

    @abstractmethod
    def average_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Computes the Average Precision for a single query."""
        pass

    @abstractmethod
    def ndcg_at_k(self, retrieved_docs: List[str], graded_relevance_dict: Dict[str, int], k: int) -> float:
        """Computes Normalized Discounted Cumulative Gain at K."""
        pass

class IRetrievalModelAdapter(ABC):
    @abstractmethod
    def get_ranked_docs(self, query_id: str, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        """Returns a list of (doc_id, score) for a given query."""
        pass

class IEvaluationService(ABC):
    @abstractmethod
    def evaluate(self, model: IRetrievalModelAdapter, model_name: str, queries: Dict[str, str], qrels: List[RelevanceJudgment], top_k_list: List[int]) -> EvaluationReport:
        pass
