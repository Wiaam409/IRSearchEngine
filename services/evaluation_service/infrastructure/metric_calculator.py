import math
from typing import List, Dict
from services.evaluation_service.domain.interfaces import IMetricCalculator

class MetricCalculator(IMetricCalculator):
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        if k == 0:
            return 0.0
        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        relevant_count = sum(1 for doc in top_k if doc in relevant_set)
        return relevant_count / k

    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        if not relevant_docs:
            return 0.0
        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        relevant_count = sum(1 for doc in top_k if doc in relevant_set)
        return relevant_count / len(relevant_docs)

    def average_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        if not relevant_docs:
            return 0.0
            
        relevant_set = set(relevant_docs)
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
                
        return precision_sum / len(relevant_docs)

    def ndcg_at_k(self, retrieved_docs: List[str], graded_relevance_dict: Dict[str, int], k: int) -> float:
        top_k = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(top_k):
            rel = graded_relevance_dict.get(doc, 0)
            if rel > 0:
                dcg += rel / math.log2(i + 2) # i is 0-indexed, so log2(index + 1) -> log2(i+1+1) -> log2(i+2)

        # Calculate IDCG
        ideal_docs = sorted(graded_relevance_dict.values(), reverse=True)
        ideal_top_k = ideal_docs[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_top_k):
            if rel > 0:
                idcg += rel / math.log2(i + 2)

        if idcg == 0.0:
            return 0.0
            
        return dcg / idcg
