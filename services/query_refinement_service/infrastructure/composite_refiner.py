from typing import List
from services.query_refinement_service.domain.interfaces import IQueryRefiner
from services.query_refinement_service.domain.models import RefinedQuery

class QueryRefinerComposite(IQueryRefiner):
    """
    Executes multiple refiners sequentially.
    """
    def __init__(self, refiners: List[IQueryRefiner]):
        self.refiners = refiners

    def refine(self, query: str) -> RefinedQuery:
        current_query = RefinedQuery(original_text=query, refined_text=query)
        
        for refiner in self.refiners:
            res = refiner.refine(current_query.refined_text)
            current_query.refined_text = res.refined_text
            current_query.suggestions.extend(res.suggestions)
            current_query.corrections.update(res.corrections)
            
        # Deduplicate suggestions
        current_query.suggestions = list(dict.fromkeys(current_query.suggestions))
        return current_query
