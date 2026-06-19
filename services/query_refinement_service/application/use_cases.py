from typing import List, Dict
from services.query_refinement_service.domain.interfaces import IQueryRefiner, ISpellChecker, ISynonymProvider, IQueryHistory
from services.query_refinement_service.domain.models import RefinedQuery

class RefineQueryUseCase(IQueryRefiner):
    def __init__(self, spell_checker: ISpellChecker, synonym_provider: ISynonymProvider, history_tracker: IQueryHistory = None):
        self.spell_checker = spell_checker
        self.synonym_provider = synonym_provider
        self.history_tracker = history_tracker

    def refine(self, query: str) -> RefinedQuery:
        if not query.strip():
            return RefinedQuery(query, query)
            
        if self.history_tracker:
            self.history_tracker.add_query(query)

        tokens = query.split()
        refined_tokens = []
        corrections: Dict[str, str] = {}
        suggestions: List[str] = []

        for token in tokens:
            # 1. Spell Correction
            corrected_token = self.spell_checker.correct(token)
            if corrected_token != token:
                corrections[token] = corrected_token
            
            # 2. Synonym Expansion
            synonyms = self.synonym_provider.get_synonyms(corrected_token)
            
            if synonyms:
                # Add synonyms as OR expansion
                expansion = f"({corrected_token} OR {' OR '.join(synonyms)})"
                refined_tokens.append(expansion)
                suggestions.extend(synonyms)
            else:
                refined_tokens.append(corrected_token)

        refined_text = " ".join(refined_tokens)
        
        # Add historical suggestions
        if self.history_tracker:
            suggestions.extend(self.history_tracker.get_recent_queries(5))
            
        return RefinedQuery(
            original_text=query,
            refined_text=refined_text,
            suggestions=list(dict.fromkeys(suggestions)), # Deduplicate
            corrections=corrections
        )
