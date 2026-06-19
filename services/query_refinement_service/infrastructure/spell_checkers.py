from typing import Set
from services.query_refinement_service.domain.interfaces import ISpellChecker

class NoOpSpellChecker(ISpellChecker):
    def correct(self, term: str) -> str:
        return term

class SimpleSpellChecker(ISpellChecker):
    """
    A basic spell checker using Levenshtein distance against a known dictionary.
    Returns the closest word if distance is <= max_distance.
    """
    def __init__(self, dictionary: Set[str], max_distance: int = 1):
        self.dictionary = dictionary
        self.max_distance = max_distance

    def correct(self, term: str) -> str:
        if term in self.dictionary or not self.dictionary:
            return term

        best_match = term
        min_dist = float('inf')

        for word in self.dictionary:
            if abs(len(word) - len(term)) > self.max_distance:
                continue
                
            dist = self._levenshtein(term, word)
            if dist < min_dist and dist <= self.max_distance:
                min_dist = dist
                best_match = word
                if dist == 1: 
                    break
                    
        return best_match

    def _levenshtein(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
