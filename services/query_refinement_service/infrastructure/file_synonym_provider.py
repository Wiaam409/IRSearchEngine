import os
import json
from typing import List, Dict
from services.query_refinement_service.domain.interfaces import ISynonymProvider

class FileBasedSynonymProvider(ISynonymProvider):
    def __init__(self, filepath: str):
        self.synonyms_map: Dict[str, List[str]] = {}
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.synonyms_map = json.load(f)

    def get_synonyms(self, term: str) -> List[str]:
        return self.synonyms_map.get(term, [])
