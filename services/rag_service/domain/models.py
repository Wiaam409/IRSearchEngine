from dataclasses import dataclass
from typing import List
from services.ranking_service.domain.models import ScoredDocument

@dataclass
class RagResult:
    answer: str
    context: str
    documents: List[ScoredDocument]
