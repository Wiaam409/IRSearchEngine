from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ScoredDocument:
    doc_id: str
    score: float
    title: Optional[str] = None
    content_snippet: Optional[str] = None

@dataclass
class RetrievalResult:
    query: str
    results: List[ScoredDocument]
    execution_time_ms: float
