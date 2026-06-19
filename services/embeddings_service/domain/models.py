from dataclasses import dataclass
from typing import List, Optional

DenseVector = List[float]

@dataclass
class VectorSearchResult:
    doc_id: str
    score: float
    metadata: Optional[dict] = None
