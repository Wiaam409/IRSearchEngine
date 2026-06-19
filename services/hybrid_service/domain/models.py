from dataclasses import dataclass
from typing import List

@dataclass
class ScoredDocument:
    doc_id: str
    score: float

@dataclass
class HybridResult:
    query: str
    results: List[ScoredDocument]
    fusion_method_used: str

@dataclass
class FusionConfig:
    sparse_weight: float = 0.5
    dense_weight: float = 0.5
    rrf_k_constant: int = 60
