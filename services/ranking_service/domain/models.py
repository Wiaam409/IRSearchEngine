from dataclasses import dataclass
from typing import List

@dataclass
class ScoredDocument:
    doc_id: str
    score: float

@dataclass
class Bm25Parameters:
    k1: float = 1.2
    b: float = 0.75
