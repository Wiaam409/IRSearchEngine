from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class RefinedQuery:
    original_text: str
    refined_text: str
    suggestions: List[str] = field(default_factory=list)
    corrections: Dict[str, str] = field(default_factory=dict)
