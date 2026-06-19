from dataclasses import dataclass

@dataclass
class Qrel:
    query_id: str
    doc_id: str
    relevance: int