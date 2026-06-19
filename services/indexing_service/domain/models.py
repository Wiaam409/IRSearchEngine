from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Posting:
    doc_id: str
    tf: int

@dataclass
class PostingList:
    postings: List[Posting] = field(default_factory=list)
    df: int = 0
    cf: int = 0

@dataclass
class DocumentStatistics:
    doc_id: str
    length: int

@dataclass
class CollectionStatistics:
    total_documents: int = 0
    total_tokens: int = 0
    average_document_length: float = 0.0
