from typing import List, Optional, Any
from pydantic import BaseModel, Field

class ScoredDocumentResponse(BaseModel):
    doc_id: str
    score: float

class BaseAPIResponse(BaseModel):
    status: str = "success"
    data: Any

class SearchData(BaseModel):
    results: List[ScoredDocumentResponse]
    query: str
    execution_time_ms: float

class RefineData(BaseModel):
    original_query: str
    refined_query: str
    suggestions: List[str]
    corrections: dict
    execution_time_ms: float

class EvaluateData(BaseModel):
    metrics: dict
    execution_time_ms: float

class TfidfRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)

class Bm25Request(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)
    k1: float = 1.2
    b: float = 0.75

class EmbeddingsRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)

class HybridParallelRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)
    sparse_model: str = Field(default="bm25", description="'bm25' or 'tfidf'")
    fusion: str = Field(default="rrf", description="'rrf' or 'score'")
    sparse_weight: float = 0.5
    dense_weight: float = 0.5

class HybridSerialRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)
    candidate_multiplier: int = Field(default=10, ge=1)

class RefineRequest(BaseModel):
    query: str
    spell_check: bool = True
    expand_synonyms: bool = True

class EvaluateRequest(BaseModel):
    model: str = Field(..., description="'tfidf', 'bm25', or 'hybrid'")
    top_k_list: List[int] = [1, 5, 10]
    qrels_path: Optional[str] = None
