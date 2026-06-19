from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class RelevanceJudgment:
    query_id: str
    doc_id: str
    relevance_score: int

@dataclass
class MetricResult:
    metric_name: str
    value: float
    query_id: Optional[str] = None

@dataclass
class EvaluationReport:
    model_name: str
    aggregate_metrics: List[MetricResult] = field(default_factory=list)
    per_query_metrics: Dict[str, List[MetricResult]] = field(default_factory=dict)
