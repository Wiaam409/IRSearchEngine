import os
from typing import List
from services.evaluation_service.domain.interfaces import IQrelLoader
from services.evaluation_service.domain.models import RelevanceJudgment

class QrelFileLoader(IQrelLoader):
    def load(self, filepath: str) -> List[RelevanceJudgment]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Qrel file not found: {filepath}")
            
        judgments = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # Expected standard format: query_id, _, doc_id, relevance
                if len(parts) >= 4:
                    qid = parts[0]
                    doc_id = parts[2]
                    try:
                        rel = int(parts[3])
                        judgments.append(RelevanceJudgment(query_id=qid, doc_id=doc_id, relevance_score=rel))
                    except ValueError:
                        continue
        return judgments
