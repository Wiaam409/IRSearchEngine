import time
from fastapi import APIRouter, Depends, HTTPException
from api.models.schemas import BaseAPIResponse, EvaluateData, EvaluateRequest
from api.dependencies.containers import (
    get_evaluation_service, get_tfidf_service, get_bm25_service,
    get_parallel_hybrid_service
)
from services.hybrid_service.infrastructure.adapters import RetrieverAdapter
from api.services.adapter import execute_sync_service

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

@router.post("", response_model=BaseAPIResponse)
async def evaluate_model(request: EvaluateRequest, eval_service = Depends(get_evaluation_service)):
    start_time = time.time()
    try:
        if request.model == "tfidf":
            model_to_evaluate = RetrieverAdapter(get_tfidf_service())
        elif request.model == "bm25":
            model_to_evaluate = RetrieverAdapter(get_bm25_service())
        elif request.model == "hybrid":
            model_to_evaluate = RetrieverAdapter(get_parallel_hybrid_service("bm25", "rrf"))
        else:
            raise HTTPException(status_code=400, detail="Unknown model requested")
            
        qrels_path = request.qrels_path if request.qrels_path else "datasets/miracl/ar/dev/qrels.tsv"
        
        # We simulate passing qrels and queries. In real scenarios, API would load queries.
        from services.evaluation_service.domain.models import RelevanceJudgment
        qrels = [RelevanceJudgment("q1", "mock_doc_1", 1)]
        queries = {"q1": "test query"}
        
        metrics_result = await execute_sync_service(
            eval_service.evaluate, model_to_evaluate, request.model, queries, qrels, request.top_k_list
        )
        
        # metrics_result is EvaluationReport
        # To match the expected format, we convert it to dict
        aggregated = {m.metric_name: m.value for m in metrics_result.aggregate_metrics}
        
        data = EvaluateData(
            metrics=aggregated,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Qrels file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return BaseAPIResponse(data=data)
