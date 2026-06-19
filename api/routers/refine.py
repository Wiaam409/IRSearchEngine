import time
from fastapi import APIRouter, Depends, HTTPException
from api.models.schemas import BaseAPIResponse, RefineData, RefineRequest
from api.dependencies.containers import get_refinement_service
from api.services.adapter import execute_sync_service

router = APIRouter(prefix="/refine", tags=["Query Refinement"])

@router.post("", response_model=BaseAPIResponse)
async def refine_query(request: RefineRequest, service = Depends(get_refinement_service)):
    start_time = time.time()
    try:
        result = await execute_sync_service(service.refine, request.query)
        
        data = RefineData(
            original_query=result.original_text,
            refined_query=result.refined_text,
            suggestions=result.suggestions,
            corrections=result.corrections,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return BaseAPIResponse(data=data)
