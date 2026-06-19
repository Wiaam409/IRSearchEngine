import time
from fastapi import APIRouter, Depends, HTTPException
from api.models.schemas import (
    BaseAPIResponse, SearchData, ScoredDocumentResponse,
    TfidfRequest, Bm25Request, EmbeddingsRequest,
    HybridParallelRequest, HybridSerialRequest
)
from api.dependencies.containers import (
    get_tfidf_service, get_bm25_service, get_dense_service,
    get_parallel_hybrid_service, get_serial_hybrid_service
)
from api.services.adapter import execute_sync_service
from services.hybrid_service.domain.models import FusionConfig

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("/tfidf", response_model=BaseAPIResponse)
async def search_tfidf(request: TfidfRequest, service = Depends(get_tfidf_service)):
    start_time = time.time()
    try:
        result = await execute_sync_service(service.retrieve, request.query, request.top_k)
        docs = [ScoredDocumentResponse(doc_id=d.doc_id, score=d.score) for d in result.results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/bm25", response_model=BaseAPIResponse)
async def search_bm25(request: Bm25Request, service = Depends(get_bm25_service)):
    start_time = time.time()
    try:
        from services.ranking_service.domain.models import Bm25Parameters
        params = Bm25Parameters(k1=request.k1, b=request.b)
        result = await execute_sync_service(service.rank, request.query, request.top_k, params)
        docs = [ScoredDocumentResponse(doc_id=d.doc_id, score=d.score) for d in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/embeddings", response_model=BaseAPIResponse)
async def search_embeddings(request: EmbeddingsRequest, service = Depends(get_dense_service)):
    start_time = time.time()
    try:
        result = await execute_sync_service(service.retrieve, request.query, request.top_k)
        docs = [ScoredDocumentResponse(doc_id=d.doc_id, score=d.score) for d in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/hybrid/parallel", response_model=BaseAPIResponse)
async def search_hybrid_parallel(request: HybridParallelRequest):
    start_time = time.time()
    if request.sparse_model not in ["bm25", "tfidf"]:
        raise HTTPException(status_code=400, detail="Invalid sparse_model")
    if request.fusion not in ["rrf", "score"]:
        raise HTTPException(status_code=400, detail="Invalid fusion method")
        
    try:
        service = get_parallel_hybrid_service(request.sparse_model, request.fusion)
        config = FusionConfig(sparse_weight=request.sparse_weight, dense_weight=request.dense_weight)
        result = await execute_sync_service(service.retrieve, request.query, request.top_k, config)
        docs = [ScoredDocumentResponse(doc_id=d.doc_id, score=d.score) for d in result.results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/hybrid/serial", response_model=BaseAPIResponse)
async def search_hybrid_serial(request: HybridSerialRequest, service = Depends(get_serial_hybrid_service)):
    start_time = time.time()
    try:
        result = await execute_sync_service(service.retrieve, request.query, request.top_k, request.candidate_multiplier)
        docs = [ScoredDocumentResponse(doc_id=d.doc_id, score=d.score) for d in result.results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )
