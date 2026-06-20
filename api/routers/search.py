import time
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from api.models.schemas import (
    BaseAPIResponse, SearchData, ScoredDocumentResponse,
    TfidfRequest, Bm25Request, EmbeddingsRequest,
    HybridParallelRequest, HybridSerialRequest
)
from api.dependencies.containers import (
    get_tfidf_service, get_bm25_service, get_dense_service,
    get_parallel_hybrid_service, get_serial_hybrid_service,
    get_rag_service, get_db_store
)
from api.services.adapter import execute_sync_service
from services.hybrid_service.domain.models import FusionConfig

def format_results(result_docs, db_store):
    if not result_docs:
        return []
        
    # If the use case returned a RetrievalResult object instead of a list, extract the list
    if hasattr(result_docs, 'results'):
        result_docs = result_docs.results
    
    doc_ids = [d.doc_id if hasattr(d, 'doc_id') else d for d in result_docs]
    db_docs = db_store.get_documents(doc_ids)
    
    docs = []
    for d in result_docs:
        db_doc = db_docs.get(d.doc_id, {})
        docs.append(ScoredDocumentResponse(
            doc_id=d.doc_id,
            score=d.score,
            title=db_doc.get("title"),
            text=db_doc.get("text")
        ))
    return docs

class RagRequest(BaseModel):
    query: str
    top_k: int = 3

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("/tfidf", response_model=BaseAPIResponse)
async def search_tfidf(request: TfidfRequest, service = Depends(get_tfidf_service), db_store = Depends(get_db_store)):
    start_time = time.time()
    try:
        result = await execute_sync_service(service.retrieve, request.query, request.top_k)
        docs = format_results(result, db_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/bm25", response_model=BaseAPIResponse)
async def search_bm25(request: Bm25Request, service = Depends(get_bm25_service), db_store = Depends(get_db_store)):
    start_time = time.time()
    try:
        from services.ranking_service.domain.models import Bm25Parameters
        params = Bm25Parameters(k1=request.k1, b=request.b)
        result = await execute_sync_service(service.rank, request.query, request.top_k, params)
        docs = format_results(result, db_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/embeddings", response_model=BaseAPIResponse)
async def search_embeddings(request: EmbeddingsRequest, service = Depends(get_dense_service), db_store = Depends(get_db_store)):
    start_time = time.time()
    try:
        result = await execute_sync_service(service.retrieve, request.query, request.top_k)
        docs = format_results(result, db_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/hybrid/parallel", response_model=BaseAPIResponse)
async def search_hybrid_parallel(request: HybridParallelRequest, db_store = Depends(get_db_store)):
    start_time = time.time()
    if request.sparse_model not in ["bm25", "tfidf"]:
        raise HTTPException(status_code=400, detail="Invalid sparse_model")
    if request.fusion not in ["rrf", "score"]:
        raise HTTPException(status_code=400, detail="Invalid fusion method")
        
    try:
        service = get_parallel_hybrid_service(request.sparse_model, request.fusion)
        config = FusionConfig(sparse_weight=request.sparse_weight, dense_weight=request.dense_weight)
        result = await execute_sync_service(service.retrieve, request.query, request.top_k, config)
        docs = format_results(result.results, db_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/hybrid/serial", response_model=BaseAPIResponse)
async def search_hybrid_serial(request: HybridSerialRequest, service = Depends(get_serial_hybrid_service), db_store = Depends(get_db_store)):
    start_time = time.time()
    try:
        result = await execute_sync_service(service.retrieve, request.query, request.top_k, request.candidate_multiplier)
        docs = format_results(result.results, db_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    exec_ms = (time.time() - start_time) * 1000
    return BaseAPIResponse(
        data=SearchData(results=docs, query=request.query, execution_time_ms=exec_ms)
    )

@router.post("/rag", response_model=BaseAPIResponse)
async def search_rag(request: RagRequest, rag_service = Depends(get_rag_service), db_store = Depends(get_db_store)):
    start_time = time.time()
    try:
        result = await execute_sync_service(rag_service.generate_answer, request.query, request.top_k)
        
        exec_ms = (time.time() - start_time) * 1000
        docs = format_results(result.documents, db_store)
        return BaseAPIResponse(
            data={"query": request.query, "answer": result.answer, "context": result.context, "results": docs, "execution_time_ms": exec_ms}
        )
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())
