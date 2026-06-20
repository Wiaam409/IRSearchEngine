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
    get_llm_pipeline
)
from api.services.adapter import execute_sync_service
from services.hybrid_service.domain.models import FusionConfig

class RagRequest(BaseModel):
    query: str
    top_k: int = 3

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("/tfidf", response_model=BaseAPIResponse)
async def search_tfidf(request: TfidfRequest, service = Depends(get_tfidf_service)):
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

@router.post("/rag", response_model=BaseAPIResponse)
async def search_rag(request: RagRequest):
    start_time = time.time()
    try:
        bm25 = get_bm25_service()
        from services.ranking_service.domain.models import Bm25Parameters
        params = Bm25Parameters(k1=1.5, b=0.75)
        result = await execute_sync_service(bm25.rank, request.query, request.top_k, params)
        
        from datasets.adapters.beir_adapter import BeirAdapter
        adapter = BeirAdapter()
        context_texts = []
        for d in result:
            text = adapter.get_document_text(d.doc_id)
            if text: context_texts.append(text[:500])
            
        context = " ".join(context_texts)
        try:
            generator = get_llm_pipeline()
            if generator:
                prompt = f"Answer the question using ONLY the context below.\n\nContext:\n{context}\n\nQuestion:\n{request.query}\n\nAnswer:\n"
                res = generator(prompt, max_new_tokens=50)
                answer = res[0]['generated_text'].strip()
            else:
                answer = f"[LLM Offline Mode] Retrieved Context Summary: {context[:200]}..."
        except Exception:
            answer = f"[LLM Offline Mode] Retrieved Context Summary: {context[:200]}..."
            
        exec_ms = (time.time() - start_time) * 1000
        docs = [ScoredDocumentResponse(doc_id=d.doc_id, score=d.score) for d in result]
        return BaseAPIResponse(data={"query": request.query, "answer": answer, "context": context, "results": docs, "execution_time_ms": exec_ms})
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())
