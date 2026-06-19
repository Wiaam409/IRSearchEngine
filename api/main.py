from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tqdm import tqdm
from api.config import settings
from api.dependencies.containers import (
    get_tfidf_service, get_bm25_service, get_dense_service, get_refinement_service
)
from api.routers import search, refine, evaluate
from api.models.schemas import BaseAPIResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Pre-loading models and indexes into memory...")
        services_to_load = [
            ("TF-IDF", get_tfidf_service),
            ("BM25", get_bm25_service),
            ("Dense/Embeddings", get_dense_service),
            ("Refinement", get_refinement_service)
        ]
        
        for name, service_loader in tqdm(services_to_load, desc="Loading Models", unit="model"):
            service_loader()
            
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not pre-load all models. {e}")
    yield
    print("Shutting down IR API...")

app = FastAPI(
    title="Information Retrieval Search Engine API",
    description="Language-Agnostic IR REST API exposing TF-IDF, BM25, Embeddings, and Hybrid search.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(refine.router)
app.include_router(evaluate.router)

@app.get("/health", response_model=BaseAPIResponse, tags=["System"])
async def health_check():
    return BaseAPIResponse(data={"status": "healthy"})

# Example cURL to test BM25 search:
# curl -X POST "http://localhost:8000/search/bm25" -H "Content-Type: application/json" -d '{"query": "سيارة", "top_k": 5}'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.host, port=settings.port, reload=True)
