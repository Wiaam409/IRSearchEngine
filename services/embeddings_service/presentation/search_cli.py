import os
import argparse
import time

from services.embeddings_service.infrastructure.embedding_models import SentenceTransformerEmbedding
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
from services.embeddings_service.application.use_cases import DenseRetrieveUseCase

def main():
    parser = argparse.ArgumentParser(description="Dense Search CLI Demo")
    parser.add_argument("--index", type=str, default="datasets/cache/embeddings", help="Path to dense index dir")
    args = parser.parse_args()

    print("Loading embedding model...")
    try:
        model = SentenceTransformerEmbedding()
    except ImportError as e:
        print(f"Error: {e}")
        return

    print(f"Loading vector index from {args.index}...")
    index = InMemoryVectorIndex()
    index.load(args.index)
    
    use_case = DenseRetrieveUseCase(model, index)

    print("\n--- Dense Retrieval System Ready ---")
    while True:
        try:
            query = input("\nEnter query (or Ctrl+C to exit): ").strip()
            if not query:
                continue
                
            start_time = time.time()
            results = use_case.retrieve(query, top_k=5)
            execution_ms = (time.time() - start_time) * 1000
            
            print(f"\nFound {len(results)} results in {execution_ms:.2f} ms")
            for i, res in enumerate(results, 1):
                print(f"{i}. DocID: {res.doc_id} | Cosine Score: {res.score:.4f}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
