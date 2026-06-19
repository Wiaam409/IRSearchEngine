import os
import argparse
import json
import time

from services.embeddings_service.infrastructure.embedding_models import SentenceTransformerEmbedding
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
from services.embeddings_service.application.use_cases import BuildDenseIndexUseCase

def main():
    parser = argparse.ArgumentParser(description="Dense Index Builder CLI")
    parser.add_argument("--docs", type=str, required=True, help="Path to documents JSON {doc_id: text}")
    parser.add_argument("--output", type=str, default="datasets/cache/embeddings", help="Output directory")
    args = parser.parse_args()

    print(f"Loading documents from {args.docs}...")
    with open(args.docs, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print("Loading embedding model (this may take a moment to download/initialize)...")
    try:
        model = SentenceTransformerEmbedding()
    except ImportError as e:
        print(f"Error: {e}")
        return

    index = InMemoryVectorIndex()
    use_case = BuildDenseIndexUseCase(model, index, batch_size=32)

    print(f"Building dense index for {len(documents)} documents...")
    start_time = time.time()
    use_case.build(documents, save_path=args.output)
    duration = time.time() - start_time
    
    print(f"Index built and saved to {args.output} in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
