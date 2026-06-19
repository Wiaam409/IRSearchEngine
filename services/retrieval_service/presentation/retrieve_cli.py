import os
import argparse

from services.preprocessing_service.application.pipeline import PreprocessingPipeline
from services.preprocessing_service.application.arabic_processors import (
    ArabicNormalizer, ArabicTokenizer, ArabicStopwordRemover
)

from services.retrieval_service.infrastructure.tfidf_index_reader import TfidfIndexReader
from services.retrieval_service.infrastructure.query_processor_adapter import QueryProcessorAdapter
from services.retrieval_service.infrastructure.tfidf_scorer import TfidfScorer
from services.retrieval_service.application.use_cases import TfidfRetrieveUseCase

def main():
    parser = argparse.ArgumentParser(description="TF-IDF Retrieval CLI Demo")
    parser.add_argument("--index", type=str, default="datasets/cache/index", help="Path to index directory")
    args = parser.parse_args()

    print("Loading index...")
    try:
        index_reader = TfidfIndexReader(args.index)
        print(f"Index loaded. Total docs: {index_reader.get_collection_stats().total_documents}")
    except Exception as e:
        print(f"Failed to load index from {args.index}: {e}")
        print("Please run the indexing service first to generate the inverted index.")
        return

    # Set up preprocessing for Arabic
    stopwords = {"في", "من", "على", "إلى"} # Hardcoded for demo simplicity
    pipeline = PreprocessingPipeline(
        normalizer=ArabicNormalizer(),
        tokenizer=ArabicTokenizer(),
        stopword_remover=ArabicStopwordRemover(stopwords)
    )
    
    query_processor = QueryProcessorAdapter(pipeline)
    scorer = TfidfScorer(index_reader)
    retrieval_service = TfidfRetrieveUseCase(query_processor, index_reader, scorer)

    print("\n--- TF-IDF Retrieval System Ready ---")
    while True:
        try:
            query = input("\nEnter query (or Ctrl+C to exit): ").strip()
            if not query:
                continue
                
            result = retrieval_service.retrieve(query, top_k=5)
            print(f"\nFound {len(result.results)} results in {result.execution_time_ms:.2f} ms")
            for i, doc in enumerate(result.results, 1):
                print(f"{i}. DocID: {doc.doc_id} | Score: {doc.score:.4f}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
