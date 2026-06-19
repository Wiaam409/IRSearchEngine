import os
import time
import argparse

from services.preprocessing_service.application.pipeline import PreprocessingPipeline
from services.preprocessing_service.application.arabic_processors import (
    ArabicNormalizer, ArabicTokenizer, ArabicStopwordRemover
)

from services.ranking_service.domain.models import Bm25Parameters
from services.ranking_service.infrastructure.bm25_index_reader import Bm25IndexReader
from services.ranking_service.infrastructure.query_processor_adapter import QueryProcessorAdapter
from services.ranking_service.infrastructure.bm25_scorer import Bm25Scorer
from services.ranking_service.application.use_cases import Bm25RankUseCase

def main():
    parser = argparse.ArgumentParser(description="BM25 Ranking CLI Demo")
    parser.add_argument("--index", type=str, default="datasets/cache/index", help="Path to index directory")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    args = parser.parse_args()

    print("Loading index...")
    try:
        index_reader = Bm25IndexReader(args.index)
        print(f"Index loaded. Avg Doc Length: {index_reader.get_avgdl():.2f}")
    except Exception as e:
        print(f"Failed to load index from {args.index}: {e}")
        return

    # Set up preprocessing for Arabic
    stopwords = {"في", "من", "على", "إلى"} # Hardcoded for demo simplicity
    pipeline = PreprocessingPipeline(
        normalizer=ArabicNormalizer(),
        tokenizer=ArabicTokenizer(),
        stopword_remover=ArabicStopwordRemover(stopwords)
    )
    
    query_processor = QueryProcessorAdapter(pipeline)
    scorer = Bm25Scorer()
    ranking_service = Bm25RankUseCase(query_processor, index_reader, scorer)

    params = Bm25Parameters(k1=args.k1, b=args.b)
    print(f"\n--- BM25 Ranking System Ready (k1={params.k1}, b={params.b}) ---")
    
    while True:
        try:
            query = input("\nEnter query (or Ctrl+C to exit): ").strip()
            if not query:
                continue
                
            start_time = time.time()
            results = ranking_service.rank(query, top_k=5, params=params)
            execution_time_ms = (time.time() - start_time) * 1000
            
            print(f"\nFound {len(results)} results in {execution_time_ms:.2f} ms")
            for i, doc in enumerate(results, 1):
                print(f"{i}. DocID: {doc.doc_id} | Score: {doc.score:.4f}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
