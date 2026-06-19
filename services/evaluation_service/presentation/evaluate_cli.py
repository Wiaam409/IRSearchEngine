import os
import argparse
import json

from services.preprocessing_service.application.pipeline import PreprocessingPipeline
from services.preprocessing_service.application.arabic_processors import ArabicNormalizer, ArabicTokenizer, ArabicStopwordRemover

from services.retrieval_service.infrastructure.tfidf_index_reader import TfidfIndexReader
from services.retrieval_service.infrastructure.query_processor_adapter import QueryProcessorAdapter as RetrievalQueryAdapter
from services.retrieval_service.infrastructure.tfidf_scorer import TfidfScorer
from services.retrieval_service.application.use_cases import TfidfRetrieveUseCase

from services.ranking_service.infrastructure.bm25_index_reader import Bm25IndexReader
from services.ranking_service.infrastructure.query_processor_adapter import QueryProcessorAdapter as RankingQueryAdapter
from services.ranking_service.infrastructure.bm25_scorer import Bm25Scorer
from services.ranking_service.application.use_cases import Bm25RankUseCase

from services.evaluation_service.infrastructure.qrel_loader import QrelFileLoader
from services.evaluation_service.infrastructure.metric_calculator import MetricCalculator
from services.evaluation_service.infrastructure.retrieval_adapter import TfidfModelAdapter, Bm25ModelAdapter
from services.evaluation_service.application.use_cases import EvaluateModelUseCase

def main():
    parser = argparse.ArgumentParser(description="Evaluation CLI")
    parser.add_argument("--model", type=str, choices=["tfidf", "bm25"], required=True)
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--queries", type=str, required=True, help="Path to queries file (JSON format {qid: text})")
    parser.add_argument("--index", type=str, default="datasets/cache/index")
    parser.add_argument("--k", type=str, default="1,5,10", help="Comma-separated K values")
    args = parser.parse_args()

    k_list = [int(k.strip()) for k in args.k.split(",")]

    print(f"Loading queries from {args.queries}...")
    with open(args.queries, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print(f"Loading qrels from {args.qrels}...")
    loader = QrelFileLoader()
    qrels = loader.load(args.qrels)
    
    stopwords = {"في", "من", "على", "إلى"}
    pipeline = PreprocessingPipeline(ArabicNormalizer(), ArabicTokenizer(), ArabicStopwordRemover(stopwords))

    if args.model == "tfidf":
        print("Initializing TF-IDF Model...")
        idx_reader = TfidfIndexReader(args.index)
        query_adapter = RetrievalQueryAdapter(pipeline)
        scorer = TfidfScorer(idx_reader)
        use_case = TfidfRetrieveUseCase(query_adapter, idx_reader, scorer)
        adapter = TfidfModelAdapter(use_case)
    else:
        print("Initializing BM25 Model...")
        idx_reader = Bm25IndexReader(args.index)
        query_adapter = RankingQueryAdapter(pipeline)
        scorer = Bm25Scorer()
        use_case = Bm25RankUseCase(query_adapter, idx_reader, scorer)
        adapter = Bm25ModelAdapter(use_case)

    evaluator = EvaluateModelUseCase(loader, MetricCalculator())
    
    print("Running evaluation... This may take a moment.")
    report = evaluator.evaluate(adapter, args.model, queries, qrels, k_list)

    print(f"\n--- Evaluation Report: {report.model_name.upper()} ---")
    
    # Sort for deterministic printing
    report.aggregate_metrics.sort(key=lambda x: x.metric_name)
    for m in report.aggregate_metrics:
        print(f"{m.metric_name}: {m.value:.4f}")

if __name__ == "__main__":
    main()
