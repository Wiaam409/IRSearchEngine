from typing import List, Dict
from collections import defaultdict
from services.evaluation_service.domain.interfaces import IEvaluationService, IMetricCalculator, IQrelLoader, IRetrievalModelAdapter
from services.evaluation_service.domain.models import RelevanceJudgment, MetricResult, EvaluationReport
from tqdm import tqdm

class EvaluateModelUseCase(IEvaluationService):
    def __init__(self, qrel_loader: IQrelLoader, calculator: IMetricCalculator):
        self.qrel_loader = qrel_loader
        self.calculator = calculator

    def evaluate(self, model: IRetrievalModelAdapter, model_name: str, queries: Dict[str, str], qrels: List[RelevanceJudgment], top_k_list: List[int]) -> EvaluationReport:
        report = EvaluationReport(model_name=model_name)
        
        # Group qrels by query
        qrels_by_query: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in qrels:
            qrels_by_query[qrel.query_id][qrel.doc_id] = qrel.relevance_score

        # Prepare accumulators for macro-averaging
        sum_metrics: Dict[str, float] = defaultdict(float)
        query_count = 0

        max_k = max(top_k_list) if top_k_list else 10

        for qid, text in tqdm(queries.items(), desc=f"Evaluating {model_name}"):
            if qid not in qrels_by_query:
                continue # Skip queries without relevance judgments

            query_count += 1
            graded_rels = qrels_by_query[qid]
            # For binary metrics (Precision, Recall, MAP), treat rel > 0 as relevant
            relevant_docs = [doc for doc, rel in graded_rels.items() if rel > 0]

            # 1. Run Retrieval via Adapter
            ranked_docs = model.get_ranked_docs(qid, text, top_k=max_k)
            retrieved_ids = [doc_id for doc_id, _ in ranked_docs]

            # 2. Compute Metrics per query
            query_metrics = []
            
            # MAP
            ap = self.calculator.average_precision(retrieved_ids, relevant_docs)
            query_metrics.append(MetricResult("MAP", ap, qid))
            sum_metrics["MAP"] += ap

            # K-dependent metrics
            for k in top_k_list:
                p_k = self.calculator.precision_at_k(retrieved_ids, relevant_docs, k)
                r_k = self.calculator.recall_at_k(retrieved_ids, relevant_docs, k)
                ndcg_k = self.calculator.ndcg_at_k(retrieved_ids, graded_rels, k)

                query_metrics.append(MetricResult(f"P@{k}", p_k, qid))
                query_metrics.append(MetricResult(f"Recall@{k}", r_k, qid))
                query_metrics.append(MetricResult(f"nDCG@{k}", ndcg_k, qid))

                sum_metrics[f"P@{k}"] += p_k
                sum_metrics[f"Recall@{k}"] += r_k
                sum_metrics[f"nDCG@{k}"] += ndcg_k

            report.per_query_metrics[qid] = query_metrics

        # 3. Aggregate across queries
        if query_count > 0:
            for metric_name, total_val in sum_metrics.items():
                avg_val = total_val / query_count
                report.aggregate_metrics.append(MetricResult(metric_name, avg_val))

        return report
