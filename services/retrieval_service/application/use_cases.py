import time
from typing import Dict, List
from collections import defaultdict
from services.retrieval_service.domain.interfaces import IRetrievalService, IQueryProcessor, IIndexReader, IScorer
from services.retrieval_service.domain.models import ScoredDocument, RetrievalResult

class TfidfRetrieveUseCase(IRetrievalService):
    def __init__(self, query_processor: IQueryProcessor, index_reader: IIndexReader, scorer: IScorer):
        self.query_processor = query_processor
        self.index_reader = index_reader
        self.scorer = scorer

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()
        
        # 1. Process Query
        tokens = self.query_processor.process(query)
        if not tokens:
            return RetrievalResult(query=query, results=[], execution_time_ms=(time.time() - start_time) * 1000)

        # 2. Compute Query Vector (TF * IDF)
        query_tf = defaultdict(int)
        for token in tokens:
            query_tf[token] += 1
            
        query_vector = {}
        for term, tf in query_tf.items():
            idf = self.index_reader.get_idf(term)
            query_vector[term] = tf * idf

        # 3. Term-At-A-Time Accumulation
        doc_dot_products = defaultdict(float)
        
        for term, q_weight in query_vector.items():
            postings = self.index_reader.get_term_posting_list(term)
            idf = self.index_reader.get_idf(term)
            
            for doc_id, doc_tf in postings:
                doc_term_weight = doc_tf * idf
                doc_dot_products[doc_id] += q_weight * doc_term_weight

        # 4 & 5. Compute Final Cosine Similarity using Scorer
        scored_docs = []
        for doc_id, dot_product in doc_dot_products.items():
            score = self.scorer.score(query_vector, doc_id, dot_product)
            scored_docs.append(ScoredDocument(doc_id=doc_id, score=score))

        # 6. Sort and truncate
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        top_docs = scored_docs[:top_k]
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(query=query, results=top_docs, execution_time_ms=execution_time_ms)
