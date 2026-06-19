from typing import List, Dict
from collections import defaultdict
from services.ranking_service.domain.interfaces import IRankingService, IQueryProcessor, IIndexReader, IBm25Scorer
from services.ranking_service.domain.models import ScoredDocument, Bm25Parameters

class Bm25RankUseCase(IRankingService):
    def __init__(self, query_processor: IQueryProcessor, index_reader: IIndexReader, scorer: IBm25Scorer):
        self.query_processor = query_processor
        self.index_reader = index_reader
        self.scorer = scorer

    def rank(self, query: str, top_k: int = 10, params: Bm25Parameters = None) -> List[ScoredDocument]:
        if params is None:
            params = Bm25Parameters()
            
        # Validate parameters
        if params.k1 < 0:
            params.k1 = 1.2
        if not (0 <= params.b <= 1):
            params.b = 0.75

        # 1. Process Query
        tokens = self.query_processor.process(query)
        if not tokens:
            return []

        # Find unique terms
        unique_terms = set(tokens)
        
        doc_scores: Dict[str, float] = defaultdict(float)
        avgdl = self.index_reader.get_avgdl()

        # 2. Accumulate BM25 scores
        for term in unique_terms:
            idf = self.index_reader.get_idf(term)
            if idf <= 0: # Term not in index or idf is effectively 0
                continue
                
            postings = self.index_reader.get_posting_list(term)
            for doc_id, tf in postings:
                doc_len = self.index_reader.get_doc_length(doc_id)
                score = self.scorer.compute_score(
                    tf=tf, 
                    doc_len=doc_len, 
                    avgdl=avgdl, 
                    idf=idf, 
                    k1=params.k1, 
                    b=params.b
                )
                doc_scores[doc_id] += score

        # 3. Sort and truncate
        scored_docs = [ScoredDocument(doc_id=doc_id, score=score) for doc_id, score in doc_scores.items()]
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        
        return scored_docs[:top_k]
