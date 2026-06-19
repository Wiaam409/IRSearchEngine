import os
import math
import json
import ijson
from typing import List, Dict, Tuple
from services.retrieval_service.domain.interfaces import IIndexReader
from services.indexing_service.domain.models import CollectionStatistics

class TfidfIndexReader(IIndexReader):
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.postings_cache: Dict[str, List[Tuple[str, int]]] = {}
        self.doc_norms: Dict[str, float] = {}
        self.idf_cache: Dict[str, float] = {}
        self.collection_stats = CollectionStatistics()
        self._load_index()

    def _load_index(self):
        # 1. Load Collection Stats
        with open(os.path.join(self.index_dir, "collection_statistics.json"), "r", encoding="utf-8") as f:
            c_data = json.load(f)
            self.collection_stats.total_documents = c_data["total_documents"]
            self.collection_stats.total_tokens = c_data["total_tokens"]
            self.collection_stats.average_document_length = c_data["average_document_length"]

        N = self.collection_stats.total_documents

        # 2. Load Inverted Index and Precompute IDF
        with open(os.path.join(self.index_dir, "inverted_index.json"), "r", encoding="utf-8") as f:
            for term, data in ijson.kvitems(f, ''):
                df = data["df"]
                # Standard smooth IDF
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                self.idf_cache[term] = idf
                
                # Cache postings as list of tuples
                self.postings_cache[term] = [(p["doc_id"], p["tf"]) for p in data["postings"]]

        # 3. Precompute Document Norms for Cosine Similarity
        norm_sums: Dict[str, float] = {}
        for term, postings in self.postings_cache.items():
            idf = self.idf_cache[term]
            for doc_id, tf in postings:
                weight = tf * idf
                if doc_id not in norm_sums:
                    norm_sums[doc_id] = 0.0
                norm_sums[doc_id] += weight * weight
                
        for doc_id, sum_sq in norm_sums.items():
            self.doc_norms[doc_id] = math.sqrt(sum_sq)

    def get_term_posting_list(self, term: str) -> List[Tuple[str, int]]:
        return self.postings_cache.get(term, [])

    def get_doc_norm(self, doc_id: str) -> float:
        return self.doc_norms.get(doc_id, 1.0)

    def get_idf(self, term: str) -> float:
        return self.idf_cache.get(term, 0.0)

    def get_collection_stats(self) -> CollectionStatistics:
        return self.collection_stats
