import os
import math
import json
import ijson
from typing import List, Dict, Tuple
from services.ranking_service.domain.interfaces import IIndexReader

class Bm25IndexReader(IIndexReader):
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.postings_cache: Dict[str, List[Tuple[str, int]]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self._load_index()

    def _load_index(self):
        # 1. Load Collection Stats for avgdl and N
        with open(os.path.join(self.index_dir, "collection_statistics.json"), "r", encoding="utf-8") as f:
            c_data = json.load(f)
            self.avgdl = c_data["average_document_length"]
            N = c_data["total_documents"]

        # 2. Load Document Statistics
        with open(os.path.join(self.index_dir, "document_statistics.json"), "r", encoding="utf-8") as f:
            d_data = json.load(f)
            for doc_id, stats in d_data.items():
                self.doc_lengths[doc_id] = stats["length"]

        # 3. Load Inverted Index and Precompute IDF
        with open(os.path.join(self.index_dir, "inverted_index.json"), "r", encoding="utf-8") as f:
            for term, data in ijson.kvitems(f, ''):
                df = data["df"]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                self.idf_cache[term] = idf
                
                # Cache postings as list of tuples
                self.postings_cache[term] = [(p["doc_id"], p["tf"]) for p in data["postings"]]

    def get_posting_list(self, term: str) -> List[Tuple[str, int]]:
        return self.postings_cache.get(term, [])

    def get_doc_length(self, doc_id: str) -> int:
        return self.doc_lengths.get(doc_id, 1)

    def get_avgdl(self) -> float:
        return self.avgdl

    def get_idf(self, term: str) -> float:
        return self.idf_cache.get(term, 0.0)
