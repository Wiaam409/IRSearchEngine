import os
import json
import ijson
from typing import List, Dict, Optional
from .memory_index import InMemoryIndexWriter
from services.indexing_service.domain.interfaces import IIndexReader
from services.indexing_service.domain.models import Posting, PostingList, DocumentStatistics, CollectionStatistics

class JsonIndexWriter(InMemoryIndexWriter):
    """
    Extends InMemoryIndexWriter. Builds the index in memory during `add_document`,
    and dumps it to JSON files during `commit`.
    """
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir

    def commit(self) -> None:
        super().commit()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save Inverted Index (lexicon + postings)
        inverted_index_dict = {}
        for term, plist in self.lexicon.items():
            inverted_index_dict[term] = {
                "df": plist.df,
                "cf": plist.cf,
                "postings": [{"doc_id": p.doc_id, "tf": p.tf} for p in plist.postings]
            }
        
        with open(os.path.join(self.output_dir, "inverted_index.json"), "w", encoding="utf-8") as f:
            json.dump(inverted_index_dict, f, ensure_ascii=False)
            
        # Save Document Stats
        doc_stats_dict = {
            doc_id: {"length": stat.length}
            for doc_id, stat in self.doc_stats.items()
        }
        with open(os.path.join(self.output_dir, "document_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(doc_stats_dict, f, ensure_ascii=False)
            
        # Save Collection Stats
        coll_stats_dict = {
            "total_documents": self.collection_stats.total_documents,
            "total_tokens": self.collection_stats.total_tokens,
            "average_document_length": self.collection_stats.average_document_length
        }
        with open(os.path.join(self.output_dir, "collection_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(coll_stats_dict, f, ensure_ascii=False)


class JsonIndexReader(IIndexReader):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.lexicon: Dict[str, PostingList] = {}
        self.doc_stats: Dict[str, DocumentStatistics] = {}
        self.collection_stats = CollectionStatistics()
        self._loaded = False

    def _load_if_needed(self):
        if self._loaded: return
        
        with open(os.path.join(self.input_dir, "collection_statistics.json"), "r", encoding="utf-8") as f:
            c_data = json.load(f)
            self.collection_stats.total_documents = c_data["total_documents"]
            self.collection_stats.total_tokens = c_data["total_tokens"]
            self.collection_stats.average_document_length = c_data["average_document_length"]

        with open(os.path.join(self.input_dir, "document_statistics.json"), "r", encoding="utf-8") as f:
            d_data = json.load(f)
            for doc_id, stats in d_data.items():
                self.doc_stats[doc_id] = DocumentStatistics(doc_id=doc_id, length=stats["length"])
                
        with open(os.path.join(self.input_dir, "inverted_index.json"), "r", encoding="utf-8") as f:
            for term, data in ijson.kvitems(f, ''):
                postings = [Posting(doc_id=p["doc_id"], tf=p["tf"]) for p in data["postings"]]
                self.lexicon[term] = PostingList(postings=postings, df=data["df"], cf=data["cf"])
                
        self._loaded = True

    def get_postings(self, term: str) -> Optional[PostingList]:
        self._load_if_needed()
        return self.lexicon.get(term)

    def get_document_statistics(self, doc_id: str) -> Optional[DocumentStatistics]:
        self._load_if_needed()
        return self.doc_stats.get(doc_id)

    def get_collection_statistics(self) -> CollectionStatistics:
        self._load_if_needed()
        return self.collection_stats
