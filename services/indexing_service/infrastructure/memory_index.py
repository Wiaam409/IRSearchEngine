from typing import List, Dict, Optional
from collections import defaultdict
from services.indexing_service.domain.interfaces import IIndexWriter
from services.indexing_service.domain.models import Posting, PostingList, DocumentStatistics, CollectionStatistics

class InMemoryIndexWriter(IIndexWriter):
    def __init__(self):
        # term -> PostingList
        self.lexicon: Dict[str, PostingList] = defaultdict(PostingList)
        # doc_id -> DocumentStatistics
        self.doc_stats: Dict[str, DocumentStatistics] = {}
        self.collection_stats = CollectionStatistics()

    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        if not tokens:
            return
            
        # Calculate TF for this document
        tf_dict: Dict[str, int] = defaultdict(int)
        for token in tokens:
            tf_dict[token] += 1
            
        doc_length = len(tokens)
        self.doc_stats[doc_id] = DocumentStatistics(doc_id=doc_id, length=doc_length)
        
        # Update Collection Stats
        self.collection_stats.total_documents += 1
        self.collection_stats.total_tokens += doc_length
        self.collection_stats.average_document_length = (
            self.collection_stats.total_tokens / self.collection_stats.total_documents
        )
        
        # Update Lexicon and Postings
        for term, tf in tf_dict.items():
            posting_list = self.lexicon[term]
            posting_list.postings.append(Posting(doc_id=doc_id, tf=tf))
            posting_list.df += 1
            posting_list.cf += tf

    def commit(self) -> None:
        """For the pure in-memory writer, commit is a no-op."""
        pass
