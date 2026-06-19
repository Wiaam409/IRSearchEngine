from typing import List, Tuple
from tqdm import tqdm
from services.indexing_service.domain.interfaces import IIndexWriter

class BuildIndexUseCase:
    """Orchestrates building the inverted index by adding documents sequentially."""
    
    def __init__(self, writer: IIndexWriter):
        self.writer = writer

    def execute(self, documents: List[Tuple[str, List[str]]]) -> None:
        """
        Takes a list of tuples containing (doc_id, tokens).
        This abstracts away where the tokens come from (e.g., preprocessing_service).
        """
        for doc_id, tokens in tqdm(documents, desc="Building index"):
            self.writer.add_document(doc_id, tokens)
        
        print("Committing index to disk...")
        self.writer.commit()

class UpdateIndexUseCase:
    """Adds a single document to the index and immediately commits."""
    
    def __init__(self, writer: IIndexWriter):
        self.writer = writer

    def execute(self, doc_id: str, tokens: List[str]) -> None:
        self.writer.add_document(doc_id, tokens)
        self.writer.commit()
