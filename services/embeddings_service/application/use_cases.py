from typing import Dict, List
from tqdm import tqdm
from services.embeddings_service.domain.interfaces import IEmbeddingModel, IVectorIndex, IDenseRetrievalService
from services.embeddings_service.domain.models import VectorSearchResult

class BuildDenseIndexUseCase:
    def __init__(self, model: IEmbeddingModel, index: IVectorIndex, batch_size: int = 32):
        self.model = model
        self.index = index
        self.batch_size = batch_size

    def build(self, documents: Dict[str, str], save_path: str = None, checkpoint_every: int = 500):
        doc_ids = list(documents.keys())
        texts = list(documents.values())

        # Check for already-indexed docs (resume support)
        already_indexed = set(self.index.doc_ids) if hasattr(self.index, 'doc_ids') else set()
        if already_indexed:
            print(f"Resuming: {len(already_indexed)} docs already indexed, skipping them.")

        total_batches = (len(doc_ids) + self.batch_size - 1) // self.batch_size
        batches_since_save = 0

        with tqdm(total=len(doc_ids), desc="Embedding", unit="doc") as pbar:
            for i in range(0, len(doc_ids), self.batch_size):
                batch_ids = doc_ids[i:i + self.batch_size]
                batch_texts = texts[i:i + self.batch_size]

                # Skip docs that are already indexed
                new_ids = []
                new_texts = []
                for doc_id, text in zip(batch_ids, batch_texts):
                    if doc_id not in already_indexed:
                        new_ids.append(doc_id)
                        new_texts.append(text)

                if new_ids:
                    vectors = self.model.encode_batch(new_texts)
                    for doc_id, vector in zip(new_ids, vectors):
                        self.index.add(doc_id, vector)
                    batches_since_save += 1

                pbar.update(len(batch_ids))

                # Periodic checkpoint
                if save_path and batches_since_save >= checkpoint_every:
                    self.index.save(save_path)
                    batches_since_save = 0

        if save_path:
            self.index.save(save_path)

class DenseRetrieveUseCase(IDenseRetrievalService):
    def __init__(self, model: IEmbeddingModel, index: IVectorIndex):
        self.model = model
        self.index = index

    def retrieve(self, query: str, top_k: int = 10) -> List[VectorSearchResult]:
        query_vector = self.model.encode(query)
        return self.index.search(query_vector, top_k)

