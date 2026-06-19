import os
import json
import numpy as np
from typing import List
from services.embeddings_service.domain.interfaces import IVectorIndex
from services.embeddings_service.domain.models import DenseVector, VectorSearchResult

class InMemoryVectorIndex(IVectorIndex):
    def __init__(self):
        self.doc_ids: List[str] = []
        self.vectors: List[DenseVector] = []
        self._matrix: np.ndarray = None
        self._is_dirty = True

    def add(self, doc_id: str, vector: DenseVector):
        self.doc_ids.append(doc_id)
        self.vectors.append(vector)
        self._is_dirty = True

    def _build_matrix(self):
        if self._is_dirty and self.vectors:
            new_matrix = np.array(self.vectors, dtype=np.float32)
            if self._matrix is not None and self._matrix.size > 0:
                self._matrix = np.vstack([self._matrix, new_matrix])
            else:
                self._matrix = new_matrix
            self.vectors = []
            self._is_dirty = False

    def search(self, query_vector: DenseVector, k: int) -> List[VectorSearchResult]:
        self._build_matrix()
        if self._matrix is None or self._matrix.size == 0:
            return []
        q_vec = np.array(query_vector, dtype=np.float32)
        
        # Inner product (acts as cosine similarity since vectors are expected to be normalized)
        scores = np.dot(self._matrix, q_vec)
        
        k = min(k, len(self.doc_ids))
        
        if len(scores) > k:
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(scores)[::-1]

        results = []
        for idx in top_k_idx:
            results.append(VectorSearchResult(
                doc_id=self.doc_ids[idx],
                score=float(scores[idx])
            ))
            
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._build_matrix()
        
        npy_path = os.path.join(path, "index.npy")
        if self._matrix is not None:
            np.save(npy_path, self._matrix)
            
        doc_ids_path = os.path.join(path, "doc_ids.json")
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

    def load(self, path: str):
        npy_path = os.path.join(path, "index.npy")
        doc_ids_path = os.path.join(path, "doc_ids.json")
        
        if os.path.exists(doc_ids_path):
            with open(doc_ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
                
        if os.path.exists(npy_path):
            self._matrix = np.load(npy_path)
            self.vectors = []
            self._is_dirty = False
