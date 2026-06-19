from typing import List
from services.embeddings_service.domain.interfaces import IEmbeddingModel
from services.embeddings_service.domain.models import DenseVector

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class SentenceTransformerEmbedding(IEmbeddingModel):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS:
            # Lazy loading to save memory until actually used
            self.model = SentenceTransformer(self.model_name)
        else:
            raise ImportError("sentence-transformers is not installed. Please 'pip install sentence-transformers'.")

    def encode(self, text: str) -> DenseVector:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: List[str]) -> List[DenseVector]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
