import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.adapters.beir_adapter import BeirAdapter
from services.indexing_service.infrastructure.json_index import JsonIndexWriter
from services.indexing_service.application.use_cases import BuildIndexUseCase
from services.preprocessing_service.application.pipeline import PreprocessingPipeline
from services.preprocessing_service.application.english_processors import (
    EnglishNormalizer, EnglishTokenizer, EnglishStopwordRemover
)
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
from services.embeddings_service.infrastructure.embedding_models import SentenceTransformerEmbedding
from services.embeddings_service.application.use_cases import BuildDenseIndexUseCase

def build_index(limit: int = None, force: bool = False):
    print(f"Building indexes. Limit: {'Full Dataset' if limit is None else limit} docs.")

    index_dir = "datasets/cache/index"
    emb_dir = "datasets/cache/embeddings"
    
    sparse_exists = os.path.exists(os.path.join(index_dir, "inverted_index.json"))
    dense_exists = os.path.exists(os.path.join(emb_dir, "index.npy"))

    if sparse_exists and dense_exists and not force:
        print("Indexes already exist in cache! Use --force to rebuild.")
        return

    # 1. Load documents
    adapter = BeirAdapter()
    docs = {}
    print("Loading documents...")
    for i, doc in tqdm(enumerate(adapter.load_documents())):
        if limit is not None and i >= limit:
            break
        # Combine title and text if available, otherwise just use text
        title = getattr(doc, 'title', '')
        text = getattr(doc, 'text', str(doc))
        docs[doc.doc_id] = f"{title} {text}".strip()
    
    print(f"Loaded {len(docs)} documents.")

    # 2. Sparse (Inverted) Index
    if not sparse_exists or force:
        os.makedirs(index_dir, exist_ok=True)

        idx = JsonIndexWriter(index_dir)
        pipeline = PreprocessingPipeline(
            normalizer=EnglishNormalizer(),
            tokenizer=EnglishTokenizer(),
            stopword_remover=EnglishStopwordRemover()
        )

        print("\nPreprocessing documents for sparse index...")
        processed = []
        for d_id, text in tqdm(docs.items(), desc="Preprocessing"):
            processed.append((d_id, pipeline.process(text)))

        use_case = BuildIndexUseCase(idx)
        use_case.execute(processed)
        print(f"[OK] Sparse index written to: {index_dir}/")
    else:
        print(f"[OK] Sparse index already exists in: {index_dir}/ (skipping)")

    # 3. Dense (Embedding) Index
    if not dense_exists or force:
        os.makedirs(emb_dir, exist_ok=True)

        print("\nLoading embedding model (this may take a moment)...")
        emb_model = SentenceTransformerEmbedding()
        vec_index = InMemoryVectorIndex()

        # Load existing partial index for resume support
        if os.path.exists(os.path.join(emb_dir, "doc_ids.json")) and not force:
            print("Loading existing partial dense index for resume...")
            vec_index.load(emb_dir)
            print(f"Loaded {len(vec_index.doc_ids)} previously embedded docs.")

        dense_use_case = BuildDenseIndexUseCase(emb_model, vec_index)
        print("Building dense embeddings...")
        dense_use_case.build(docs, save_path=emb_dir)
        print(f"[OK] Dense index written to: {emb_dir}/")
    else:
        print(f"[OK] Dense index already exists in: {emb_dir}/ (skipping)")

    print("\n[DONE] All indexes built successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build indexes from Beir dataset")
    parser.add_argument("--limit", type=int, default=None, help="Number of documents to index (e.g., 1000). If not provided, it will index the full dataset.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of indexes even if they already exist.")
    args = parser.parse_args()

    build_index(limit=args.limit, force=args.force)
