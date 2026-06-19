# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
Build Dummy Index Script
========================
Creates minimal dummy indexes so you can test the Streamlit UI + REST API
without needing the full MIRACL dataset.

Run from project root:
    python build_dummy_index.py
"""
import os
import json

def build():
    print("Building dummy indexes for testing...")

    # ─── 1. Sparse (Inverted) Index ───────────────────────────────────────────
    index_dir = "datasets/cache/index"
    os.makedirs(index_dir, exist_ok=True)

    from services.indexing_service.infrastructure.json_index import JsonIndexWriter
    from services.indexing_service.application.use_cases import BuildIndexUseCase
    from services.preprocessing_service.application.pipeline import PreprocessingPipeline
    from services.preprocessing_service.application.arabic_processors import (
        ArabicNormalizer, ArabicTokenizer, ArabicStopwordRemover
    )

    idx = JsonIndexWriter(index_dir)
    pipeline = PreprocessingPipeline(
        normalizer=ArabicNormalizer(),
        tokenizer=ArabicTokenizer(),
        stopword_remover=ArabicStopwordRemover(set())
    )

    docs = {
        "doc1": "هذه سيارة جديدة وجميلة",
        "doc2": "اشتريت سيارة حمراء سريعة",
        "doc3": "التعليم المدرسي مهم جدا للأجيال",
        "doc4": "سيارة أخرى للبيع بسعر مناسب",
        "doc5": "المعلومات والبيانات في الذكاء الاصطناعي",
        "doc6": "استرجاع المعلومات علم قديم ومفيد",
    }

    print("\nPreprocessing documents:")
    processed = [(d_id, pipeline.process(text)) for d_id, text in docs.items()]
    for d_id, tokens in processed:
        print(f"  {d_id}: {len(tokens)} tokens")

    use_case = BuildIndexUseCase(idx)
    use_case.execute(processed)
    print(f"\n✓ Sparse index written to: {index_dir}/")

    # ─── 2. Dense (Embedding) Index ───────────────────────────────────────────
    emb_dir = "datasets/cache/embeddings"
    os.makedirs(emb_dir, exist_ok=True)

    from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
    from services.embeddings_service.infrastructure.embedding_models import SentenceTransformerEmbedding
    from services.embeddings_service.application.use_cases import BuildDenseIndexUseCase

    print("\nLoading embedding model (this may take a moment)...")
    emb_model = SentenceTransformerEmbedding()
    vec_index = InMemoryVectorIndex()

    dense_use_case = BuildDenseIndexUseCase(emb_model, vec_index)
    # save() expects a directory path
    dense_use_case.build(docs, save_path=emb_dir)
    print(f"✓ Dense index written to: {emb_dir}/")

    # ─── 3. Synonyms file ─────────────────────────────────────────────────────
    syn_dir = "datasets/miracl/ar"
    os.makedirs(syn_dir, exist_ok=True)
    synonyms = {
        "سيار":   ["سيارة", "مركبة"],
        "سياره":  ["سيارة"],
        "تعلم":   ["تعليم", "دراسة"],
        "معلومه": ["معلومات", "بيانات"],
    }
    syn_path = os.path.join(syn_dir, "synonyms.json")
    with open(syn_path, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    print(f"✓ Synonyms written to: {syn_path}")

    print("\n✅ All dummy indexes built successfully!")
    print("You can now test the Streamlit UI at http://localhost:8501")

if __name__ == "__main__":
    build()
