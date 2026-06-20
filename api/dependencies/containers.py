from functools import lru_cache
from api.config import settings

from services.preprocessing_service.application.pipeline import PreprocessingPipeline
from services.preprocessing_service.application.english_processors import (
    EnglishNormalizer, EnglishTokenizer, EnglishStopwordRemover
)

from services.retrieval_service.infrastructure.tfidf_index_reader import TfidfIndexReader
from services.retrieval_service.infrastructure.tfidf_scorer import TfidfScorer
from services.retrieval_service.infrastructure.query_processor_adapter import QueryProcessorAdapter as TfidfQueryProcessor
from services.retrieval_service.application.use_cases import TfidfRetrieveUseCase

from services.ranking_service.infrastructure.bm25_index_reader import Bm25IndexReader
from services.ranking_service.infrastructure.bm25_scorer import Bm25Scorer
from services.ranking_service.infrastructure.query_processor_adapter import QueryProcessorAdapter as Bm25QueryProcessor
from services.ranking_service.application.use_cases import Bm25RankUseCase

from services.embeddings_service.infrastructure.embedding_models import SentenceTransformerEmbedding
from services.embeddings_service.infrastructure.vector_index import InMemoryVectorIndex
from services.embeddings_service.application.use_cases import DenseRetrieveUseCase

from services.query_refinement_service.infrastructure.file_synonym_provider import FileBasedSynonymProvider
from services.query_refinement_service.infrastructure.spell_checkers import NoOpSpellChecker, SimpleSpellChecker
from services.query_refinement_service.application.use_cases import RefineQueryUseCase

from services.hybrid_service.infrastructure.adapters import RetrieverAdapter
from services.hybrid_service.infrastructure.fusion_strategies import RrFusionStrategy, ScoreFusionStrategy
from services.hybrid_service.application.use_cases import ParallelHybridRetrieveUseCase, SerialHybridRetrieveUseCase

from services.evaluation_service.infrastructure.metric_calculator import MetricCalculator
from services.evaluation_service.application.use_cases import EvaluateModelUseCase

def _make_pipeline():
    return PreprocessingPipeline(
        normalizer=EnglishNormalizer(),
        tokenizer=EnglishTokenizer(),
        stopword_remover=EnglishStopwordRemover()
    )

@lru_cache()
def get_tfidf_service() -> TfidfRetrieveUseCase:
    reader = TfidfIndexReader(settings.index_cache_dir)
    scorer = TfidfScorer(reader)
    qp = TfidfQueryProcessor(_make_pipeline())
    return TfidfRetrieveUseCase(qp, reader, scorer)

@lru_cache()
def get_bm25_service() -> Bm25RankUseCase:
    reader = Bm25IndexReader(settings.index_cache_dir)
    scorer = Bm25Scorer()
    qp = Bm25QueryProcessor(_make_pipeline())
    return Bm25RankUseCase(qp, reader, scorer)

@lru_cache()
def get_dense_service() -> DenseRetrieveUseCase:
    model = SentenceTransformerEmbedding()
    index = InMemoryVectorIndex()
    index.load(settings.embeddings_cache_dir)
    return DenseRetrieveUseCase(model, index)

@lru_cache()
def get_spell_checker() -> SimpleSpellChecker:
    service = get_tfidf_service()
    vocab = set(service.reader.idf_cache.keys()) if hasattr(service, 'reader') and hasattr(service.reader, 'idf_cache') else set()
    return SimpleSpellChecker(dictionary=vocab, max_distance=1)

@lru_cache()
def get_refinement_service() -> RefineQueryUseCase:
    synonym_provider = FileBasedSynonymProvider(settings.synonyms_path)
    spell_checker = get_spell_checker()
    return RefineQueryUseCase(spell_checker, synonym_provider)

@lru_cache()
def get_llm_pipeline():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        def generator(prompt, max_new_tokens=50):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return [{"generated_text": answer}]
            
        return generator
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        return None

@lru_cache()
def get_evaluation_service() -> EvaluateModelUseCase:
    from services.evaluation_service.infrastructure.qrel_loader import QrelFileLoader
    reader = QrelFileLoader()
    calculator = MetricCalculator()
    return EvaluateModelUseCase(reader, calculator)

def get_parallel_hybrid_service(sparse_model: str, fusion: str) -> ParallelHybridRetrieveUseCase:
    if sparse_model == "bm25":
        sparse_use_case = get_bm25_service()
    else:
        sparse_use_case = get_tfidf_service()
        
    dense_use_case = get_dense_service()
    
    sparse_adapter = RetrieverAdapter(sparse_use_case)
    dense_adapter = RetrieverAdapter(dense_use_case)
    
    strategy = RrFusionStrategy() if fusion == "rrf" else ScoreFusionStrategy()
    
    return ParallelHybridRetrieveUseCase([sparse_adapter, dense_adapter], strategy)

@lru_cache()
def get_serial_hybrid_service() -> SerialHybridRetrieveUseCase:
    sparse_adapter = RetrieverAdapter(get_bm25_service())
    dense_adapter = RetrieverAdapter(get_dense_service())
    return SerialHybridRetrieveUseCase(sparse_adapter, dense_adapter)
