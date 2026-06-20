from typing import Callable, Any
from services.rag_service.domain.models import RagResult

class RagUseCase:
    def __init__(self, retriever_service: Any, document_store: Any, llm_generator: Callable):
        self.retriever = retriever_service
        self.document_store = document_store
        self.llm_generator = llm_generator

    def generate_answer(self, query: str, top_k: int) -> RagResult:
        # 1. Retrieve documents using the injected retriever (e.g. DenseRetrieveUseCase)
        # Note: Dense uses 'retrieve', BM25 uses 'rank'
        # We will assume DenseRetrieveUseCase here, which uses .retrieve(query, top_k)
        result = self.retriever.retrieve(query, top_k)
        
        # 2. Fetch full text for context
        context_texts = []
        for d in result:
            text = self.document_store.get_document_text(d.doc_id)
            if text: 
                context_texts.append(text[:500])
                
        context = " ".join(context_texts)
        
        # 3. Generate Answer
        try:
            if self.llm_generator:
                prompt = f"Answer the question using ONLY the context below.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:\n"
                res = self.llm_generator(prompt, max_new_tokens=50)
                answer = res[0]['generated_text'].strip()
            else:
                answer = f"[LLM Offline Mode] Retrieved Context Summary: {context[:200]}..."
        except Exception:
            answer = f"[LLM Offline Mode] Retrieved Context Summary: {context[:200]}..."
            
        return RagResult(
            answer=answer,
            context=context,
            documents=result
        )
