import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.adapters.beir_adapter import BeirAdapter
from api.dependencies.containers import get_rag_service, get_llm_pipeline

def evaluate_rag():
    print("Starting RAG Evaluation...")
    
    # 1. Load some test queries (from BEIR)
    adapter = BeirAdapter()
    
    # For generative evaluation, it's common to use a smaller subset (e.g., 10 queries)
    # because running generation and LLM-as-a-judge is computationally expensive.
    queries = []
    for i, q in enumerate(adapter.load_queries()):
        if i >= 10:
            break
        queries.append(q.text)
        
    # 2. Get services
    rag_service = get_rag_service()
    
    # We will repurpose the local LLM to act as our "Judge" for evaluation.
    # In a production system, you might use a stronger model (like GPT-4) for the judge.
    llm_judge = get_llm_pipeline() 
    
    if not llm_judge:
        print("Error: LLM pipeline is not available for evaluation.")
        return

    results = []
    
    # 3. Evaluation Loop
    for query in tqdm(queries, desc="Evaluating RAG Generation"):
        # Run standard RAG pipeline
        rag_output = rag_service.generate_answer(query, top_k=3)
        
        # --- LLM-as-a-Judge Metrics ---
        
        # Metric A: Faithfulness / Groundedness
        # Checks if the generated answer is hallucinating or if it strictly relies on the context.
        faithfulness_prompt = f"""Determine if the answer is completely faithful to the context (Yes or No).
Context:
{rag_output.context}

Answer:
{rag_output.answer}

Faithful:
"""
        faith_res = llm_judge(faithfulness_prompt, max_new_tokens=5)
        faithfulness_score = 1 if "yes" in faith_res[0]['generated_text'].lower() else 0
        
        # Metric B: Answer Relevance
        # Checks if the generated answer actually answers the user's question.
        relevance_prompt = f"""Determine if the answer directly and accurately answers the question (Yes or No).
Question:
{query}

Answer:
{rag_output.answer}

Relevant:
"""
        rel_res = llm_judge(relevance_prompt, max_new_tokens=5)
        relevance_score = 1 if "yes" in rel_res[0]['generated_text'].lower() else 0
        
        results.append({
            "query": query,
            "answer": rag_output.answer,
            "faithfulness": faithfulness_score,
            "relevance": relevance_score
        })
        
    # 4. Summarize and Print
    avg_faithfulness = sum(r['faithfulness'] for r in results) / len(results)
    avg_relevance = sum(r['relevance'] for r in results) / len(results)
    
    print("\n=== RAG Evaluation Summary ===")
    print(f"Total Queries Evaluated: {len(results)}")
    print(f"Faithfulness Score (Groundedness): {avg_faithfulness:.2f}")
    print(f"Answer Relevance Score: {avg_relevance:.2f}")
    
    # Save to disk
    os.makedirs("datasets/cache", exist_ok=True)
    with open("datasets/cache/rag_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("Detailed results saved to datasets/cache/rag_evaluation.json")

if __name__ == "__main__":
    evaluate_rag()
