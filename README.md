# IRSearchEngine 🔍

An advanced Information Retrieval (IR) Engine built with a Service-Oriented Architecture (SOA). This project features a robust hybrid retrieval system combining classical lexical matching (BM25, TF-IDF) with modern semantic search (Dense Embeddings), topped with a Retrieval-Augmented Generation (RAG) pipeline for intelligent, context-aware question answering.

## 🌟 Key Features
- **Hybrid Search Strategy:** Seamlessly blends lexical and semantic retrieval for maximum recall and precision.
- **Retrieval-Augmented Generation (RAG):** Answers user queries natively using a Large Language Model (LLM) grounded by retrieved documents.
- **Query Refinement:** Automatically expands and cleans queries to handle vocabulary mismatch.
- **Comprehensive Evaluation:** Built-in evaluation service utilizing qrels to compute metrics like MAP, nDCG, Recall, and Precision@10.
- **Interactive UI:** Streamlit-based web interface for testing and visualizing search and RAG capabilities.

## 🏗️ System Architecture (SOA)

The engine is built around a **Service-Oriented Architecture (SOA)** via FastAPI, ensuring loose coupling, maintainability, and scalability. 

### Core Services:

1. **Preprocessing Service:** 
   - Handles text normalization, stop-word removal, and lemmatization/stemming.
   - Ensures the query and the documents undergo the exact same pipeline to avoid vocabulary mismatches.
2. **Indexing Service:**
   - Manages inverted indexes for lexical models (TF-IDF/BM25) and dense vector stores (e.g., FAISS) for semantic embeddings.
3. **Retrieval Service:** 
   - Executes the initial document fetch. Supports isolated retrieval (Lexical only / Semantic only) and **Hybrid Retrieval** (combining both concurrently).
4. **Ranking Service:**
   - Re-ranks the initially fetched documents based on relevance scores (Cosine Similarity for embeddings, scoring functions for BM25).
5. **Gateway / API Layer (FastAPI):**
   - Orchestrates communication between the UI and backend services.
6. **Evaluation Service:**
   - Evaluates system performance automatically using standard benchmarks and ground-truth `qrels`.
7. **RAG Service (Additional Feature):**
   - Interacts with an LLM to generate precise answers based solely on the context provided by the top retrieved documents.

## 🤖 RAG Pipeline

The RAG (Retrieval-Augmented Generation) capability solves the traditional IR problem of "link-listing" by directly answering complex questions.

1. **User Query:** The user submits a natural language question.
2. **Retrieval:** The Hybrid Retrieval engine fetches the top-K most relevant document chunks.
3. **Context Construction:** These chunks are synthesized into a grounded context window.
4. **LLM Generation:** The context and original query are passed to the LLM with strict prompts to prevent hallucination.
5. **Final Output:** The user receives a human-readable answer.

## 📊 Evaluation & Metrics

The system uses recognized ground-truth queries (`qrels`) from standard datasets (e.g., BEIR/MSMARCO). The evaluation service tracks:
- **MAP (Mean Average Precision)**
- **nDCG (Normalized Discounted Cumulative Gain)**
- **Recall**
- **Precision@10**

*(The hybrid approach demonstrates significant improvements across all metrics compared to baseline lexical retrieval).*

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Recommended: Virtual Environment

### Running the Services

1. **Start the FastAPI Backend Server:**
   ```bash
   python -m api.main
   ```

2. **Start the Streamlit UI:**
   ```bash
   python -m streamlit run ui/app.py
   ```

Navigate to `http://localhost:8501` to access the search interface.