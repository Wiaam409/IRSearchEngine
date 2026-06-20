# IR Search Engine Architecture

This document visualizes the Service-Oriented Architecture (SOA) of the IR Search Engine.

## Core System Architecture

```mermaid
graph TD
    UI[Streamlit UI] --> API[FastAPI Gateway]
    
    API --> Hybrid[Hybrid Service]
    API --> Retrieval[Retrieval Service]
    API --> Ranking[Ranking Service]
    API --> Embeddings[Embeddings Service]
    API --> Refinement[Query Refinement Service]
    
    Refinement -->|Spellcheck| IndexCache[(Index Cache)]
    
    Hybrid --> Retrieval
    Hybrid --> Embeddings
    
    Retrieval -->|TF-IDF| Preprocessing[Preprocessing Pipeline]
    Ranking -->|BM25| Preprocessing
    
    Retrieval --> IndexCache
    Ranking --> IndexCache
    Embeddings --> EmbCache[(Embeddings Cache)]
```

## Retrieval Pipeline

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Refinement as Query Refinement
    participant Hybrid as Hybrid Retriever
    participant Preprocess as Preprocessing
    participant BM25
    participant Dense
    
    User->>API: Search("سياراة سريعة")
    API->>Refinement: refine("سياراة سريعة")
    Refinement-->>API: "سيارة سريعة"
    
    API->>Hybrid: retrieve("سيارة سريعة")
    
    par BM25 Thread
        Hybrid->>BM25: rank()
        BM25->>Preprocess: tokenize & stem
        Preprocess-->>BM25: ["سيار", "سريع"]
        BM25-->>Hybrid: Sparse Scores
    and Dense Thread
        Hybrid->>Dense: retrieve()
        Dense-->>Hybrid: Dense Scores
    end
    
    Hybrid->>Hybrid: RRF / Score Fusion
    Hybrid-->>API: Final Ranked Documents
    API-->>User: Search Results
```

## Evaluation Pipeline

```mermaid
graph LR
    Dataset[BEIR Dataset] --> Qrels[Qrels Loader]
    Dataset --> Queries[Queries Loader]
    
    Qrels --> Eval[Evaluation Service]
    Queries --> Eval
    
    Eval --> BM25[BM25]
    Eval --> BM25R[BM25 Refined]
    Eval --> TFIDF[TF-IDF]
    Eval --> Hybrid[Hybrid]
    
    BM25 --> Eval
    BM25R --> Eval
    TFIDF --> Eval
    Hybrid --> Eval
    
    Eval --> MetricCalc[Metric Calculator]
    MetricCalc --> Output[(evaluation_metrics.json)]
    Output --> Jupyter[Jupyter Notebook]
```
