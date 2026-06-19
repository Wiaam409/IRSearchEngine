import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import os
import json
from ui.components.layout import setup_page
from ui.components.charts import render_latency_chart, render_metric_cards

setup_page("Analytics")

st.markdown("## 📊 System Analytics")

# Mock or read actual stats
# In a real scenario, the backend should provide a /stats endpoint.
# We will simulate reading the index sizes locally for demonstration.

def get_dir_size(path):
    total = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
    return total

def get_corpus_stats():
    # Attempt to read the sample docs count or doc_ids
    docs_count = 0
    doc_ids_path = "datasets/cache/embeddings/doc_ids.json"
    if os.path.exists(doc_ids_path):
        try:
            with open(doc_ids_path, "r", encoding="utf-8") as f:
                docs_count = len(json.load(f))
        except:
            pass
    
    sparse_size = get_dir_size("datasets/cache/index")
    dense_size = get_dir_size("datasets/cache/embeddings")
    
    return docs_count, sparse_size + dense_size, docs_count * 120 # Mock vocab size

corpus, total_size, vocab = get_corpus_stats()

st.subheader("Index Overview")
render_metric_cards(corpus, total_size, vocab)

st.divider()

st.subheader("Performance Analytics")
history = st.session_state.get("search_history", [])
render_latency_chart(history)

st.divider()
st.subheader("Recent Queries")
if history:
    for h in history[:5]:
        st.write(f"- **{h['query']}** (Found {h['results']} in {h['latency']:.1f}ms via {h['method']})")
else:
    st.write("No queries executed yet in this session.")
