import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import time
from ui.components.layout import setup_page
from ui.components.search import render_search_bar
from ui.components.results import render_result_card
from ui.components.charts import render_score_distribution
from ui.services.api_client import APIClient
from ui.services.state_manager import get_setting, add_to_history

setup_page("Search")

api_client = APIClient(st.session_state.backend_url)
st.session_state.backend_healthy = api_client.check_health()

def perform_search(query: str):
    method = get_setting("method")
    top_k = get_setting("top_k")
    
    start_time = time.time()
    result = None
    
    if method == "TF-IDF":
        result = api_client.search_tfidf(query, top_k)
    elif method == "BM25":
        result = api_client.search_bm25(query, top_k, get_setting("bm25_k1"), get_setting("bm25_b"))
    elif method == "Embeddings":
        result = api_client.search_embeddings(query, top_k)
    elif method == "Hybrid Parallel":
        result = api_client.search_hybrid_parallel(
            query, top_k, 
            get_setting("hybrid_sparse_model"), 
            get_setting("hybrid_fusion"), 
            get_setting("hybrid_sparse_weight"), 
            get_setting("hybrid_dense_weight")
        )
    elif method == "Hybrid Serial":
        result = api_client.search_hybrid_serial(query, top_k, get_setting("hybrid_serial_multiplier"))

    latency_ms = (time.time() - start_time) * 1000
    
    if result and result.get("status") == "success":
        results_list = result.get("data", {}).get("results", [])
        
        # Add to history
        add_to_history(query, len(results_list), latency_ms, method)
        
        st.success(f"Found {len(results_list)} results in {latency_ms:.0f} ms using {method}")
        
        if results_list:
            col1, col2 = st.columns([3, 1])
            with col1:
                for i, res in enumerate(results_list):
                    render_result_card(res, i + 1)
            with col2:
                render_score_distribution(results_list)
        else:
            st.info("No documents found matching your query.")
    else:
        st.error("Failed to retrieve results from the backend.")

render_search_bar(api_client, perform_search)
