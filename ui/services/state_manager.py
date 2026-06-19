import streamlit as st

def init_state():
    """Initializes default Streamlit session state variables."""
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = "http://localhost:8000"
    
    if "backend_healthy" not in st.session_state:
        st.session_state.backend_healthy = False
        
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
        
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": 10,
            "use_refinement": True,
            "method": "Hybrid Parallel",
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "hybrid_sparse_model": "bm25",
            "hybrid_fusion": "rrf",
            "hybrid_sparse_weight": 0.5,
            "hybrid_dense_weight": 0.5,
            "hybrid_serial_multiplier": 100
        }

def add_to_history(query: str, results_count: int, latency_ms: float, method: str):
    """Adds a query to the search history."""
    if not query.strip():
        return
        
    entry = {
        "query": query,
        "results": results_count,
        "latency": latency_ms,
        "method": method
    }
    
    # Avoid duplicate consecutive entries
    if st.session_state.search_history and st.session_state.search_history[0]["query"] == query:
        return
        
    st.session_state.search_history.insert(0, entry)
    
    # Keep only the last 20 queries
    if len(st.session_state.search_history) > 20:
        st.session_state.search_history = st.session_state.search_history[:20]

def update_setting(key: str, value: any):
    """Updates a global application setting."""
    if "settings" in st.session_state:
        st.session_state.settings[key] = value

def get_setting(key: str, default: any = None):
    """Retrieves a global application setting."""
    if "settings" in st.session_state:
        return st.session_state.settings.get(key, default)
    return default
