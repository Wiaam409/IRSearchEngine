import streamlit as st
import requests
import time
import pandas as pd

# Page config
st.set_page_config(page_title="IR Search Engine", layout="wide", page_icon="🔍")

# Custom CSS for modern professional aesthetic
st.markdown("""
<style>
    /* Add any custom styling here, e.g., accent color */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔍 Information Retrieval Search Engine")
st.markdown("Search over Arabic documents using TF-IDF, BM25, Embeddings, or Hybrid retrieval.")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Configuration")
    backend_url = st.text_input("Backend URL", value="http://localhost:8000")
    top_k = st.slider("Number of results (top-K)", min_value=1, max_value=50, value=10)
    use_refinement = st.checkbox("Enable Query Refinement (spell + synonyms)")

    st.divider()
    method = st.selectbox("Retrieval Method", ["TF-IDF", "BM25", "Embeddings", "Hybrid Parallel", "Hybrid Serial"])

    # Method-specific parameters
    params = {}
    if method == "BM25":
        col1, col2 = st.columns(2)
        with col1:
            params["k1"] = st.number_input("k1", value=1.2, step=0.1, format="%.2f")
        with col2:
            params["b"] = st.number_input("b", value=0.75, step=0.05, format="%.2f")
    elif method == "Hybrid Parallel":
        params["sparse_model"] = st.selectbox("Sparse Model", ["bm25", "tfidf"])
        params["fusion"] = st.selectbox("Fusion Method", ["rrf", "score"])
        col1, col2 = st.columns(2)
        with col1:
            params["sparse_weight"] = st.slider("Sparse Weight", 0.0, 1.0, 0.5, 0.1)
        with col2:
            params["dense_weight"] = st.slider("Dense Weight", 0.0, 1.0, 0.5, 0.1)
    elif method == "Hybrid Serial":
        params["candidate_multiplier"] = st.number_input("Candidate Multiplier", value=100, step=10)

# Main area: query input and search button
query = st.text_input("Enter your search query (Arabic):", placeholder="مثال: تعليم, سيارة, ...")

col1, col2 = st.columns([1, 5])
with col1:
    search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)

# Function definitions
def call_api(endpoint, payload, backend_url):
    url = f"{backend_url.rstrip('/')}/{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {backend_url}. Is the API running?")
        return None
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return None

def refine_query(query, backend_url, use_refinement):
    if not use_refinement or not query:
        return query
    payload = {"query": query, "spell_check": True, "expand_synonyms": True}
    result = call_api("refine", payload, backend_url)
    if result and result.get("status") == "success":
        refined = result["data"].get("refined_query", query)
        suggestions = result["data"].get("suggestions", [])
        if refined != query:
            st.info(f"✨ Refined query: **{refined}**" + (f" (suggestions: {', '.join(suggestions)})" if suggestions else ""))
        return refined
    return query

def perform_search(method, query, top_k, backend_url, params):
    if method == "TF-IDF":
        payload = {"query": query, "top_k": top_k}
        return call_api("search/tfidf", payload, backend_url)
    elif method == "BM25":
        payload = {"query": query, "top_k": top_k, "k1": params.get("k1", 1.2), "b": params.get("b", 0.75)}
        return call_api("search/bm25", payload, backend_url)
    elif method == "Embeddings":
        payload = {"query": query, "top_k": top_k}
        return call_api("search/embeddings", payload, backend_url)
    elif method == "Hybrid Parallel":
        payload = {
            "query": query, "top_k": top_k,
            "sparse_model": params.get("sparse_model", "bm25"),
            "fusion": params.get("fusion", "rrf"),
            "sparse_weight": params.get("sparse_weight", 0.5),
            "dense_weight": params.get("dense_weight", 0.5)
        }
        return call_api("search/hybrid/parallel", payload, backend_url)
    elif method == "Hybrid Serial":
        payload = {"query": query, "top_k": top_k, "candidate_multiplier": params.get("candidate_multiplier", 100)}
        return call_api("search/hybrid/serial", payload, backend_url)
    else:
        st.error("Unknown method")
        return None

# Search logic
if search_clicked and query:
    with st.spinner("Processing..."):
        start_time = time.time()
        # Step 1: refine query if enabled
        refined_q = refine_query(query, backend_url, use_refinement)
        # Step 2: perform search with refined query (or original)
        result = perform_search(method, refined_q, top_k, backend_url, params)
        elapsed_ms = (time.time() - start_time) * 1000

        if result and result.get("status") == "success":
            data = result.get("data", {})
            results_list = data.get("results", [])
            st.success(f"Found {len(results_list)} results in {elapsed_ms:.0f} ms")
            if results_list:
                # Prepare dataframe
                df = pd.DataFrame(results_list)
                df.index = range(1, len(df)+1)
                df.index.name = "Rank"
                # Add visual score bar formatted
                if "score" in df.columns:
                    df["Score"] = df["score"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(df[["doc_id", "Score"]], use_container_width=True)
                else:
                    st.dataframe(df[["doc_id"]], use_container_width=True)
                
                # Optional: show detailed view
                with st.expander("View raw response"):
                    st.json(result)
            else:
                st.warning("No results found.")
        elif result is None:
            # error already shown by call_api
            pass
        else:
            st.error("Unexpected response format")
elif search_clicked and not query:
    st.warning("Please enter a query.")
