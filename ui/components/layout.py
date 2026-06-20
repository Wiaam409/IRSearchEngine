import streamlit as st
import os
from ui.widgets.badges import status_badge
from ui.services.state_manager import init_state, get_setting, update_setting

def load_css():
    """Loads custom CSS styles."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_navbar():
    """Renders the top navigation bar."""
    is_healthy = st.session_state.get("backend_healthy", False)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
            <div class="top-nav">
                <div class="top-nav-title">
                    🔍 Semantic Explorer <span style="font-size: 0.8rem; font-weight: 400; color: var(--text-muted);"></span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        # Align to right nicely using CSS but for Streamlit we just use markdown
        st.markdown('<div style="text-align: right; padding-top: 1rem;">', unsafe_allow_html=True)
        status_badge(is_healthy)
        st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar():
    """Renders the unified sidebar controls."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Backend URL config
        new_url = st.text_input("Backend URL", value=st.session_state.get("backend_url", "http://localhost:8000"))
        if new_url != st.session_state.get("backend_url"):
            st.session_state.backend_url = new_url
            
        st.divider()
        
        # General Settings
        new_top_k = st.slider("Top K Results", 1, 50, get_setting("top_k"))
        if new_top_k != get_setting("top_k"):
            update_setting("top_k", new_top_k)
            
        new_refine = st.checkbox("Enable Query Refinement", value=get_setting("use_refinement"))
        if new_refine != get_setting("use_refinement"):
            update_setting("use_refinement", new_refine)

        st.divider()
        
        # Retrieval Model
        st.subheader("Retrieval Model")
        methods = ["TF-IDF", "BM25", "Embeddings", "Hybrid Parallel", "Hybrid Serial", "RAG"]
        new_method = st.selectbox("Method", methods, index=methods.index(get_setting("method")))
        if new_method != get_setting("method"):
            update_setting("method", new_method)
            
        # Model-specific settings
        if new_method == "BM25":
            with st.expander("BM25 Parameters", expanded=True):
                k1 = st.number_input("k1", 0.0, 3.0, get_setting("bm25_k1"), 0.1)
                b = st.number_input("b", 0.0, 1.0, get_setting("bm25_b"), 0.05)
                update_setting("bm25_k1", k1)
                update_setting("bm25_b", b)
                
        elif new_method == "Hybrid Parallel":
            with st.expander("Hybrid Parameters", expanded=True):
                sparse = st.selectbox("Sparse Model", ["bm25", "tfidf"], index=["bm25", "tfidf"].index(get_setting("hybrid_sparse_model")))
                fusion = st.selectbox("Fusion Method", ["rrf", "score"], index=["rrf", "score"].index(get_setting("hybrid_fusion")))
                s_weight = st.slider("Sparse Weight", 0.0, 1.0, get_setting("hybrid_sparse_weight"), 0.1)
                d_weight = st.slider("Dense Weight", 0.0, 1.0, get_setting("hybrid_dense_weight"), 0.1)
                update_setting("hybrid_sparse_model", sparse)
                update_setting("hybrid_fusion", fusion)
                update_setting("hybrid_sparse_weight", s_weight)
                update_setting("hybrid_dense_weight", d_weight)
                
        elif new_method == "Hybrid Serial":
            with st.expander("Serial Parameters", expanded=True):
                mult = st.number_input("Candidate Multiplier", 10, 500, get_setting("hybrid_serial_multiplier"), 10)
                update_setting("hybrid_serial_multiplier", mult)
                
def setup_page(title: str):
    """Common page setup including CSS, layout, and state initialization."""
    st.set_page_config(page_title=f"{title} | IR Explorer", layout="wide", page_icon="🔍")
    init_state()
    load_css()
    render_navbar()
    render_sidebar()
