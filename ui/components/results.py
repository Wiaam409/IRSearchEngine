import streamlit as st
from ui.widgets.badges import score_badge, rank_badge, source_badge

def render_result_card(result: dict, rank: int):
    """Renders a single result as a modern card."""
    doc_id = result.get("doc_id", f"Unknown-{rank}")
    score = result.get("score", 0.0)
    title = result.get("title", "No Title Available")
    if not title or title.strip() == "":
        title = f"Document {doc_id}"
    text = result.get("text", "")
    
    # Truncate text for snippet
    snippet = text[:300] + "..." if len(text) > 300 else text
    
    # Determine source based on methods if hybrid
    source = result.get("retrieval_method", "BM25")
    if "sparse_score" in result and "dense_score" in result:
        source = "Hybrid"
    elif result.get("dense_score"):
        source = "Dense"
        
    card_html = f"""
    <div class="result-card rtl-text">
        <div class="result-card-header">
            <h3 class="result-card-title">{title}</h3>
        </div>
        <div class="result-card-body">
            {snippet}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    b_col1, b_col2, b_col3, b_col4 = st.columns([1, 1, 1, 5])
    with b_col1:
        rank_badge(rank)
    with b_col2:
        score_badge(score)
    with b_col3:
        source_badge(source)
    
    with st.expander("Show Full Document & Meta"):
        st.markdown(f"<div class='rtl-text' style='padding: 10px; background: var(--bg-main); border-radius: 5px;'>{text}</div>", unsafe_allow_html=True)
        st.divider()
        st.json(result)
    
    st.write("")
