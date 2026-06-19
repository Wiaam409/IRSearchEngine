import streamlit as st

def status_badge(is_healthy: bool):
    """Renders a backend status indicator."""
    color = "var(--success-color)" if is_healthy else "var(--danger-color)"
    text = "Connected" if is_healthy else "Disconnected"
    
    st.markdown(f"""
        <div class="status-indicator">
            <div class="status-dot {'healthy' if is_healthy else ''}"></div>
            <span>{text}</span>
        </div>
    """, unsafe_allow_html=True)

def score_badge(score: float):
    """Renders a score badge."""
    st.markdown(f"""
        <span class="badge badge-score">Score: {score:.4f}</span>
    """, unsafe_allow_html=True)

def source_badge(source: str):
    """Renders a source/method badge."""
    st.markdown(f"""
        <span class="badge badge-source">{source}</span>
    """, unsafe_allow_html=True)

def rank_badge(rank: int):
    """Renders a rank badge."""
    st.markdown(f"""
        <span class="badge badge-rank">#{rank}</span>
    """, unsafe_allow_html=True)
