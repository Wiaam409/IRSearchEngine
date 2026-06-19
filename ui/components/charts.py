import streamlit as st
import pandas as pd
import plotly.express as px

def render_score_distribution(results: list):
    """Renders a histogram of retrieval scores."""
    if not results or "score" not in results[0]:
        return
        
    scores = [float(r["score"]) for r in results]
    df = pd.DataFrame({"Score": scores})
    
    fig = px.histogram(
        df, 
        x="Score", 
        nbins=20, 
        title="Score Distribution",
        color_discrete_sequence=["#005A9E"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)

def render_latency_chart(history: list):
    """Renders a line chart of query latencies from history."""
    if not history:
        st.info("No history available for latency chart.")
        return
        
    hist = list(reversed(history))
    df = pd.DataFrame({
        "Query": [h["query"] for h in hist],
        "Latency (ms)": [h["latency"] for h in hist]
    })
    
    fig = px.line(
        df, 
        y="Latency (ms)", 
        title="Recent Query Latency",
        markers=True
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
def render_metric_cards(corpus_size: int, index_size: int, vocab_size: int):
    """Renders high-level metrics."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Corpus Size (Docs)", f"{corpus_size:,}")
    col2.metric("Index Size (Bytes)", f"{index_size:,}")
    col3.metric("Vocabulary Size", f"{vocab_size:,}")
