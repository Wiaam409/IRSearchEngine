import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
from ui.components.layout import setup_page

setup_page("Evaluation")

st.markdown("## 📈 Model Evaluation Metrics")
st.write("These metrics represent performance on the Beir validation set.")

import json

metrics_file = "datasets/cache/evaluation_metrics.json"

if os.path.exists(metrics_file):
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.success("Displaying real metrics from the latest offline evaluation run.")
else:
    st.warning("⚠️ Real evaluation metrics not found. Showing placeholder data. Run `python scripts/run_evaluation.py` to generate real metrics.")
    metrics_data = {
        "Model": ["TF-IDF", "BM25", "Embeddings", "Hybrid (RRF)", "Hybrid (Serial)"],
        "NDCG@10": [0.42, 0.48, 0.65, 0.72, 0.71],
        "Recall@100": [0.75, 0.82, 0.91, 0.95, 0.94],
        "MAP": [0.38, 0.45, 0.58, 0.68, 0.66]
    }

df = pd.DataFrame(metrics_data)

st.subheader("Leaderboard")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    fig_ndcg = px.bar(df, x="Model", y="NDCG@10", title="NDCG@10 Comparison", color="Model")
    fig_ndcg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ndcg, use_container_width=True)

with col2:
    fig_recall = px.bar(df, x="Model", y="Recall@100", title="Recall@100 Comparison", color="Model")
    fig_recall.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_recall, use_container_width=True)
