import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
from ui.components.layout import setup_page
from ui.services.api_client import APIClient
from ui.widgets.badges import score_badge, rank_badge

setup_page("Compare Models")

st.markdown("## ⚖️ Compare Retrieval Models")
st.write("Enter a query to see how different algorithms rank the documents.")

api_client = APIClient(st.session_state.backend_url)

query = st.text_input("Comparison Query (Arabic):", placeholder="Example: سيارة سريعة")
top_k = st.slider("Top K", 1, 20, 5)

if st.button("Compare", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching results from multiple models..."):
            res_bm25 = api_client.search_bm25(query, top_k, 1.2, 0.75)
            res_dense = api_client.search_embeddings(query, top_k)
            res_hybrid = api_client.search_hybrid_parallel(query, top_k, "bm25", "rrf", 0.5, 0.5)
            
            c1, c2, c3 = st.columns(3)
            
            def render_col(col, title, res):
                with col:
                    st.subheader(title)
                    if res and res.get("status") == "success":
                        docs = res["data"]["results"]
                        for i, d in enumerate(docs):
                            with st.container():
                                st.markdown(f"**{d.get('title', 'Doc ' + d['doc_id'])}**")
                                rank_badge(i+1)
                                score_badge(d['score'])
                                st.caption(d.get('text', '')[:100] + "...")
                                st.divider()
                    else:
                        st.error("Failed to fetch.")

            render_col(c1, "Sparse (BM25)", res_bm25)
            render_col(c2, "Dense (Embeddings)", res_dense)
            render_col(c3, "Hybrid (RRF)", res_hybrid)
