import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

st.set_page_config(
    page_title="IR Explorer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semantic Explorer Platform")
st.markdown("""
Welcome to the professional Information Retrieval Platform.

Please select a page from the sidebar to begin:
- **🔍 Search**: The main unified search experience.
- **⚖️ Compare**: Side-by-side comparison of different retrieval algorithms.
- **📊 Analytics**: View corpus and engine metrics.
- **📈 Evaluation**: Review IR performance metrics.
""")

# Setup basic session state if missing
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"
