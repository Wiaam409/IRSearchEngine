import streamlit as st
import re
from typing import Callable
from ui.services.api_client import APIClient
from ui.services.state_manager import get_setting

def render_search_bar(api_client: APIClient, on_search: Callable[[str], None]):
    """Renders the main search bar with history and refinement."""
    col1, col2 = st.columns([5, 1])
    
    # Check if we should populate from history
    default_query = ""
    history_queries = [""] + [h["query"] for h in st.session_state.get("search_history", [])]
    
    with col1:
        # User input query
        query = st.text_input("Search", placeholder="Enter your query in Arabic (e.g., سيارة, تعليم...)", label_visibility="collapsed")
        
        # History dropdown for convenience
        if len(history_queries) > 1:
            selected_history = st.selectbox("Recent Searches", history_queries, label_visibility="collapsed")
            if selected_history and not query:
                query = selected_history
    
    with col2:
        search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        
    if search_clicked and query.strip():
        # Handle Query Refinement
        if get_setting("use_refinement"):
            with st.spinner("Refining query..."):
                ref_result = api_client.refine_query(query)
                if ref_result and ref_result.get("status") == "success":
                    refined = ref_result["data"].get("refined_query", query)
                    suggestions = ref_result["data"].get("suggestions", [])
                    if refined != query:
                        sug_text = f" *(Suggestions: {', '.join(suggestions)})*" if suggestions else ""
                        st.info(f"✨ Auto-corrected to: **{refined}**{sug_text}")
                        query = refined

        # Trigger the callback
        on_search(query)
    elif search_clicked:
        st.warning("Please enter a query to search.")

def highlight_terms(text: str, query: str) -> str:
    """Simple term highlighting for snippets."""
    if not query:
        return text
    terms = query.split()
    highlighted = text
    for term in terms:
        # Case insensitive replace with bold tag
        pattern = re.compile(f"({re.escape(term)})", re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)
    return highlighted
