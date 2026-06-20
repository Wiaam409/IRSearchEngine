import requests
import streamlit as st

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.timeout = 300  # Increased to 5 minutes to allow downloading the RAG model and building doc store

    def _post(self, endpoint: str, payload: dict):
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                err_data = e.response.json()
                st.error(f"API Backend Error: {err_data.get('detail', str(e))}")
            except:
                st.error(f"API Error: {str(e)}")
            return None
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at {self.base_url}. Is the API running?")
            return None
        except Exception as e:
            st.error(f"API error: {str(e)}")
            return None

    def _get(self, endpoint: str):
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def check_health(self) -> bool:
        """Checks if the backend API is reachable."""
        res = self._get("health")
        return res is not None and res.get("data", {}).get("status") == "healthy"

    def refine_query(self, query: str) -> dict:
        """Calls the /refine endpoint."""
        payload = {"query": query, "spell_check": True, "expand_synonyms": True}
        return self._post("refine", payload)

    def search_tfidf(self, query: str, top_k: int) -> dict:
        payload = {"query": query, "top_k": top_k}
        return self._post("search/tfidf", payload)

    def search_bm25(self, query: str, top_k: int, k1: float, b: float) -> dict:
        payload = {"query": query, "top_k": top_k, "k1": k1, "b": b}
        return self._post("search/bm25", payload)

    def search_embeddings(self, query: str, top_k: int) -> dict:
        payload = {"query": query, "top_k": top_k}
        return self._post("search/embeddings", payload)

    def search_hybrid_parallel(self, query: str, top_k: int, sparse_model: str, fusion: str, sparse_weight: float, dense_weight: float) -> dict:
        payload = {
            "query": query, 
            "top_k": top_k,
            "sparse_model": sparse_model,
            "fusion": fusion,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight
        }
        return self._post("search/hybrid/parallel", payload)

    def search_hybrid_serial(self, query: str, top_k: int, candidate_multiplier: int) -> dict:
        payload = {
            "query": query, 
            "top_k": top_k, 
            "candidate_multiplier": candidate_multiplier
        }
        return self._post("search/hybrid/serial", payload)

    def search_rag(self, query: str, top_k: int) -> dict:
        payload = {"query": query, "top_k": top_k}
        return self._post("search/rag", payload)
