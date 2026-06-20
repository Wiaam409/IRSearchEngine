import sqlite3
from typing import List, Dict, Any

class DatabaseDocumentStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_documents(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetches the documents from the SQLite database by their IDs.
        Returns a dictionary mapping doc_id to a dict containing 'title' and 'text'.
        """
        if not doc_ids:
            return {}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prepare the query with the exact number of placeholders
        placeholders = ','.join(['?'] * len(doc_ids))
        query = f"SELECT doc_id, title, text FROM documents WHERE doc_id IN ({placeholders})"
        
        cursor.execute(query, doc_ids)
        rows = cursor.fetchall()
        
        conn.close()
        
        results = {}
        for row in rows:
            doc_id, title, text = row
            results[doc_id] = {
                "title": title,
                "text": text
            }
            
        return results

    def get_document_text(self, doc_id: str) -> str:
        """Compatibility method for older services like RAG that just need text"""
        docs = self.get_documents([doc_id])
        if doc_id in docs:
            return docs[doc_id]["text"]
        return ""
