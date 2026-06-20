import sqlite3
import os
import sys
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.adapters.beir_adapter import BeirAdapter
from api.config import settings

def seed_database():
    db_path = os.path.join(project_root, "documents.db")
    print(f"Creating database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT
        )
    ''')
    
    # Load dataset
    print("Loading BEIR dataset...")
    adapter = BeirAdapter()
    docs_iter = adapter.load_documents()
    
    # Insert documents
    print("Seeding documents into SQLite...")
    
    # We use a transaction and executemany for high performance
    batch_size = 10000
    batch = []
    
    # Count docs for tqdm if possible. Touche2020 has 382,545 docs
    for doc in tqdm(docs_iter, total=382545):
        # Touche2020 documents don't always have a title attribute, some have 'title', some don't.
        # ir_datasets GenericDoc has doc_id and text.
        title = getattr(doc, 'title', '')
        text = getattr(doc, 'text', str(doc))
        doc_id = doc.doc_id
        
        batch.append((doc_id, title, text))
        
        if len(batch) >= batch_size:
            cursor.executemany('INSERT OR IGNORE INTO documents (doc_id, title, text) VALUES (?, ?, ?)', batch)
            conn.commit()
            batch = []
            
    # Insert remaining
    if batch:
        cursor.executemany('INSERT OR IGNORE INTO documents (doc_id, title, text) VALUES (?, ?, ?)', batch)
        conn.commit()
        
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)')
    conn.commit()
    conn.close()
    
    print("Database seeding complete!")

if __name__ == "__main__":
    seed_database()
