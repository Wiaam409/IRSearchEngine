import os
import json
import sqlite3
import synnamon

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "beir", "webis-touche2020", "v2", "synonyms.json"))

print("Extracting offline synonyms from synnamon package...")
db_path = os.path.join(os.path.dirname(synnamon.__file__), "data", "en_thesaurus.db")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT word, synonyms FROM thesaurus")
    
    synonyms_dict = {}
    for word, syns_str in cursor.fetchall():
        if word and syns_str:
            # Synonyms are stored as comma-separated values in the DB
            syns_list = [s.strip() for s in syns_str.split(',') if s.strip()]
            if syns_list:
                synonyms_dict[word] = syns_list
                
    conn.close()
    
    print(f"Extracted {len(synonyms_dict)} words from offline database.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(synonyms_dict, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved comprehensive synonyms to {output_path}")

except Exception as e:
    print(f"Error extracting from DB: {e}")
