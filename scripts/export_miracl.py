
from datasets.adapters.beir_adapter import BeirAdapter
import json

adapter = BeirAdapter()

with open("sample_docs.jsonl", "w", encoding="utf-8") as f:

    for i, doc in enumerate(adapter.load_documents()):

        record = {
            "doc_id": doc.doc_id,
            "title": getattr(doc, 'title', ''),
            "text": getattr(doc, 'text', str(doc))
        }

        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


print("Documents Exported")