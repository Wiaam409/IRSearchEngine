# pyrefly: ignore [missing-import]
import ir_datasets

datasets = [
    "beir/touche2020-train/v2",
]

for name in datasets:
    print(f"\nLoading {name}...")

    ds = ir_datasets.load(name)

    print("Docs:", ds.has_docs())
    print("Queries:", ds.has_queries())
    print("Qrels:", ds.has_qrels())

    doc = next(ds.docs_iter())

    print("Sample ID:", doc.doc_id)