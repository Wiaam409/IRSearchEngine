import ir_datasets

for name in sorted(ir_datasets.registry):
    try:
        ds = ir_datasets.load(name)
        has_docs = hasattr(ds, 'docs_count') and ds.docs_count() is not None
        has_queries = hasattr(ds, 'queries_iter')
        has_qrels = hasattr(ds, 'qrels_iter')
        if has_docs and has_queries and has_qrels:
            count = ds.docs_count()
            if 200_000 <= count <= 500_000:
                print(f"{name}: {count:,} docs")
    except Exception:
        continue
