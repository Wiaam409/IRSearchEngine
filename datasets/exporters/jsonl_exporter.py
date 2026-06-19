import json
from pathlib import Path


class JsonlExporter:

    @staticmethod
    def export_documents(documents, output_path):

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w", encoding="utf-8") as f:

            for doc in documents:

                record = {
                    "doc_id": doc.doc_id,
                    "title": getattr(doc, "title", ""),
                    "text": getattr(doc, "text", "")
                }

                f.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

    @staticmethod
    def export_queries(queries, output_path):

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w", encoding="utf-8") as f:

            for query in queries:

                record = {
                    "query_id": query.query_id,
                    "text": query.text
                }

                f.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

    @staticmethod
    def export_qrels(qrels, output_path):

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w", encoding="utf-8") as f:

            for qrel in qrels:

                record = {
                    "query_id": qrel.query_id,
                    "doc_id": qrel.doc_id,
                    "relevance": qrel.relevance
                }

                f.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )