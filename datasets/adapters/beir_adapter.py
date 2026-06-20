# pyrefly: ignore [missing-import]
import ir_datasets

class BeirAdapter:

    def __init__(self):
        self.dataset = ir_datasets.load(
            "beir/webis-touche2020/v2"
        )

    def load_documents(self):
        return self.dataset.docs_iter()

    def load_queries(self):
        return self.dataset.queries_iter()

    def load_qrels(self):
        return self.dataset.qrels_iter()

    def get_document_text(self, doc_id: str) -> str:
        if not hasattr(self, '_docs_store'):
            self._docs_store = self.dataset.docs_store()
        try:
            doc = self._docs_store.get(doc_id)
            return getattr(doc, 'text', str(doc))
        except KeyError:
            return ""