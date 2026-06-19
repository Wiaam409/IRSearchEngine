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