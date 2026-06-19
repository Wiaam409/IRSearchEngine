from abc import ABC, abstractmethod

class BaseDatasetAdapter(ABC):

    @abstractmethod
    def load_documents(self):
        pass

    @abstractmethod
    def load_queries(self):
        pass

    @abstractmethod
    def load_qrels(self):
        pass