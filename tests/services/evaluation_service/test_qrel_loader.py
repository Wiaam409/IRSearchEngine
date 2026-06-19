import os
import unittest
from services.evaluation_service.infrastructure.qrel_loader import QrelFileLoader

class TestQrelLoader(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_qrels.txt"
        with open(self.test_file, "w") as f:
            f.write("q1 0 d1 1\n")
            f.write("q1 0 d2 2\n")
            f.write("q2 0 d1 0\n")
            
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load(self):
        loader = QrelFileLoader()
        qrels = loader.load(self.test_file)
        self.assertEqual(len(qrels), 3)
        self.assertEqual(qrels[0].query_id, "q1")
        self.assertEqual(qrels[0].doc_id, "d1")
        self.assertEqual(qrels[0].relevance_score, 1)

if __name__ == '__main__':
    unittest.main()
