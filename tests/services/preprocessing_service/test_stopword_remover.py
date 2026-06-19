import unittest
from services.preprocessing_service.application.arabic_processors import ArabicStopwordRemover

class TestArabicStopwordRemover(unittest.TestCase):
    def setUp(self):
        self.stopwords = {"في", "من", "على"}
        self.remover = ArabicStopwordRemover(self.stopwords)

    def test_remove_filters_stopwords(self):
        tokens = ["أنا", "في", "البيت", "من", "فضلك"]
        expected = ["أنا", "البيت", "فضلك"]
        self.assertEqual(self.remover.remove(tokens), expected)

    def test_remove_keeps_all_if_no_stopwords_present(self):
        tokens = ["أنا", "أحب", "اللغة", "العربية"]
        expected = ["أنا", "أحب", "اللغة", "العربية"]
        self.assertEqual(self.remover.remove(tokens), expected)

if __name__ == '__main__':
    unittest.main()
