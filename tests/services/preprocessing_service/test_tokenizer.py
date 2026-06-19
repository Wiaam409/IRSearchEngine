import unittest
from services.preprocessing_service.application.arabic_processors import ArabicTokenizer

class TestArabicTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ArabicTokenizer()

    def test_tokenize_splits_by_whitespace(self):
        text = "أنا أحب اللغة العربية"
        expected = ["أنا", "أحب", "اللغة", "العربية"]
        self.assertEqual(self.tokenizer.tokenize(text), expected)

    def test_tokenize_handles_extra_spaces(self):
        text = "أنا    أحب   اللغة"
        expected = ["أنا", "أحب", "اللغة"]
        self.assertEqual(self.tokenizer.tokenize(text), expected)

if __name__ == '__main__':
    unittest.main()
