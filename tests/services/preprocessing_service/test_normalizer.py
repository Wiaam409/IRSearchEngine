import unittest
import unicodedata
from services.preprocessing_service.application.arabic_processors import ArabicNormalizer

class TestArabicNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = ArabicNormalizer()

    def test_normalize_removes_punctuation(self):
        # Text with English and Arabic punctuation
        text = "مرحبا, العالم! كيف حالك؟ نعم، جيد؛"
        expected = "مرحبا العالم كيف حالك نعم جيد"
        
        # Replace multiple spaces with single space for reliable testing
        result = " ".join(self.normalizer.normalize(text).split())
        self.assertEqual(result, expected)

    def test_normalize_keeps_arabic_characters(self):
        text = "أنا أحب اللغة العربية"
        expected = unicodedata.normalize('NFKC', text)
        self.assertEqual(self.normalizer.normalize(text), expected)

if __name__ == '__main__':
    unittest.main()
