import unittest
from services.query_refinement_service.infrastructure.spell_checkers import SimpleSpellChecker, NoOpSpellChecker

class TestSpellCheckers(unittest.TestCase):
    def test_noop(self):
        checker = NoOpSpellChecker()
        self.assertEqual(checker.correct("wrong"), "wrong")

    def test_simple_spell_checker_correct(self):
        vocab = {"hello", "world", "سيارة"}
        checker = SimpleSpellChecker(vocab, max_distance=1)
        
        self.assertEqual(checker.correct("hello"), "hello")
        self.assertEqual(checker.correct("helo"), "hello")
        self.assertEqual(checker.correct("he"), "he")

    def test_arabic_correction(self):
        vocab = {"سيارة", "تعليم"}
        checker = SimpleSpellChecker(vocab, max_distance=1)
        self.assertEqual(checker.correct("سيار"), "سيارة")

if __name__ == '__main__':
    unittest.main()
