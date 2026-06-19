import os
import json
import unittest
from services.query_refinement_service.infrastructure.file_synonym_provider import FileBasedSynonymProvider

class TestSynonyms(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_syns.json"
        with open(self.test_file, "w", encoding="utf-8") as f:
            json.dump({"سيارة": ["مركبة", "عربة"]}, f)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_get_synonyms(self):
        provider = FileBasedSynonymProvider(self.test_file)
        res = provider.get_synonyms("سيارة")
        self.assertEqual(len(res), 2)
        self.assertIn("مركبة", res)

    def test_missing_word(self):
        provider = FileBasedSynonymProvider(self.test_file)
        self.assertEqual(provider.get_synonyms("unknown"), [])

if __name__ == '__main__':
    unittest.main()
