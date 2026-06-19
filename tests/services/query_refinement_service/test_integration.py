import os
import unittest
from services.query_refinement_service.infrastructure.file_synonym_provider import FileBasedSynonymProvider
from services.query_refinement_service.infrastructure.spell_checkers import NoOpSpellChecker
from services.query_refinement_service.application.use_cases import RefineQueryUseCase

class TestRefinementIntegration(unittest.TestCase):
    def test_integration_with_fixture(self):
        fixture_path = os.path.join("tests", "fixtures", "synonyms.json")
        self.assertTrue(os.path.exists(fixture_path), f"Fixture not found at {fixture_path}")
        
        provider = FileBasedSynonymProvider(fixture_path)
        checker = NoOpSpellChecker()
        use_case = RefineQueryUseCase(checker, provider)
        
        res = use_case.refine("تعليم")
        self.assertEqual(res.original_text, "تعليم")
        self.assertTrue("تربية" in res.refined_text)
        self.assertTrue("تدريس" in res.refined_text)
        self.assertEqual(len(res.suggestions), 2)

if __name__ == '__main__':
    unittest.main()
