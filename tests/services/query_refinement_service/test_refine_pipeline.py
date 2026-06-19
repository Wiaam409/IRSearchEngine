import unittest
from typing import List
from services.query_refinement_service.domain.interfaces import ISynonymProvider, ISpellChecker
from services.query_refinement_service.application.use_cases import RefineQueryUseCase

class MockSpell(ISpellChecker):
    def correct(self, term: str) -> str:
        return "سيارة" if term == "سيار" else term

class MockSyn(ISynonymProvider):
    def get_synonyms(self, term: str) -> List[str]:
        return ["مركبة"] if term == "سيارة" else []

class TestRefinePipeline(unittest.TestCase):
    def test_pipeline_ordering(self):
        use_case = RefineQueryUseCase(MockSpell(), MockSyn())
        res = use_case.refine("سيار")
        
        self.assertEqual(res.original_text, "سيار")
        self.assertIn("سيار", res.corrections)
        self.assertEqual(res.corrections["سيار"], "سيارة")
        self.assertEqual(res.refined_text, "(سيارة OR مركبة)")
        self.assertIn("مركبة", res.suggestions)

    def test_empty_query(self):
        use_case = RefineQueryUseCase(MockSpell(), MockSyn())
        res = use_case.refine("   ")
        self.assertEqual(res.refined_text, "   ")

if __name__ == '__main__':
    unittest.main()
