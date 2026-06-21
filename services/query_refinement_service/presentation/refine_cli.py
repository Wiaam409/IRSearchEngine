import os
import argparse
import json

from services.query_refinement_service.domain.models import RefinedQuery
from services.query_refinement_service.infrastructure.file_synonym_provider import FileBasedSynonymProvider
from services.query_refinement_service.infrastructure.spell_checkers import SimpleSpellChecker, NoOpSpellChecker
from services.query_refinement_service.application.use_cases import RefineQueryUseCase

def main():
    parser = argparse.ArgumentParser(description="Query Refinement CLI Demo")
    parser.add_argument("--synonyms", type=str, default="datasets/beir/webis-touche2020/v2/synonyms.json", help="Path to synonyms JSON file")
    parser.add_argument("--spell", action="store_true", help="Enable basic spell checking")
    args = parser.parse_args()

    print(f"Loading synonyms from {args.synonyms}...")
    synonym_provider = FileBasedSynonymProvider(args.synonyms)

    if args.spell:
        mock_dictionary = {"سيارة", "تعليم", "مدرسة", "جامعة"}
        spell_checker = SimpleSpellChecker(mock_dictionary, max_distance=1)
        print("Spell checking enabled (mock dictionary).")
    else:
        spell_checker = NoOpSpellChecker()
        
    refine_use_case = RefineQueryUseCase(spell_checker, synonym_provider)

    print("\n--- Query Refinement System Ready ---")
    while True:
        try:
            query = input("\nEnter raw query (or Ctrl+C to exit): ").strip()
            if not query:
                continue
                
            result: RefinedQuery = refine_use_case.refine(query)
            
            print(f"\n[Original] : {result.original_text}")
            
            if result.corrections:
                print(f"[Corrected]: {result.corrections}")
                
            print(f"[Refined]  : {result.refined_text}")
            
            if result.suggestions:
                print(f"[Suggests] : {', '.join(result.suggestions)}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
