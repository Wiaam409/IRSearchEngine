import os
import argparse

from services.preprocessing_service.infrastructure.stopwords_loader import FileStopwordLoader
from services.preprocessing_service.application.bengali_processors import (
    BengaliNormalizer,
    BengaliTokenizer,
    BengaliStopwordRemover
)
from services.preprocessing_service.application.pipeline import PreprocessingPipeline

def main():
    parser = argparse.ArgumentParser(description="Bengali Preprocessing CLI Demo")
    parser.add_argument("text", type=str, help="Bengali text to process")
    args = parser.parse_args()

    # Determine paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(current_dir, "..", "infrastructure", "stopwords", "bengali_stopwords.txt")

    # Dependency Injection Setup
    loader = FileStopwordLoader()
    try:
        stopwords = loader.load(stopwords_path)
    except FileNotFoundError:
        print(f"Warning: Stopwords file not found at {stopwords_path}. Using empty set.")
        stopwords = set()

    normalizer = BengaliNormalizer()
    tokenizer = BengaliTokenizer()
    stopword_remover = BengaliStopwordRemover(stopwords)

    pipeline = PreprocessingPipeline(
        normalizer=normalizer,
        tokenizer=tokenizer,
        stopword_remover=stopword_remover
    )

    # Process
    tokens = pipeline.process(args.text)
    print(f"Input: {args.text}")
    print(f"Tokens: {tokens}")

if __name__ == "__main__":
    main()
