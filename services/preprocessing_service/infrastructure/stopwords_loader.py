import os
from typing import Set

class FileStopwordLoader:
    """Infrastructure component to load stopwords from a text file."""
    
    def load(self, filepath: str) -> Set[str]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stopwords file not found at {filepath}")
            
        stopwords = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        return stopwords
