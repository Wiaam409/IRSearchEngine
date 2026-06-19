import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.adapters.beir_adapter import BeirAdapter

beir = BeirAdapter()

print(next(beir.load_documents()))