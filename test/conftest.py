# Ensure src is on sys.path for imports in tests
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
# Ensure repository root is on sys.path so absolute imports like `src.*` resolve
if ROOT.exists():
    sys.path.insert(0, str(ROOT))
