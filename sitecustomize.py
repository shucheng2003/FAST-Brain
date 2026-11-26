from pathlib import Path
import sys

root = Path(__file__).resolve().parent
src = str(root / "src")
if src not in sys.path:
    sys.path.insert(0, src)
