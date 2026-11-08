import sys
from pathlib import Path

# Ensure F:\POLAR-FLASH is the first entry in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)
