import sys
from pathlib import Path

# scripts/ isn't part of the installed `src*` package (see pyproject.toml),
# so tests that need to import a script module (e.g. promote_if_better.py)
# need it on sys.path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
