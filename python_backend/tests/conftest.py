import sys
import os

# Add python_backend/ to sys.path so that `from main import app`, `from vision_engine import ...`,
# etc. resolve correctly when pytest is run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
