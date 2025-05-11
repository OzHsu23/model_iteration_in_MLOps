# app/globals.py

import os
import tempfile
from pathlib import Path

# Default setting path
SETTING_PATH: str = os.getenv("SETTING_PATH")

# Package dir for model zipping and evaluation
PACKAGE_DIR = Path.home() / "model_iteration_in_MLOps" / "fastapi_server"
if not os.path.exists(PACKAGE_DIR):
    os.makedirs(PACKAGE_DIR, exist_ok=True)