# dags/globals.py

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MONITORING_CONFIG = os.path.join(BASE_DIR, "configs", "monitoring_config.json")
